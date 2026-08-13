from typing import Any, Self

import numpy as np
import numpy.typing as npt
import pandas as pd

from .base import Resampler

# The node in the hierarchy tree
HierarchyNode = dict[Any, "HierarchyNode"] | npt.NDArray


class HierarchicalResampler(Resampler):
    """
    Resampler for hierarchical data.

    Observations are assumed to be organized into one or more nested grouping
    levels. The hierarchy is defined by a sequence of DataFrame columns, from
    the highest to the lowest level. At each level, groups may either be
    resampled with replacement or retained exactly once.

    For example, given the hierarchy

        [("school", True), ("classroom", False)]

    schools are sampled with replacement, while all classrooms belonging to
    each sampled school are retained.

    Once a terminal group is reached, observations may optionally be
    resampled with replacement.

    Parameters
    ----------
    data_sample : pd.DataFrame
        Dataset containing the observations and hierarchy columns.
    hierarchy : list[tuple[str, bool]]
        Sequence of ``(column, replace)`` pairs defining the hierarchy and
        replacement strategy from highest to lowest level.

        ``replace=True`` indicates that groups at that level are sampled with
        replacement, while ``False`` retains each group exactly once.
    observation_replacement : bool
        Whether observations within each terminal group are sampled with
        replacement. Defaults to ``False``.

    Raises
    ------
    ValueError
        If ``data_sample`` is empty.
        If ``hierarchy`` is empty.
        If ``hierarcy`` contains duplicate columns.
        If all columns in ``hierarchy`` aren't in the ``data_sample``.
    TypeError
        If ``data_sample`` is not a Pandas DataFrame.

    Notes
    -----
    Hierarchy columns are treated as group identifiers. Bootstrap samples relabel
    group occurrences so that repeated draws of the same original group are represented
    as distinct groups.

    Consequently, the hierarchy columns should not simultaneously be used as variables whose
    original labels are required by the statistic.
    """

    def __init__(
        self,
        data_sample: pd.DataFrame,
        hierarchy: list[tuple[str, bool]],
        observation_replacement: bool = False,
    ) -> None:
        if not isinstance(data_sample, pd.DataFrame):
            raise TypeError("HierarchicalResampler requires a Pandas DataFrame")
        if data_sample.empty:
            raise ValueError("data_sample should not be empty")

        # Not necessary but add this if Resampler changes in the future
        super().__init__(data_sample)

        self._data_sample: pd.DataFrame = data_sample

        if not hierarchy:
            raise ValueError("Hierarchy must contain at least one level")
        if any(len(pair) != 2 for pair in hierarchy):
            raise ValueError(
                "Hierarchy entries must be (column, replace) pairs"
            )

        self._hierarchy = hierarchy
        self._observation_replacement = observation_replacement

        # Group columns and replacement strategies
        self._hierarchy_cols, self._group_replacement = zip(*hierarchy)

        if len(set(self._hierarchy_cols)) != len(self._hierarchy_cols):
            raise ValueError("Columns defined in the hierarchy must be unique!")

        missing = set(self._hierarchy_cols) - set(self._data_sample.columns)
        if missing:
            raise ValueError(f"Missing hierarchy columns: {missing}")

        # Build the hierarchy tree
        self._hierarchy_tree = self._build_hierarchy_tree()

    def _build_hierarchy_tree(self) -> HierarchyNode:
        """
        Construct the hierarchy tree from the input DataFrame.

        Internal nodes correspond to grouping levels, while leaf nodes contain the
        row indices of the observations belonging to each terminal group. Note that
        rows whose hierarchy cols include missing values are excluded.

        Returns
        -------
        HierarchyNode
            Root node of the hierarchy tree.
        """
        root = {}

        # Pandas >=3.0 has observed=True as default (can be problematic with categorical columns)
        grouped_data = self._data_sample.groupby(
            list(self._hierarchy_cols),
            sort=False,
            observed=True,
            dropna=True,
        ).indices

        for keys, idx in grouped_data.items():
            # Traverse the tree
            node = root

            # Given hierarchy like ["A"], keys will be an int not a tuple
            if not isinstance(keys, tuple):
                keys = (keys,)

            for key in keys[:-1]:
                node = node.setdefault(key, {})

            node[keys[-1]] = idx

        return root

    def draw_sample(self, rng: np.random.Generator) -> pd.DataFrame:
        """
        Generate a bootstrap sample according to the configured hierarchy.

        Groups are recursively traversed from the highest to the lowest level.
        At each level, groups are either resampled with replacement or visited
        exactly once according to the provided strategy.

        Once a terminal group is reached, observations are either retained or
        resampled with replacement depending on ``observation_replacement``. Note
        that this might produce a sample of a different size than the original.

        Parameters
        ----------
        rng : np.random.Generator
            NumPy random number generator used for all random operations.

        Returns
        -------
        pd.DataFrame

        """
        cases: list[pd.DataFrame] = []
        # Since we are doing nested resampling and might repeat a group,
        # we have to make sure that the group labels aren't duplicated when we
        # sample with replacement (so if we sample group A twice we don't run into a problem
        # of having one big group A but rather two relabeled groupd A_1, A_2). This
        # is very important for fitting hierarchical models.
        next_labels = [0] * len(self._hierarchy_cols)

        def build_sample(
            node: HierarchyNode,
            depth: int = 0,
            bootstrap_labels: tuple[int, ...] = (),
        ) -> None:
            if not isinstance(node, dict):
                if self._observation_replacement:
                    sampled_indices = rng.choice(
                        node, size=len(node), replace=True
                    )
                else:
                    sampled_indices = node

                sample = self._data_sample.iloc[sampled_indices].copy()

                # Relabel the groups
                for level, column in enumerate(self._hierarchy_cols):
                    sample[column] = bootstrap_labels[level]

                cases.append(sample)

                return

            keys = list(node.keys())

            # See whether the strategy requires resampling with replacement
            if self._group_replacement[depth]:
                selected_group_ids = rng.integers(
                    low=0,
                    high=len(keys),
                    size=len(keys),
                )
            else:
                selected_group_ids = np.arange(len(keys))

            for group_id in selected_group_ids:
                # Depth determines the hierarchy level
                new_group_label = next_labels[depth]
                # Increase the counter in order to prevent duplication of groups
                next_labels[depth] += 1

                build_sample(
                    node[keys[group_id]],
                    depth=depth + 1,
                    bootstrap_labels=bootstrap_labels + (new_group_label,),
                )

        build_sample(self._hierarchy_tree)

        # Create the new data frame and reindex the rows
        return pd.concat(cases, ignore_index=True)

    def with_data(
        self,
        new_data_sample: pd.DataFrame,
    ) -> Self:
        return type(self)(
            new_data_sample,
            hierarchy=self._hierarchy,
            observation_replacement=self._observation_replacement,
        )
