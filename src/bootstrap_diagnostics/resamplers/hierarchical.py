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
        If all columns in ``hierarchy`` aren't in the ``data_sample``.
    TypeError
        If ``data_sample`` is not a Pandas DataFrame.
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
            list(self._hierarchy_cols), sort=False, observed=True
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

    def _draw_indices(
        self,
        rng: np.random.Generator,
    ) -> npt.NDArray:
        """
        Generate bootstrap sample indices according to the configured hierarchy.

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
        npt.NDArray
            Array of row indices defining the bootstrap sample.
        """

        indices = []

        def build_sample(
            node: HierarchyNode,
            depth: int = 0,
        ) -> None:
            if not isinstance(node, dict):
                if self._observation_replacement:
                    # Resample individual observations with replacement
                    indices.append(rng.choice(node, len(node), replace=True))
                else:
                    # Add all observations
                    indices.append(node)
                return

            keys = list(node.keys())

            # Use the strategy to determine if we should resample keys with replacement
            if self._group_replacement[depth]:
                key_idx = rng.integers(
                    len(keys),
                    size=len(keys),
                )
                resampled_keys = [keys[i] for i in key_idx]
            else:
                resampled_keys = keys

            for key in resampled_keys:
                build_sample(node[key], depth + 1)  # ty:ignore[invalid-argument-type]

        build_sample(self._hierarchy_tree)

        # Concatenate the sampled row indices
        return np.concatenate(indices)

    def with_data(
        self,
        new_data_sample: pd.DataFrame,
    ) -> Self:
        return type(self)(
            new_data_sample,
            hierarchy=self._hierarchy,
            observation_replacement=self._observation_replacement,
        )
