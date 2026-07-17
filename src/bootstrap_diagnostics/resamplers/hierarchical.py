from typing import Any, Self

import numpy as np
import numpy.typing as npt
import pandas as pd

from .base import Resampler

# The node in the hierarchy tree
HierarchNode = dict[Any, "HierarchNode"] | npt.NDArray


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
        If ``hierarchy`` is empty.
    TypeError
        If ``data_sample`` is not a pandas DataFrame.
    """

    def __init__(
        self,
        data_sample: pd.DataFrame,
        hierarchy: list[tuple[str, bool]],
        observation_replacement: bool = False,
    ) -> None:
        super().__init__(data_sample)

        if not hierarchy:
            raise ValueError("Hierarchy must contain at least one level.")

        self._hierarchy = hierarchy
        self._observation_replacement = observation_replacement

        # Group columns and replacement strategies
        self._hierarchy_cols, self._group_replacement = zip(*hierarchy)

        # Build the hierarchy tree
        self._hierarchy_tree = self._build_hierarchy_tree()

    def _build_hierarchy_tree(self) -> HierarchNode:
        """
        Construct the hierarchy tree from the input DataFrame.

        Internal nodes correspond to grouping levels, while leaf nodes contain the
        row indices of the observations belonging to each terminal group.

        Returns
        -------
        HierarchNode
            Root node of the hierarchy tree.
        """
        if not isinstance(self._data_sample, pd.DataFrame):
            raise TypeError("HierarchicalResampler requires a pandas DataFrame")

        root = {}

        # Pandas >=3.0 has observed=True as default (can be problematic with categorical columns)
        grouped_data = self._data_sample.groupby(
            list(self._hierarchy_cols), sort=False, observed=True
        ).groups

        for keys, idx in grouped_data.items():
            # Traverse the tree
            node = root

            # Given hierarchy like ["A"], keys will be an int not a tuple
            if not isinstance(keys, tuple):
                keys = (keys,)

            for key in keys[:-1]:
                node = node.setdefault(key, {})

            node[keys[-1]] = idx.to_numpy()

        return root

    def draw_sample(
        self,
        rng: np.random.Generator,
    ) -> pd.DataFrame:
        """
        Draw a bootstrap sample according to the configured hierarchy.

        Groups are recursively traversed from the highest to the lowest level.
        At each level, groups are either resampled with replacement or visited
        exactly once according to the provided strategy.
        Observations within each terminal group may optionally be resampled with
        replacement.

        Returns
        -------
        pandas.DataFrame
            A bootstrap sample preserving the hierarchical structure implied by
            the resampling strategy.
        """
        row_idxs = []

        def build_sample(
            node: HierarchNode,
            depth: int = 0,
        ) -> None:
            if not isinstance(node, dict):
                if self._observation_replacement:
                    # Resample individual observations with replacement
                    row_idxs.append(rng.choice(node, len(node), replace=True))
                else:
                    # Add all observations
                    row_idxs.append(node)
                return

            keys = list(node.keys())

            # Use the strategy to determine if we should resample keys with replacement
            if self._group_replacement[depth]:
                key_idx = rng.integers(len(keys), size=len(keys))
                resampled_keys = [keys[i] for i in key_idx]
            else:
                resampled_keys = keys

            for key in resampled_keys:
                build_sample(node[key], depth + 1)  # ty:ignore[invalid-argument-type]

        build_sample(self._hierarchy_tree)

        # Concatenate the sampled row indices
        idxs = np.concatenate(row_idxs)

        # Reset the index so it doesn't lead to problems down the line
        return self._data_sample.iloc[idxs].reset_index(drop=True)  # ty:ignore[unresolved-attribute]

    def with_data(
        self,
        new_data_sample: pd.DataFrame,
    ) -> Self:
        return type(self)(
            new_data_sample,
            hierarchy=self._hierarchy,
            observation_replacement=self._observation_replacement,
        )
