from typing import Any, Self, Sequence

import numpy as np
import numpy.typing as npt
import pandas as pd

from .base import Resampler

# The node in the hierarchy tree
HierarchNode = dict[Any, "HierarchNode"] | npt.NDArray


class HierarchicalResampler(Resampler):
    """
    Resampler that draws new samples according to the pre-defined hierarchy.
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
        self._hierarchy_tree = self._build_hierarchy(self._hierarchy_cols)

    def _build_hierarchy(self, hierarchy: Sequence[str]) -> HierarchNode:
        if not isinstance(self._data_sample, pd.DataFrame):
            raise TypeError("HierarchicalResampler requires a pandas DataFrame")

        tree = {}

        # Pandas >=3.0 has observed=True as default
        grouped_data = self._data_sample.groupby(
            hierarchy, sort=False, observed=True
        ).groups

        for keys, idx in grouped_data.items():
            # Traverse the tree
            node = tree

            # Given hierarchy like ["A"], keys will be an int not a tuple
            if not isinstance(keys, tuple):
                keys = (keys,)

            for key in keys[:-1]:
                node = node.setdefault(key, {})

            node[keys[-1]] = idx.to_numpy()

        return tree

    def draw_sample(
        self,
        rng: np.random.Generator,
    ) -> pd.DataFrame:
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

        # Concatenate the cases indeces
        idxs = np.concatenate(row_idxs)

        return self._data_sample.iloc[idxs]  # ty:ignore[unresolved-attribute]

    def with_data(
        self,
        new_data_sample: pd.DataFrame,
    ) -> Self:
        return type(self)(
            new_data_sample,
            hierarchy=self._hierarchy,
            observation_replacement=self._observation_replacement,
        )
