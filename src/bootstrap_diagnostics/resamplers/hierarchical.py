from typing import Any, Self

import numpy as np
import numpy.typing as npt
import pandas as pd

from .base import Resampler

# The node in the hierarchy tree
HierarchNode = dict[Any, "HierarchNode"] | npt.NDArray


class HiearchicalResampler(Resampler):
    """
    Resampler that draws new samples.
    """

    def __init__(self, data_sample: pd.DataFrame, hierarchy: list[str]) -> None:
        super().__init__(data_sample)

        # Build the hierarchy tree
        self._hierarchy = hierarchy
        self._hierarchy_tree = self._build_hierarchy(self._hierarchy)

    def _build_hierarchy(self, hierarchy: list[str]) -> HierarchNode:
        if not isinstance(self._data_sample, pd.DataFrame):
            raise TypeError("HierarchicalResampler requires a pandas DataFrame")

        tree = {}

        for keys, idx in self._data_sample.groupby(
            hierarchy, sort=False
        ).groups.items():
            # Traverse the tree
            node = tree

            # Given hierarchy like ["A"], keys will be an int not a tuple
            if not isinstance(keys, tuple):
                keys = (keys,)

            # Add this so type checker stops complaining
            for key in keys[:-1]:
                node = node.setdefault(key, {})

            node[keys[-1]] = idx.to_numpy()

        return tree

    def draw_sample(
        self,
        rng: np.random.Generator,
    ) -> pd.DataFrame:
        row_idxs = []

        def build_sample(node: HierarchNode | npt.NDArray) -> None:
            if not isinstance(node, dict):
                # Node carries the rows of the cases
                row_idxs.append(node)
                return
            else:
                keys = list(node.keys())
                for key in rng.choice(keys, len(keys), replace=True):  # ty:ignore[no-matching-overload]
                    build_sample(node[key])

        build_sample(self._hierarchy_tree)

        idxs = np.concat(row_idxs)

        return self._data_sample.iloc[idxs]  # ty:ignore[unresolved-attribute]

    def with_data(
        self,
        new_data_sample: pd.DataFrame,
    ) -> Self:
        return type(self)(new_data_sample, self._hierarchy)
