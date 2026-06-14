import time
from dataclasses import dataclass
from typing import Callable, Iterable, Self

import numpy as np


def train_test_split(
    data: np.ndarray, test_proportion: float = 0.2
) -> tuple[np.ndarray, np.ndarray]:
    assert test_proportion <= 1.0 and test_proportion >= 0

    data_copy = np.copy(data)
    np.random.shuffle(data_copy)
    test_count = round(len(data) * test_proportion)

    train_split = data_copy[test_count:]
    test_split = data_copy[:test_count]
    return train_split, test_split


def feature_indices_of(array: np.ndarray) -> range:
    return range(array.shape[1] - 1)


def target_column_of(array: np.ndarray) -> np.ndarray:
    return array[:, array.shape[1] - 1]


def gini_impurity(data: np.ndarray) -> float:
    if len(data) == 0:
        return 0.0

    counts = np.unique_counts(target_column_of(data)).counts

    return 1 - np.sum((counts / len(data)) ** 2)


def frequency(target: float, samples: np.ndarray) -> float:
    if len(samples) == 0:
        return 0.0

    uc = np.unique_counts(target_column_of(samples))
    if target in uc.values:
        d = list(uc.values).index(target)
        r = uc.counts[d] / len(samples)
        return r
    else:
        return 0.0


def class_distribution(samples: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Return (probability distribution, sorted unique class labels)."""
    if len(samples) == 0:
        return np.array([1.0]), np.array([0])
    uc = np.unique_counts(target_column_of(samples))
    return uc.counts / len(samples), uc.values


def mode(samples: np.ndarray) -> float:
    """Return the most frequent class label in the samples."""
    if len(samples) == 0:
        return 0.0
    uc = np.unique_counts(target_column_of(samples))
    return uc.values[np.argmax(uc.counts)]


def filter_rows(array: np.ndarray, fun: Callable[[np.ndarray], bool]):
    return array[np.apply_along_axis(fun, axis=1, arr=array)]


def splits_of(samples, feature_idx):
    for threshold in np.unique(samples[:, feature_idx]):
        l_cases = filter_rows(samples, lambda row: row[feature_idx] <= threshold)
        r_cases = filter_rows(samples, lambda row: row[feature_idx] > threshold)
        yield l_cases, r_cases, threshold


class MeasureExecutionTime:
    def __init__(self, title: str):
        self.title = title
        self.t1 = -1
        self.t2 = -1

    def __enter__(self):
        self.t1 = time.perf_counter()

    def __exit__(self, exc_type, exc_val, exc_traceback):
        self.t2 = time.perf_counter()
        print(f"Task {self.title} took {self.t2 - self.t1} seconds")


@dataclass
class ValueNode:
    class_distribution: np.ndarray
    classes: np.ndarray

    def classify(self, sample):
        return self.classes[self.class_distribution.argmax()]


@dataclass
class DecisionNode:
    left: ValueNode | Self
    right: ValueNode | Self
    feature_idx: int
    threshold: float

    def classify(self, sample):
        if sample[self.feature_idx] <= self.threshold:
            return self.left.classify(sample)
        else:
            return self.right.classify(sample)


@dataclass
class RandomForest:
    trees: list[DecisionNode | ValueNode]

    def classify(self, sample) -> float:
        """Predict class by majority vote across all trees."""
        votes = [tree.classify(sample) for tree in self.trees]
        counts = np.bincount(votes, minlength=len(set(votes)))
        return float(np.argmax(counts))


class TreeBuilder:
    """
    I build decision trees and random forests.
    """

    def __init__(
        self,
        impurity: Callable[[np.ndarray], float],
        min_samples: int = 1,
        max_depth: int = 10,
    ):
        self.impurity = impurity
        self.min_samples = min_samples
        self.max_depth = max_depth

    def build_value_node(self, samples):
        dist, cls = class_distribution(samples)
        return ValueNode(class_distribution=dist, classes=cls)

    def build_decision_tree(
        self,
        samples: np.ndarray,
        _depth=1,
        *,
        feature_indices: Iterable[int] | None = None,
    ):
        """
        Build a single decision tree from a `sample`. Optionally consider
        only specified `feature_indices` when splitting.
        """

        current_impurity = self.impurity(samples)
        samples_count = len(samples)

        if (
            samples_count < self.min_samples
            or self.impurity(samples) == 0.0
            or _depth >= self.max_depth
        ):
            return self.build_value_node(samples)

        max_information_gain = float("-inf")
        best_l_cases = None
        best_threshold = None
        best_feature_idx = None
        best_r_cases = None

        for feature_idx in feature_indices or feature_indices_of(samples):
            for l_cases, r_cases, threshold in splits_of(samples, feature_idx):
                information_gain = (
                    current_impurity
                    - len(l_cases) * self.impurity(l_cases) / samples_count
                    - len(r_cases) * self.impurity(r_cases) / samples_count
                )

                if information_gain > max_information_gain:
                    max_information_gain = information_gain
                    best_l_cases = l_cases
                    best_r_cases = r_cases
                    best_threshold = threshold
                    best_feature_idx = feature_idx

        if (
            best_l_cases is not None
            and best_r_cases is not None
            and best_feature_idx is not None
            and best_threshold is not None
        ):
            return DecisionNode(
                left=self.build_decision_tree(
                    best_l_cases,
                    feature_indices=feature_indices,
                    _depth=_depth + 1,
                ),
                right=self.build_decision_tree(
                    best_r_cases,
                    feature_indices=feature_indices,
                    _depth=_depth + 1,
                ),
                feature_idx=best_feature_idx,
                threshold=best_threshold,
            )

        # If all else fails
        return self.build_value_node(samples)

    def build_random_forest(
        self,
        samples: np.ndarray,
        trees_count: int = 10,
        *,
        features_per_split: int | None = None,
    ) -> RandomForest:
        """
        Build a random forest of decision trees from a `sample`, using
        the bagging (bootstrap aggregation) technique.

        Optionally consider only specified `feature_indices` when splitting.
        """

        features_count: int = samples.shape[1] - 1
        if features_per_split is None:
            features_per_split = features_count

        rng: np.random.Generator = np.random.default_rng()
        trees: list[DecisionNode | ValueNode] = []

        for _ in range(trees_count):
            bootstrap_idx: np.typing.NDArray[np.int64] = rng.choice(
                len(samples), size=len(samples), replace=True
            )
            bootstrapped_samples = samples[bootstrap_idx]

            if features_per_split < features_count:
                feature_indices = rng.choice(
                    features_count, size=features_per_split, replace=False
                ).tolist()
            else:
                feature_indices = feature_indices_of(samples)

            trees.append(
                self.build_decision_tree(
                    bootstrapped_samples,
                    feature_indices=feature_indices,
                )
            )

        return RandomForest(trees=trees)
