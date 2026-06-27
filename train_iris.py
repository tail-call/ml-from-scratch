import mlflow
import mlflow.data.numpy_dataset as numpy_dataset
import numpy as np

from datasets import load_iris_dataset
from decision_tree import (
    ClassificationNodeFactory,
    MeasureExecutionTime,
    TreeBuilder,
    accuracy,
    features_of,
    gini_impurity,
    target_column_of,
    train_test_split,
)


def compare_prediction_with_true_labels(model, samples, target_classes):
    out = ""
    out += "Sample predictions vs actual:\n"
    for i in range(min(10, len(samples))):
        pred_label = target_classes[int(model.predict(samples[i]))]
        true_label = target_classes[int(samples[i, -1])]
        out += f"  Pred: {pred_label} | True: {true_label}\n"
    return out


# XXX This should be in datasets.py
class Data:
    def __init__(self) -> None:
        self.dataset = load_iris_dataset()
        self.target_classes = [0, 1, 2]
        self.train_split, self.test_split = train_test_split(
            self.dataset, test_proportion=0.2
        )
        self.mlflow_input = numpy_dataset.from_numpy(
            features=features_of(self.dataset),
            source="https://archive.ics.uci.edu/dataset/53/iris",
            targets=target_column_of(self.dataset),
            name="Iris Plants Database",
            # XXX I want to compute a digest, it seems fun
            # digest=compute_digest(self.dataset)
        )


def train_decision_tree(min_samples: int, max_depth: int):
    data = Data()
    mlflow.log_input(data.mlflow_input)

    mlflow.log_params(
        {
            "min_samples": min_samples,
            "max_depth": max_depth,
        }
    )

    tree_builder = TreeBuilder(
        impurity=gini_impurity,
        node_factory=ClassificationNodeFactory(),
        min_samples=min_samples,
        max_depth=max_depth,
    )

    # Build tree
    with MeasureExecutionTime("Build decision tree"):
        tree = tree_builder.build_decision_tree(data.train_split)

    mlflow.log_text(str(tree), artifact_file="model.txt")

    mlflow.log_metric("accuracy", accuracy(tree, data.test_split))

    mlflow.log_text(
        compare_prediction_with_true_labels(tree, data.test_split, data.target_classes),
        artifact_file="vs_true_labels.txt",
    )


def train_random_forest(min_samples: int, max_depth: int):
    data = Data()
    mlflow.log_input(data.mlflow_input)

    mlflow.log_params(
        {
            "min_samples": min_samples,
            "max_depth": max_depth,
        }
    )

    tree_builder = TreeBuilder(
        impurity=gini_impurity,
        node_factory=ClassificationNodeFactory(),
        min_samples=min_samples,
        max_depth=max_depth,
    )

    with MeasureExecutionTime("Build random forest"):
        forest = tree_builder.build_random_forest(
            data.train_split,
            trees_count=3,
            features_per_split=int(np.sqrt(data.dataset.shape[1])),
        )

    mlflow.log_text(str(forest), artifact_file="model.txt")

    mlflow.log_metric("accuracy", accuracy(forest, data.test_split))

    mlflow.log_text(
        compare_prediction_with_true_labels(
            forest, data.test_split, data.target_classes
        ),
        artifact_file="vs_true_labels.txt",
    )


if __name__ == "__main__":
    # XXX this should be in a config or somewhere like that
    mlflow.set_tracking_uri("http://192.168.0.103:5000")
    mlflow.set_experiment("DecisionTreeIris")
    with mlflow.start_run():
        np.random.seed(0)
        train_decision_tree(
            min_samples=1,
            max_depth=3,
        )
    mlflow.set_experiment("RandomForestIris")
    with mlflow.start_run():
        np.random.seed(0)
        train_random_forest(
            min_samples=1,
            max_depth=3,
        )
