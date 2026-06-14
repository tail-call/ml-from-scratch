# %% Thing

import numpy as np

from decision_tree import (
    MeasureExecutionTime,
    TreeBuilder,
    classification_predictor,
    gini_impurity,
    train_test_split,
)


def load_iris_dataset() -> np.ndarray:
    return np.loadtxt(
        "data/iris/iris.data",
        delimiter=",",
        converters={
            4: lambda x: {
                "Iris-setosa": 0,
                "Iris-versicolor": 1,
                "Iris-virginica": 2,
            }[x]
        },
    )


dataset = load_iris_dataset()
target_classes = [0, 1, 2]

# %% Train decision tree

np.random.seed(0)

# Split data
train_data, test_data = train_test_split(dataset, test_proportion=0.2)

tree_builder = TreeBuilder(
    impurity=gini_impurity,
    predictor=classification_predictor,
    min_samples=1,
    max_depth=3,
)

# Build tree
with MeasureExecutionTime("Build decision tree"):
    tree = tree_builder.build_decision_tree(train_data)

print(tree)

# %% Evaluate


def accuracy(model, samples):
    predictions = [model.predict(sample) for sample in samples]
    true_labels = samples[:, -1]
    return np.mean(np.array(predictions) == true_labels)


def compare_prediction_with_true_labels(model, samples):
    print("Sample predictions vs actual:")
    for i in range(min(10, len(samples))):
        pred_label = target_classes[int(model.predict(samples[i]))]
        true_label = target_classes[int(samples[i, -1])]
        print(f"  Pred: {pred_label} | True: {true_label}")


print(f"Accuracy: {accuracy(tree, test_data):.4f}")
compare_prediction_with_true_labels(tree, test_data)

# %% Train random forest

np.random.seed(0)

with MeasureExecutionTime("Build random forest"):
    forest = tree_builder.build_random_forest(
        train_data,
        trees_count=3,
        features_per_split=int(np.sqrt(dataset.shape[1])),
    )

# %% Compare accuracies
tree_accuracy = accuracy(tree, test_data)
forest_accuracy = accuracy(forest, test_data)

print(f"Decision Tree Accuracy: {tree_accuracy:.4f}")
print(f"Random Forest Accuracy: {forest_accuracy:.4f}")

if forest_accuracy > tree_accuracy:
    print("Random Forest wins!")
elif tree_accuracy > forest_accuracy:
    print("Decision Tree wins!")
else:
    print("It's a tie!")

# %% Show a few forest predictions

compare_prediction_with_true_labels(forest, test_data)
