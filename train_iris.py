# %% Thing

import numpy as np

from decision_tree import (
    MeasureExecutionTime,
    TreeBuilder,
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
    min_samples=1,
    max_depth=10,
)

# Build tree
with MeasureExecutionTime("Build decision tree"):
    tree = tree_builder.build_decision_tree(train_data)

print(tree)

# %% Evaluate

predictions = [tree.predict(sample) for sample in test_data]
true_labels = test_data[:, -1]
accuracy = np.mean(np.array(predictions) == true_labels)

print(f"Accuracy: {accuracy:.4f}")

# %% Show a few predictions

print("Sample predictions vs actual:")
for i in range(min(10, len(test_data))):
    pred_label = target_classes[int(predictions[i])]
    true_label = target_classes[int(true_labels[i])]
    print(f"  Pred: {pred_label} | True: {true_label}")

# %% Train random forest

np.random.seed(0)

with MeasureExecutionTime("Build random forest"):
    forest = tree_builder.build_random_forest(
        train_data,
        trees_count=10,
        features_per_split=int(np.sqrt(dataset.shape[1])),
    )

# %% Compare accuracies
tree_predictions = [tree.predict(sample) for sample in test_data]
tree_accuracy = np.mean(np.array(tree_predictions) == true_labels)

forest_predictions = [forest.predict(sample) for sample in test_data]
forest_accuracy = np.mean(np.array(forest_predictions) == true_labels)

print(f"Decision Tree Accuracy: {tree_accuracy:.4f}")
print(f"Random Forest Accuracy: {forest_accuracy:.4f}")

if forest_accuracy > tree_accuracy:
    print("Random Forest wins!")
elif tree_accuracy > forest_accuracy:
    print("Decision Tree wins!")
else:
    print("It's a tie!")

# %% Show a few forest predictions

print("\nRandom Forest sample predictions vs actual:")
for i in range(min(10, len(test_data))):
    pred_label = target_classes[int(forest_predictions[i])]
    true_label = target_classes[int(true_labels[i])]
    print(f"  Pred: {pred_label} | True: {true_label}")
