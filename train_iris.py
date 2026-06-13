# %% Thing

import numpy as np

from decision_tree import (
    build_decision_tree,
    build_random_forest,
    gini_impurity,
    train_test_split,
)

features = {
    "Iris-setosa": 0,
    "Iris-versicolor": 1,
    "Iris-virginica": 2,
}

dataset = np.loadtxt(
    "iris/iris.data",
    delimiter=",",
    converters={4: lambda x: features[x]},
)

# %% Train decision tree

np.random.seed(0)

# Split data
train_data, test_data = train_test_split(dataset, test_proportion=0.2)

# Build tree
tree = build_decision_tree(
    train_data,
    impurity=gini_impurity,
    min_samples=2,
    max_depth=10,
)

tree

# %% Evaluate

predictions = [tree.classify(sample) for sample in test_data]
true_labels = test_data[:, -1]
accuracy = np.mean(np.array(predictions) == true_labels)

print(f"Accuracy: {accuracy:.4f}")

# %% Show a few predictions

print("Sample predictions vs actual:")
for i in range(min(10, len(test_data))):
    pred_label = list(features.keys())[int(predictions[i])]
    true_label = list(features.keys())[int(true_labels[i])]
    print(f"  Pred: {pred_label:15s} | True: {true_label:15s}")

# %% Train random forest

np.random.seed(0)

forest = build_random_forest(
    train_data,
    n_trees=10,
    impurity=gini_impurity,
    min_samples=2,
    max_depth=10,
    features_per_split=int(np.sqrt(dataset.shape[1])),
)

# %% Compare accuracies
tree_predictions = [tree.classify(sample) for sample in test_data]
tree_accuracy = np.mean(np.array(tree_predictions) == true_labels)

forest_predictions = [forest.classify(sample) for sample in test_data]
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
    pred_label = list(features.keys())[int(forest_predictions[i])]
    true_label = list(features.keys())[int(true_labels[i])]
    print(f"  Pred: {pred_label:15s} | True: {true_label:15s}")
