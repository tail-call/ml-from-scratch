# %% Thing

import numpy as np

from decision_tree import (
    build_decision_tree,
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

type(tree)  # should be DecisionNode or ValueNode
print("Sample predictions vs actual:")
for i in range(min(10, len(test_data))):
    pred_label = list(features.keys())[int(predictions[i])]
    true_label = list(features.keys())[int(true_labels[i])]
    print(f"  Pred: {pred_label:15s} | True: {true_label:15s}")

