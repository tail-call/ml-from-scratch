import numpy as np
import pandas as pd

from datasets import air_quality_but_its_pandas
from decision_tree import (
    MeasureExecutionTime,
    RegressionNodeFactory,
    TreeBuilder,
    gini_impurity,
    mse,
    train_test_split,
)

tree_builder = TreeBuilder(
    impurity=gini_impurity,
    node_factory=RegressionNodeFactory(),
    max_depth=5,
)

df = air_quality_but_its_pandas()
# Move target column to the back (rquired by the data model)
df = pd.concat([df.iloc[:, 1:], df.iloc[:, :1]], axis=1)
# Cast to numpy
data = df.to_numpy()
# Smaller subset
data = data[:200]
train_data, test_data = train_test_split(data)

with MeasureExecutionTime("Build decision tree"):
    tree = tree_builder.build_decision_tree(train_data)
    print("MSE:", mse(tree, test_data))  # MSE: 0.3589967612064661


with MeasureExecutionTime("Build random forest"):
    forest = tree_builder.build_random_forest(train_data, trees_count=5)
    print("MSE:", mse(forest, test_data))  # MSE: 0.24319863356249272
