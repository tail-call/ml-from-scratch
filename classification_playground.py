import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


def generate_data() -> pd.DataFrame:
    np.random.seed(44)
    size_1, size_2 = 100, 50
    xs = np.concatenate([np.random.randn(size_1) + 3, np.random.randn(size_2) + 5])
    ys = np.concatenate([np.random.randn(size_1) - 4, np.random.randn(size_2) * 2 - 8])
    zs = np.concatenate([[0] * size_1, [1] * size_2])
    return pd.DataFrame({"xs": xs, "ys": ys, "zs": zs})


def sigmoid(x: np.ndarray) -> np.ndarray:
    return 1 / (1 + np.exp(-x))


def cross_entropy(zs: np.ndarray, zhats: np.ndarray) -> float:
    return -np.sum(zs * np.log(zhats) + (1 - zs) * np.log(1 - zhats))


def analytic_gradient_descent(w: np.ndarray, df: pd.DataFrame) -> np.ndarray:
    zhats = predict(w, df)
    error = zhats - df["zs"]
    X = np.column_stack((df["xs"], df["ys"], np.ones(len(df))))
    return X.T @ error  # shape (3,) — [dwx, dwy, db]


# df = generate_data()



def loss(w):
    zhats = predict(w, df)
    zs = df["zs"].to_numpy()
    return cross_entropy(zs, zhats)


def plot(w: np.ndarray, df: pd.DataFrame, acc: float):
    wx, wy, b = w
    # Boundary: wx*x + wy*y + b = 0 => y = (-wx*x - b) / wy
    x_line = np.linspace(df["xs"].min(), df["xs"].max(), 100)
    y_line = (-wx * x_line - b) / wy

    plt.figure(figsize=(8, 6))
    plt.scatter(
        df["xs"], df["ys"], c=df["zs"], cmap="bwr", alpha=0.6, edgecolors="none"
    )
    plt.plot(x_line, y_line, "k-", linewidth=2, label="Decision Boundary")
    plt.title(f"Logistic Regression — Accuracy: {acc:.0%}")
    plt.legend()
    plt.tight_layout()

class LogisticRegression:
    epochs: int = 200
    print_interval: int = 20
    learning_rate: float = 0.01

    def train(self, weights: np.ndarray, df: pd.DataFrame) -> np.ndarray:
        zs: np.ndarray = df["zs"].to_numpy()

        for i in range(self.epochs):
            weights -= self.learning_rate * analytic_gradient_descent(weights, df)
            if i % self.print_interval == 0:
                zhats = predict(weights, df)
                ce = cross_entropy(zs, zhats)
                acc = ((zhats >= 0.5).astype(int) == df["zs"]).mean()
                print(
                    f"Iter {i}: weights={weights} CE={ce:.3f} Acc={acc:.2f}"
                )

        return weights

    def predict(self, w: np.ndarray, df: pd.DataFrame):
        X = np.column_stack([df["xs"], df["ys"], np.ones(len(df))])
        return sigmoid(X @ w)

def train_eval_and_plot():
    df = generate_data()
    weights = np.array([0.0, 0.0, 0.0])
    zs: np.ndarray = df["zs"].to_numpy()

    lreg = LogisticRegression()
    weights = lreg.train(weights, df)

    wx, wy, b = weights
    zhats = lreg.predict(weights, df)
    preds = (zhats >= 0.5).astype(int)
    acc = (preds == df["zs"]).mean()
    ce = cross_entropy(zs, zhats)
    print(f"\nFinal: Accuracy={acc:.2f}, CE={ce:.3f}")
    print(f"Parameters: wx={wx:.4f}, wy={wy:.4f}, b={b:.4f}")
    plot(weights, df, acc)


if __name__ == "__main__":
    train_eval_and_plot()
