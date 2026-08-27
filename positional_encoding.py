import matplotlib.pyplot as plt
import numpy as np


def positional_encoding(seq_len: int, d: int, n: int = 10000) -> np.ndarray:
    # Taken from https://machinelearningmastery.com/a-gentle-introduction-to-positional-encoding-in-transformer-models-part-1/
    # Modified to handle odd values of `d` (though it's hardly useful)
    P = np.zeros((seq_len, d))
    for k in range(seq_len):
        for i in np.arange(d):
            odd = i % 2
            denominator: float = np.power(n, (i - odd) / d)
            if odd:
                P[k, i] = np.cos(k / denominator)
            else:
                P[k, i] = np.sin(k / denominator)
    return P


P = positional_encoding(seq_len=4, d=5, n=100)
print(P)

# [[ 0.          1.          0.          1.          0.        ]
#  [ 0.84147098  0.54030231  0.15782664  0.98746684  0.02511622]
#  [ 0.90929743 -0.41614684  0.31169715  0.9501815   0.0502166 ]
#  [ 0.14112001 -0.9899925   0.45775455  0.88907861  0.07528529]]
