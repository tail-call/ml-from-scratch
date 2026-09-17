from math import exp

MINUS_INFINITY = float('-inf')

def softmax(items: list[float]) -> list[float]:
    exps = [exp(item) for item in items]
    sum_exps = sum(exps)
    return [exp / sum_exps for exp in exps]

def safe_sotmax_online_normalization(items: list[float]) -> list[float]:
    "Taken from https://arxiv.org/pdf/1805.02867"
    max_value = MINUS_INFINITY
    divisor = 0.0
    for item in items:
        prev_max_value = max_value
        max_value = max(max_value, item)
        divisor = divisor * exp(prev_max_value - max_value) + exp(item - max_value)

    return [exp(item - max_value) / divisor for item in items]


class OnlineSoftmaxer:
    """
    Computes softmax for a list of floats arriving in parts.

    A similar algorithm is used in FlashAttention to avoid
    extra HBM <-> SRAM I/O. It is applied to blocks of `QK^T/(d**(-2))`
    matrix so each block can stay in SRAM and a materialization of
    the entire `QK^T/(d**(-2))` matrix in HBM is not necessary.
    """

    def __init__(self) -> None:
        self.acc_max: float = MINUS_INFINITY
        self.acc_sum: float = 0.0
        self.acc_items: list[float] = []

    def feed(self, items: list[float]):
        max_item = max(items)
        exp_sum = sum([exp(item - max_item) for item in items])

        old_max = self.acc_max
        self.acc_max = max(old_max, max_item)
        self.acc_sum = (
            exp(old_max - self.acc_max) * self.acc_sum
            + exp(max_item - self.acc_max) * exp_sum
        )
        self.acc_items += items

    def pop(self) -> list[float]:
        return [
            exp(item - self.acc_max) / self.acc_sum
            for item in self.acc_items
        ]


def variance(items: list[float]) -> float:
    n = len(items)
    mean = sum(items) / n
    return sum([(item - mean)**2 for item in items]) / n

softmax([5,2,3,4,5])
softmaxer = OnlineSoftmaxer()
softmaxer.feed([5,2,3])
softmaxer.feed([4,5])
softmaxer.pop()

safe_sotmax_online_normalization([1,2,3,4,5])
variance([1,2,3,4,5])
