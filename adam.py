"""
Implementation of the Adam optimizer as described in:

Adam: a Method for Stochastic Optimization, ICLR 2015 paper,
Diederik P. Kingma, Jimmy Lei Ba, https://arxiv.org/abs/1412.6980
"""

from typing import Callable

import numpy as np


def compute_gradient(
    function: Callable[[np.ndarray], float],
    point: np.ndarray,
    epsilon: float,
) -> np.ndarray:
    result = np.zeros(len(point))

    for i in range(len(point)):
        delta_positive = np.copy(point)
        delta_positive[i] += epsilon / 2
        delta_negative = np.copy(point)
        delta_negative[i] -= epsilon / 2
        result[i] = (function(delta_positive) - function(delta_negative)) / epsilon

    return result


def adam(
    function: Callable[[np.ndarray], float],
    initial_parameters: np.ndarray,
    step_size: float = 0.001,
    max_steps: int = 1000,
    moment_1_decay_rate: float = 0.9,
    moment_2_decay_rate: float = 0.999,
    epsilon: float = 1e-8,
) -> np.ndarray:
    moment_vector_1st_biased = np.zeros(len(initial_parameters))
    moment_vector_2nd_biased = np.zeros(len(initial_parameters))
    time_step = 0
    new_parameters = initial_parameters
    prev_parameters = initial_parameters

    def converged():
        if time_step == 0:
            return False
        elif time_step >= max_steps:
            return True
        else:
            return np.max(prev_parameters - new_parameters) <= epsilon

    while not converged():
        time_step = time_step + 1
        gradient = compute_gradient(function, new_parameters, epsilon)

        # Compute biased moment vectors. They're biased towards zero
        # because we initialize moments with zeroes. We will un-bias them
        # right after
        moment_vector_1st_biased = (
            moment_1_decay_rate * moment_vector_1st_biased
            + (1 - moment_1_decay_rate) * gradient
        )
        moment_vector_2nd_biased = (
            moment_2_decay_rate * moment_vector_2nd_biased
            + (1 - moment_2_decay_rate) * gradient**2
        )

        # Now un-bias them
        moment_vector_1st = moment_vector_1st_biased / (
            1 - moment_1_decay_rate**time_step
        )
        moment_vector_2nd = moment_vector_2nd_biased / (
            1 - moment_2_decay_rate**time_step
        )

        # Update parameters
        new_parameters = new_parameters - step_size * moment_vector_1st / (
            np.sqrt(moment_vector_2nd) + epsilon
        )

    return new_parameters
