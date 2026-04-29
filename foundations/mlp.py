import numpy as np
from numpy.typing import NDArray
from typing import List


class Solution:
    def forward(self, x: NDArray[np.float64], weights: List[NDArray[np.float64]], biases: List[NDArray[np.float64]]) -> NDArray[np.float64]:
        # x: 1D input array
        # weights: list of 2D weight matrices
        # biases: list of 1D bias vectors
        # Apply ReLU after each hidden layer, no activation on output layer
        # return np.round(your_answer, 5)
        x = np.array(x)
        a = x
        for W,b in zip(weights, biases):
            W = np.array(W)
            b = np.array(b)
            z = a @ W + b
            a = np.maximum(0, z)

        return np.round(a, 5)