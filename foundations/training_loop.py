import numpy as np
from numpy.typing import NDArray
from typing import Tuple


class Solution:
    def train(self, X: NDArray[np.float64], y: NDArray[np.float64], epochs: int, lr: float) -> Tuple[NDArray[np.float64], float]:
        # X: (n_samples, n_features)
        # y: (n_samples,) targets
        # epochs: number of training iterations
        # lr: learning rate
        #
        # Model: y_hat = X @ w + b
        # Loss: MSE = (1/n) * sum((y_hat - y)^2)
        # Initialize w = zeros, b = 0
        # return (np.round(w, 5), round(b, 5))
        X = np.array(X)
        y = np.array(y)
        w = np.zeros(X.shape[1]) # 1 is the feature column this menas one weight per feature col i think that is what linear regression needs for this example
        b = 0
        n = len(X)
        for epoch in range(epochs):
            # forward pass
            y_hat = X @ w + b
            # loss
            L = np.mean((y_hat-y)**2)
            # compute gradient of loss 
            dl_dw = (2/n) * X.T @ (y_hat - y)
            dl_db = 2 * np.mean(y_hat - y)
            # update
            w = w - lr * dl_dw
            b = b - lr * dl_db

        return np.round(w, 5), round(b, 5)