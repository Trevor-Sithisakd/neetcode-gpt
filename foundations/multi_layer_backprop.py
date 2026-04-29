import numpy as np
from typing import List


class Solution:
    def forward_and_backward(self,
                              x: List[float],
                              W1: List[List[float]], b1: List[float],
                              W2: List[List[float]], b2: List[float],
                              y_true: List[float]) -> dict:
        # Architecture: x -> Linear(W1, b1) -> ReLU -> Linear(W2, b2) -> predictions
        # Loss: MSE = mean((predictions - y_true)^2)
        #
        # Return dict with keys:
        #   'loss':  float (MSE loss, rounded to 4 decimals)
        #   'dW1':   2D list (gradient w.r.t. W1, rounded to 4 decimals)
        #   'db1':   1D list (gradient w.r.t. b1, rounded to 4 decimals)
        #   'dW2':   2D list (gradient w.r.t. W2, rounded to 4 decimals)
        #   'db2':   1D list (gradient w.r.t. b2, rounded to 4 decimals)
        x      = np.array(x)
        W1     = np.array(W1)
        b1     = np.array(b1)
        W2     = np.array(W2)
        b2     = np.array(b2)
        y_true = np.array(y_true)

        n = len(y_true)
        z1 = np.dot(x,W1.T) + b1
        a1 = np.maximum(0, z1) # has to be maximum since its elementwise comparision to numpy array
        z2 = np.dot(a1,W2.T) + b2
        y_pred = z2

        loss = (1/n) * np.sum((y_pred - y_true) ** 2)
        dl_dy = (2/n) * (y_pred - y_true) 
        dw2 = np.outer(dl_dy ,a1) # derived from chainrule dl/dw2 = dl/dy x dy/dw2 # a1 is the result passed onto the second neuron by the relu function
        db2 = dl_dy
        # compute the pass through 
        dl_da1 = np.dot(dl_dy,W2)  # pass the gradient back through w2 the transpose is what is going back through 
        # then pass it through the relu 
        dl_dz1 = dl_da1 * (z1 > 0)  
        dw1  = np.outer(dl_dz1,x)
        db1 = dl_dz1
        return {
        'loss':  round(loss, 4),
        'dW1':   (np.round(dw1, 4) + 0.0).tolist(),
        'db1':   (np.round(db1, 4) + 0.0).tolist(),
        'dW2':   (np.round(dw2, 4) + 0.0).tolist(),
        'db2':   (np.round(db2, 4) + 0.0).tolist()
    }
