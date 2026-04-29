class Solution:
    def get_minimizer(self, iterations: int, learning_rate: float, init: int) -> float:
        # Objective function: f(x) = x^2
        # Derivative:         f'(x) = 2x
        # Update rule:        x = x - learning_rate * f'(x)
        # Round final answer to 5 decimal places
        i = 0
        while i < iterations:
            dfx = 2 * init
            init = init - learning_rate * dfx
            i += 1
        return round(init, 5)
