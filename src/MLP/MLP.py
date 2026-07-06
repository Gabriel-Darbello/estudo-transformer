import numpy as np

class MLP:
    def __init__(self):
        self.ff1 = np.random.randn(6, 3)
        self.ff2 = np.random.randn(3, 6)

    def _apply_relu(self, matrix):
        for i in range(len(matrix)):
            for j in range(len(matrix[i])):
                if matrix[i][j] < 0:
                    matrix[i][j] = 0
        return np.array(matrix)


    def mlp_block(self, attention):
        h = self.ff1 @ attention
        h = self._apply_relu(h)
        y = self.ff2 @ h

        return np.array(y)

