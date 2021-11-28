import numpy as np

dx = 0
dy = 1


class Gradient:
    # wa and wb should be in [0.2..0.6] interval
    def __init__(self, image, wa=0.2, wb=0.5):
        self.image = image
        self.wa = wa
        self.wb = wb

    def compute(self):
        image = self.image
        h, w, _ = image.shape
        grad = np.zeros((h, w, 2))

        for i in range(h):
            for j in range(w):
                x_up = 0 if i < 1 else image[i - 1][j]
                x_down = 0 if i >= h - 2 else image[i + 1][j]

                y_left = 0 if j < 1 else image[i][j - 1]
                y_right = 0 if j >= w - 2 else image[i][j + 1]

                grad[i][j][dx] = self._norm(x_down - x_up)
                grad[i][j][dy] = self._norm(y_right - y_left)

        return grad

    def _norm(self, delta):
        return np.cbrt(float(delta[0]) ** 3
                       + float(delta[1] * self.wa) ** 3
                       + float(delta[2] * self.wb) ** 3)
