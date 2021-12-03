import cv2
import numpy as np

from src.gradient import Gradient, dx, dy
from src.lab_converter import LabImage


class AdaptiveGrayscaleImage(LabImage):
    def __init__(self, image, w=1.8, eps=1e-3):
        super().__init__(image, w, eps)
        self.grad = Gradient(self.lab_image).compute()
        self.corrected = False

    def correct_inconsistencies(self):
        if self.corrected:
            return
        h, w, _ = self.lab_image.shape

        while True:
            max_err = 0

            for y in range(1, w - 1):
                for x in range(1, h - 1):
                    err = self.grad[x][y][dx] + self.grad[x + 1][y][dy] \
                          - self.grad[x][y][dy] - self.grad[x][y + 1][dx]
                    if abs(err) > max_err:
                        max_err = abs(err)

                    s = err * self.w * .25

                    self.grad[x][y][dx] = -s + self.grad[x][y][dx]
                    self.grad[x + 1][y][dy] = -s + self.grad[x + 1][y][dy]
                    self.grad[x][y][dy] = s + self.grad[x][y][dy]
                    self.grad[x][y + 1][dx] = s + self.grad[x][y + 1][dx]

            if max_err < self.eps:
                break

        self.corrected = True

    def integrate(self):
        self.correct_inconsistencies()

        h, w, _ = self.lab_image.shape
        out = np.zeros((h, w))

        out[1][1] = 0

        for y in range(1, w):
            if y > 1:
                out[1][y] = out[1][y - 1] + self.grad[1][y - 1][dy]

            for x in range(2, h):
                out[x][y] = out[x - 1][y] + self.grad[x - 1][y][dx]

        out = out + np.abs(np.min(out))
        cv2.normalize(out, out, 0, 255, cv2.NORM_MINMAX)

        return out
