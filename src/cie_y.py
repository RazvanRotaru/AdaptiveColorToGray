import numpy as np


class CIEY:
    R = 0
    G = 1
    B = 2

    @staticmethod
    def convert(image):
        h, w, _ = image.shape
        res = np.zeros((h, w))

        for i in range(h):
            for j in range(w):
                res[i][j] = 0.2989 * image[i][j][CIEY.R] \
                            + 0.587 * image[i][j][CIEY.G] \
                            + 0.072 * image[i][j][CIEY.B]

        return res
