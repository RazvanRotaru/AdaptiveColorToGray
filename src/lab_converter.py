import numpy as np


class LabImage:
    def __init__(self, image, w, eps):
        self.rgb_image = image
        self.w = w
        self.eps = eps

        self.lab_image = LabImage.img2lab(self.rgb_image)

    @staticmethod
    def img2lab(image):
        h, w, _ = image.shape
        res = np.zeros(image.shape).astype(np.float64)

        for i in range(h):
            for j in range(w):
                res[i][j] = LabImage.rgb2lab(*image[i][j])

        return res

    @staticmethod
    def rgb2lab(red, green, blue):
        r = red / 255.0
        g = green / 255.0
        b = blue / 255.0

        r = pow((r + 0.055) / 1.055, 2.4) if (r > 0.04045) else r / 12.92
        g = pow((g + 0.055) / 1.055, 2.4) if (g > 0.04045) else g / 12.92
        b = pow((b + 0.055) / 1.055, 2.4) if (b > 0.04045) else b / 12.92

        x = (r * 0.4124 + g * 0.3576 + b * 0.1805) / 0.95047
        y = (r * 0.2126 + g * 0.7152 + b * 0.0722) / 1.0
        z = (r * 0.0193 + g * 0.1192 + b * 0.9505) / 1.08883

        x = pow(x, 1. / 3.) if (x > 0.008856) else 7.7787 * x + 16. / 116.
        y = pow(y, 1. / 3.) if (y > 0.008856) else 7.7787 * y + 16. / 116.
        z = pow(z, 1. / 3.) if (z > 0.008856) else 7.7787 * z + 16. / 116.

        cieL = max(0., min(100., (116 * y) - 16))
        cieA = 500. * (x - y)
        cieB = 200. * (y - z)

        return cieL, cieA, cieB

    @staticmethod
    def _channel2lab(channel):
        c = channel / 255.0
        c = pow((c + 0.055) / 1.055, 4) if (c > 0.04045) else c / 12.92
        return c
