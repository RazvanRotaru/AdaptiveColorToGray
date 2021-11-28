# implementation of http://cadik.posvete.cz/color_to_gray/color_to_gray-cae07-fin.pdf

import argparse
import time

import cv2
import matplotlib.pyplot as plt
import numpy as np

from src.adaptive_gray_image import AdaptiveGrayscaleImage
from src.cie_y import CIEY

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Process some integers.')
    parser.add_argument('image', type=str, help='path to the RGB image')

    args = parser.parse_args()
    image_path = args.image

    t0 = time.time()
    print(f"Converting {image_path} to greyscale...")

    img_rgb = plt.imread(image_path)

    plt.imshow(img_rgb)
    plt.show()

    # TODO create comparison chart with default matplotlib greyscale version
    # TODO apply algorithm to all images in ./images
    # Should create a figure of 3 or 4 entries for each image in ./images

    img_grey = AdaptiveGrayscaleImage(img_rgb).integrate()
    # print(f"result {type(img_grey)} shape {img_grey.shape}")
    # print(f"min {np.min(img_grey)} max {np.max(img_grey)}")

    img_grey = img_grey + np.abs(np.min(img_grey))
    cv2.normalize(img_grey, img_grey, 0, 255, cv2.NORM_MINMAX)

    plt.imshow(img_grey, cmap='gray', vmin=0, vmax=255)
    plt.show()

    ciey = CIEY.convert(img_rgb)
    plt.imshow(ciey, cmap='gray')
    plt.show()

    tf = time.time() - t0
    print(f"Conversion done in {format(tf, '.2f')}s!")
