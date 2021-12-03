# implementation of http://cadik.posvete.cz/color_to_gray/color_to_gray-cae07-fin.pdf

import argparse
import os
import time

import cv2
import matplotlib.pyplot as plt
import numpy as np

from src.adaptive_gray_image import AdaptiveGrayscaleImage
from src.cie_y import CIEY

size = (256, 256)


def load_images_from_folder(folder):
    imgs = []
    for file in os.listdir(folder):
        file_path = os.path.join(folder, file)

        if not os.path.isfile(file_path):
            continue

        img = plt.imread(file_path)
        if img is not None:
            imgs.append((file, img))
    return imgs


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Process arguments.')
    parser.add_argument('--images', type=str, help='path to the images folder')
    parser.add_argument('--save_output', action='store_true', help='save output images in ./images/out')

    args = parser.parse_args()
    dir_path = args.images
    save_output = args.save_output

    out_path = os.path.join(dir_path, "out")
    if not os.path.exists(out_path):
        os.makedirs(out_path)

    images = load_images_from_folder(dir_path)
    n = len(images)

    fig, axs = plt.subplots(n, 3)

    for i, (filename, image) in enumerate(images):
        t0 = time.time()
        print(f"Converting \033[1m{filename}\033[0m to greyscale...")

        img_rgb = image.copy()
        axs[i, 0].imshow(img_rgb)

        img_size = (img_rgb.shape[1], img_rgb.shape[0])
        img_rgb = cv2.resize(img_rgb, size)

        # Convert image to greyscale
        img_grey = AdaptiveGrayscaleImage(img_rgb).integrate()
        img_grey = cv2.resize(img_grey, img_size)

        # Convert image to CIE-Y for reference
        cie_y = CIEY.convert(image)

        axs[i, 1].imshow(img_grey, cmap='gray', vmin=0, vmax=255)
        axs[i, 2].imshow(cie_y, cmap='gray')
        if save_output:
            cv2.imwrite(os.path.join(out_path, f"ags_{filename}"), img_grey.astype(np.uint8))
            cv2.imwrite(os.path.join(out_path, f"ciey_{filename}"), cie_y.astype(np.uint8))

        tf = time.time() - t0
        print(f"Conversion done in \033[92m{format(tf, '.2f')}s!\033[0m\n")

    for ax in axs.flat:
        ax.axis('off')

    axs[0, 0].set_title('Color Image')
    axs[0, 1].set_title('Adaptive Greyscale')
    axs[0, 2].set_title('CIE-Y Greyscale')

    if save_output:
        plt.savefig(os.path.join(out_path, 'result.png'),
                    bbox_inches="tight",
                    pad_inches=1,
                    transparent=False,
                    orientation='landscape')
    plt.show()
