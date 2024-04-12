import os
import numpy as np
import cv2 as cv
import matplotlib.pyplot as plt

from PIL import Image


def read_grayscale_image(img_path, resize=None):
    im = cv.imread(img_path, cv.IMREAD_GRAYSCALE)
    if resize:
        im = cv.resize(im, (0, 0), fx = resize, fy = resize)
    return im


def resolution_distributions(path):
    files = os.listdir(path)
    resolutions = dict()

    for file in files:
        file_path = os.path.join(path, file)
        with Image.open(file_path) as img:
            width, height = img.size
        str_res = f"{width}x{height}"
        cnt = resolutions.get(str_res, 0)
        resolutions[str_res] = cnt + 1

    return resolutions


def display_images(sat_image, uav_image, title=""):
    image1 = Image.open(sat_image)
    image2 = Image.open(uav_image)

    fig = plt.figure(figsize=(10, 5))

    ax1 = fig.add_subplot(1, 2, 1)
    ax1.set_axis_off()
    ax1.imshow(image1)
    ax1.set_title("Satellite Image")

    ax2 = fig.add_subplot(1, 2, 2)
    ax2.set_axis_off()
    ax2.imshow(image2)
    ax2.set_title("UAV Image")

    fig.suptitle(title, fontsize=12, y=0.95)
    fig.tight_layout(pad=1.5)

    plt.show()
