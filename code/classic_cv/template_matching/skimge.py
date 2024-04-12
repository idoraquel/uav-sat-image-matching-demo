import numpy as np
import matplotlib.pyplot as plt
import cv2 as cv

from skimage.util import crop
from skimage.feature import match_template
from skimage.transform import rotate
from sklearn.metrics.pairwise import manhattan_distances, euclidean_distances
from utils import read_grayscale_image


def center_crop(image, crop_size):
    """Center crops an image to the specified size.

    Args:
        image (ndarray): The input image.
        crop_size (tuple): The desired crop size in (height, width) format.

    Returns:
        ndarray: The center cropped image.
    """
    height, width = image.shape[:2]
    crop_height, crop_width = crop_size

    start_y = height // 2 - crop_height // 2
    start_x = width // 2 - crop_width // 2
    end_y = start_y + crop_height
    end_x = start_x + crop_width

    cropped_image = image[start_y:end_y, start_x:end_x]
    return cropped_image


def get_center_point(image):
    height, width = image.shape[:2]
    x_c = width // 2
    y_c = height // 2
    return x_c, y_c


def read_grayscale_pair(basename, resize=None):

    """Reads grayscale pair of corresponding images

    Yields:
        _type_: _description_
    """
    for src in [f"data/test/{pref}_images/{basename}" for pref in ["uav", "sat"]]:
        yield read_grayscale_image(src, resize=resize)



def retrieve_image_position(uav_image, sat_image, crop_size=128):

    patch = center_crop(uav_image, (crop_size, crop_size))
    xc, yc = get_center_point(sat_image)

    result = match_template(sat_image, patch)
    ij = np.unravel_index(np.argmax(result), result.shape)
    x, y = ij[::-1]

    manhattan_distance = abs(x - xc) + abs(y - yc)

    return manhattan_distance


def plot_results(uav_image, sat_image, patch, result_map):
    fig = plt.figure(figsize=(8, 8))
    ax1 = plt.subplot(2, 2, 1)
    ax2 = plt.subplot(2, 2, 2)
    ax3 = plt.subplot(2, 2, 3)
    ax4 = plt.subplot(2, 2, 4)

    ax1.imshow(uav_image, cmap=plt.cm.gray)
    ax1.set_axis_off()
    ax1.set_title('UAV image')

    ax2.imshow(patch, cmap=plt.cm.gray)
    ax2.set_axis_off()
    ax2.set_title('Center patch')

    ax3.imshow(sat_image, cmap=plt.cm.gray)
    ax3.set_axis_off()
    ax3.set_title('Satellite image')
    # highlight matched region
    hcoin, wcoin = patch.shape
    rect = plt.Rectangle((x, y), wcoin, hcoin, edgecolor='r', facecolor='none')
    ax3.add_patch(rect)

    ax4.imshow(result_map)
    ax4.set_axis_off()
    ax4.set_title('`match_template`\nresult')
    # highlight matched region
    ax4.autoscale(False)
    ax4.plot(x, y, 'o', markeredgecolor='r', markerfacecolor='none', markersize=10)

    plt.show()



if __name__ == "__main__":
    import pandas as pd

    test_df = pd.read_csv("data/test/pairs.csv")
    
    def calc_distance(row):
        uav_image = read_grayscale_image(row["uav_image"], resize=0.2)
        sat_image = read_grayscale_image(row["sat_image"], resize=0.2)
        return retrieve_image_position(uav_image, sat_image, 128)

    test_df["mh_dst_128_f02"] = test_df.apply(calc_distance, axis=1)

    test_df.to_csv("data/test/pairs_with_results.csv", index=False)
