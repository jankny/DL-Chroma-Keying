import re
import os
import utils
import config
import numpy as np
import colorsys
from PIL import Image
import matplotlib.pyplot as plt
from multiprocessing import Pool
from collections import Counter
from functools import partial
from matplotlib.colors import hsv_to_rgb
from scipy import spatial
from imutils import paths as im_paths


def reduce_color_res(color, by: int):
    return round(color[0] / by), round(color[1] / by), round(color[2] / by)


def load_and_count(path, reduce_by=None):
    color_count = {}
    alpha_path = None
    if "Adobe" in path:
        alpha_path = str.replace(path, "fg", "alpha")

    img = utils.load_image(path, config.IMAGE_SIZE, mode="RGBA", alpha_path=alpha_path)
    pixels = np.reshape(img, (-1, 4))

    for pix in pixels:
        if (pix[3] < 0.02):
            continue

        color = (pix[0], pix[1], pix[2])
        if reduce_by:
            color = reduce_color_res(color, reduce_by)

        if color in color_count:
            color_count[color] = color_count[color] + 1
        else:
            color_count[color] = 1

    return color_count


def count_colors(paths, reduce_by=None):
    color_count = {}

    workers = os.cpu_count()
    print(f"using {workers} worker")

    with Pool(workers) as p:
        counts = p.map(partial(load_and_count, reduce_by=reduce_by), paths)

    for count in counts:
        color_count = {key: count.get(key, 0) + color_count.get(key, 0)
                  for key in set(count) | set(color_count)}

    return color_count


def plot_hsv_space(color_count, reduced_by, fill_value=0.20, s_res=5, title=None, show=True):
    V, H = np.mgrid[0:1:50j, 0:1:150j]
    S_RGB = np.zeros((s_res, V.shape[0], V.shape[1], 3))

    s_val = np.linspace(0, 1, num=s_res)
    for i, s in enumerate(s_val):
        S = np.full(V.shape, s)
        HSV = np.dstack((H, S, V))
        RGB = hsv_to_rgb(HSV)
        S_RGB[i] = RGB

    S_RGB_list = np.reshape(S_RGB, (-1, 3))
    alpha = np.full((S_RGB_list.shape[0], 1), 0)

    tree = spatial.KDTree(S_RGB_list)

    for (x, y, z), count in color_count.items():
        color = np.array([x * reduced_by / 255, y * reduced_by / 255, z * reduced_by / 255])
        _, i = tree.query(color)

        alpha[i] = alpha[i] + 1

    alpha_shape = (S_RGB.shape[0], S_RGB.shape[1], S_RGB.shape[2], 1)
    S_RGB_alpha = np.reshape(alpha, alpha_shape)

    max_alpha = np.max(alpha)
    background = max(int(max_alpha * fill_value), 1)

    fig, ax = plt.subplots(nrows=s_res, ncols=1, figsize=(8, 4*s_res))

    for i, (RGB_alpha, row) in enumerate(zip(S_RGB_alpha, ax)):
        black_alpha = np.sum(RGB_alpha[0])
        RGB_alpha[0] = black_alpha

        RGB_alpha = RGB_alpha + background
        RGB_alpha = np.divide(RGB_alpha, max_alpha + background)

        RGBA = np.dstack((S_RGB[i], RGB_alpha))

        row.imshow(RGBA, origin="lower", extent=[0, 360, 0, 1], aspect=150)
        row.set_xlabel("H", fontsize=20)
        row.set_ylabel("V", fontsize=20)
        row.set_title("$S_{HSV}=" + f"{s_val[i]}" + "$", fontsize=22)

    if title is not None:
        fig.suptitle(title)
    plt.tight_layout()
    if show:
        plt.show()
    else:
        return fig


if __name__ == "__main__":
    p_val_paths = [] \
                    + utils.load_json(config.P_VAL_PATHS_GREEN) \
                    + utils.load_json(config.P_PRETRAIN_VAL_PATHS_GREEN) \

    p_test_paths = utils.load_json(config.P_TEST_PATHS_GREEN) \

    paths = [utils.get_gt(p) for p in p_val_paths]
    # paths = [utils.get_gt(p) for p in p_test_paths]

    reduce_by = 1
    color_count = count_colors(paths, reduce_by)
    print("colors counted")

    plot_hsv_space(color_count, reduced_by=reduce_by, s_res=5)

    print("finished")
