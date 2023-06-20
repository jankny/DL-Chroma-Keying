import cv2
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
import tensorflow as tf
import config
import random
# random.seed(config.RANDOM_SEED)

def gen_trimap_with_dilate(alpha, kernel_size):
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (kernel_size, kernel_size))
    fg_and_unknown = np.array(np.not_equal(alpha, 0).astype(np.float32))
    fg = np.array(np.equal(alpha, 255).astype(np.float32))
    dilate = cv2.dilate(fg_and_unknown, kernel, iterations=1)
    erode = cv2.erode(fg, kernel, iterations=1)
    trimap = erode * 255 + (dilate - erode) * 128
    return trimap.astype(np.uint8)

def gen_trimap_with_dilate_tf(alpha, kernel_size):
    alpha_map = np.rint(alpha.numpy() * 255)

    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (int(kernel_size), int(kernel_size)))
    fg_and_unknown = np.array(np.not_equal(alpha_map, 0).astype(np.float32))
    fg = np.array(np.equal(alpha_map, 255).astype(np.float32))
    dilate = cv2.dilate(fg_and_unknown, kernel, iterations=2)
    erode = cv2.erode(fg, kernel, iterations=2)
    trimap = erode * 255 + (dilate - erode) * 127.5

    return trimap.astype(np.float32)

def get_trimap_tf(alpha, kernel_size):
    alpha_map = np.squeeze(np.rint(alpha.numpy() * 255))

    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (int(kernel_size), int(kernel_size)))
    bg = np.array(np.equal(alpha_map, 0).astype(np.float32))
    fg = np.array(np.equal(alpha_map, 255).astype(np.float32))
    bg_erode = cv2.erode(bg, kernel, iterations=1)
    fg_erode = cv2.erode(fg, kernel, iterations=1)

    trimap = np.empty(alpha_map.shape)
    trimap.fill(127.5)
    trimap = trimap - (bg_erode * 127.5) + (fg_erode * 127.5)
    return trimap.astype(np.float32)

def get_trimap_tf_rand_dims(alpha, kernel_size, iterations=3):
    alpha_map = np.squeeze(np.rint(alpha.numpy() * 255))

    bg = np.array(np.equal(alpha_map, 0).astype(np.float32))
    fg = np.array(np.equal(alpha_map, 255).astype(np.float32))
    bg_erode = bg
    fg_erode = fg

    kernel_size_range = (8, 16)
    kernel_x = random.randint(kernel_size_range[0], kernel_size_range[1])
    kernel_y = int(random.uniform(0.1666, 0.3333) * kernel_x) if kernel_x > 12 else int(random.randint(3, 6) * kernel_x)

    for i in range(iterations):
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (kernel_x, kernel_y))
        if i == 0:
            kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (20,20))
        bg_erode = cv2.erode(bg_erode, kernel, iterations=1)
        fg_erode = cv2.erode(fg_erode, kernel, iterations=1)

    trimap = np.empty(alpha_map.shape)
    trimap.fill(127.5)
    trimap = trimap - (bg_erode * 127.5) + (fg_erode * 127.5)
    return trimap.astype(np.float32)


def tf_get_trimap(alpha, kernel_size):
    kernel = tf.convert_to_tensor(cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (kernel_size, kernel_size)))
    kernel = tf.expand_dims(kernel, -1)
    fg_and_unknown = tf.cast(tf.not_equal(alpha, 0), np.float32)
    fg = tf.cast(tf.equal(alpha, 255), np.float32)
    dilate = tf.nn.dilation2d(fg_and_unknown, kernel, iterations=1)
    dilate = cv2.dilate(fg_and_unknown, kernel, iterations=1)
    erode = cv2.erode(fg, kernel, iterations=1)
    trimap = erode * 255 + (dilate - erode) * 128
    return trimap.astype(np.uint8)


def plot_trimap(tri):
    plt.imshow(tri, cmap="gray")
    plt.show()


if __name__ == "__main__":
    path = "../pascales-thesis/stick/DataSet/TrainingsSet/ImagesScenes/scene0_sintel_vincent_81_GT/0033.png"

    image = Image.open(path)
    image = np.array(image)
    print(image.shape)


    # Tensorflow Test
    alpha_tensor = tf.convert_to_tensor(image[:, :, 3:4]) / 255
    # trimap_tf = gen_trimap_with_dilate_tf(alpha_tensor, kernel_size=config.TRIMAP_KERNEL_SIZE)
    trimap_tf = get_trimap_tf(alpha_tensor, kernel_size=config.TRIMAP_KERNEL_SIZE)
    # trimap_tf = get_trimap_tf_rand_dims(alpha_tensor, kernel_size=config.TRIMAP_KERNEL_SIZE)
    plt.imshow(trimap_tf, cmap="gray")

    # trimap = get_trimap(image[:, :, 3:4], kernel_size=40)
    # trimap = gen_trimap_with_dilate(image[:, :, 3:4], kernel_size=40)
    # plt.imshow(trimap, cmap="gray")
    plt.axis("off")
    plt.tight_layout(pad=0.0)
    plt.show()
