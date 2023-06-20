# adapted from https://github.com/PaddlePaddle/PaddleSeg/blob/972e05cdcf29a835a5cdb4773b17c5c4e66c1c90/Matting/ppmatting/metrics/metric.py#L132
import math
import tensorflow as tf
import numpy as np
from numpy import testing
from cv2 import cv2

def gaussian(x, sigma):
    return tf.exp(-1 * tf.square(x) / (2 * tf.square(sigma))) / (sigma * tf.sqrt(2 * math.pi))

def dgaussian(x, sigma):
    return -x * gaussian(x, sigma) / tf.square(sigma)

def gauss_filter(sigma, epsilon=1e-2):
    half_size = tf.math.ceil(
        sigma * tf.sqrt(-2 * tf.math.log(tf.sqrt(2 * math.pi) * sigma * epsilon)))
    half_size = tf.cast(half_size, tf.float32)
    size = int(2 * half_size + 1)

    # create filter in x axis
    filter_shape = (size, size)
    index = tf.cast(tf.range(size), tf.float32)
    d_gauss = tf.expand_dims(dgaussian(index - half_size, sigma), axis=0)
    gauss = tf.expand_dims(gaussian(index - half_size, sigma), axis=1)
    filter_x = tf.broadcast_to(gauss, filter_shape) * tf.broadcast_to(d_gauss, filter_shape)

    # normalize filter
    norm = tf.sqrt(tf.reduce_sum(tf.square(filter_x)))
    filter_x = filter_x / norm
    filter_y = tf.transpose(filter_x)
    filter_x = tf.reshape(filter_x, (*filter_shape, 1, 1))
    filter_y = tf.reshape(filter_y, (*filter_shape, 1, 1))

    return filter_x, filter_y


def gauss_filter_np(sigma, epsilon=1e-2):
    half_size = np.ceil(
        sigma * np.sqrt(-2 * np.log(np.sqrt(2 * np.pi) * sigma * epsilon))).astype(np.float32)
    size = int(2 * half_size + 1)

    # create filter in x axis
    filter_x = np.zeros((size, size)).astype(np.float32)
    for i in range(size):
        for j in range(size):
            filter_x[i, j] = gaussian(
                (i - half_size).astype(np.float32), sigma) * dgaussian((j - half_size).astype(np.float32), sigma)

    # normalize filter
    norm = np.sqrt((filter_x**2).sum())
    filter_x = filter_x / norm
    filter_y = np.transpose(filter_x)

    return filter_x, filter_y


def gauss_gradient(img, sigma):
    filter_x, filter_y = gauss_filter(sigma)

    _, h, w, _ = img.shape
    fh, fw, _, _ = filter_x.shape

    if h % 1 == 0:
        pad_along_height = tf.maximum(fh - 1, 0)
    else:
        pad_along_height = tf.maximum(fh - (h % 1), 0)
    if w % 1 == 0:
        pad_along_width = tf.maximum(fw - 1, 0)
    else:
        pad_along_width = tf.maximum(fw - (w % 1), 0)
    pad_top = pad_along_height // 2
    pad_bottom = pad_along_height - pad_top
    pad_left = pad_along_width // 2
    pad_right = pad_along_width - pad_left

    pad = tf.pad(img, [[0, 0], [pad_top, pad_bottom], [pad_left, pad_right], [0, 0]], mode="SYMMETRIC")
    img_filtered_x = tf.nn.conv2d(pad, filter_x, [1, 1], "VALID")
    img_filtered_y = tf.nn.conv2d(pad, filter_y, [1, 1], "VALID")

    mag = tf.square(img_filtered_x) + tf.square(img_filtered_y)
    mag = tf.sqrt(mag + 1e-12)
    return mag


def gauss_gradient_np(img, sigma):
    filter_x, filter_y = gauss_filter_np(sigma)
    img_filtered_x = cv2.filter2D(
        img, -1, filter_x, borderType=cv2.BORDER_REFLECT)
    img_filtered_y = cv2.filter2D(
        img, -1, filter_y, borderType=cv2.BORDER_REFLECT)
    return np.sqrt(img_filtered_x**2 + img_filtered_y**2)


if __name__ == "__main__":
    sigma = tf.cast(1.4, tf.float32)
    img = tf.cast(tf.reshape(tf.range(100), (1, 10, 10, 1)), tf.float32)

    filter_x, filter_y = gauss_filter(sigma)
    # print(filter_y)
    gauss_grad_tf = gauss_gradient(img, sigma)
    print(gauss_grad_tf)

    x, y = gauss_filter_np(sigma)
    # print(y)
    gauss_grad_np = gauss_gradient_np(img[0,:,:,0].numpy(), sigma)
    print(gauss_grad_np)

    testing.assert_array_almost_equal(gauss_grad_np, gauss_grad_tf[0,:,:,0], 4)