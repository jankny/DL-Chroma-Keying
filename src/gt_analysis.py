import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
from matplotlib import colors


def brightness_analysis(image, image_gt):
    # rgb_to_hsv = lambda x: colors.rgb_to_hsv(np.uint8(x * 255)) / 255
    rgb_to_hsv = lambda x: colors.rgb_to_hsv(x)
    image_hsv = rgb_to_hsv(image[...,:3])
    image_gt_hsv = np.concatenate([rgb_to_hsv(image_gt[:,:,:3]), image_gt[:,:,3:4]], -1)

    hsv_diff = image_hsv - image_gt_hsv[...,:3]
    # plt.imshow(np.full_like(hsv_diff, 0.5)+hsv_diff[...,2:3], cmap="gray")
    plt.imshow(np.full_like(hsv_diff[...,:1], 0.5)+hsv_diff[...,0:1], cmap="gray")
    plt.show()


def rgb2gray(rgb):
    return np.dot(rgb[...,:3], [0.2989, 0.5870, 0.1140])


def color_diff(image, image_gt, image2):
    image = np.array(image)
    image_gt = np.array(image_gt)
    alpha_neq_zero = np.not_equal(image_gt[:,:,3:4], 0)

    rgb_diff_gt = (image[:,:,:3] - image_gt[:,:,:3])
    rgb_diff_gt = np.where(alpha_neq_zero, rgb_diff_gt, np.zeros_like(rgb_diff_gt))
    # rgb_diff_gt = np.max(np.abs(rgb_diff_gt), axis=-1, keepdims=True)

    rgb_diff = (image[:,:,:3] - image2[:,:,:3])
    rgb_diff = np.where(alpha_neq_zero, rgb_diff, np.zeros_like(rgb_diff))
    # rgb_diff = np.mean(np.abs(rgb_diff), axis=-1, keepdims=True)
    # rgb_diff = np.sum(np.abs(rgb_diff), axis=-1, keepdims=True)
    # rgb_diff = np.max(np.abs(rgb_diff), axis=-1, keepdims=True)
    # rgb_diff_gray = np.expand_dims(rgb2gray(np.abs(rgb_diff)), axis=-1)


    plt.imshow(np.full_like(rgb_diff, 0.5) + rgb_diff, cmap="gray", vmin=0, vmax=1)
    # plt.imshow(np.full_like(rgb_diff_gt, 0.5) + rgb_diff, cmap="gray", vmin=0, vmax=1)

    plt.axis("off")
    plt.tight_layout(pad=0.0)
    plt.show()

if __name__ == "__main__":
    # image = np.array(Image.open("/home/jannes/uni/jk-masterarbeit/pascales-thesis/stick/DataSet/TrainingsSet/ImagesScenes/scene1_sintel_20_SPILL_G/0004.png")) / 255
    # image_blue = np.array(Image.open("/home/jannes/uni/jk-masterarbeit/pascales-thesis/stick/DataSet/TrainingsSet/ImagesScenes/scene1_sintel_20_SPILL_B/0004.png")) / 255
    # image_gt = np.array(Image.open("/home/jannes/uni/jk-masterarbeit/pascales-thesis/stick/DataSet/TrainingsSet/ImagesScenes/scene1_sintel_20_GT/0004.png")) / 255
    image = np.array(Image.open("/home/jannes/uni/jk-masterarbeit/pascales-thesis/stick/DataSet/TrainingsSet/ImagesScenes/scene1_vincent_10_SPILL_G/0004.png")) / 255
    image_blue = np.array(Image.open("/home/jannes/uni/jk-masterarbeit/pascales-thesis/stick/DataSet/TrainingsSet/ImagesScenes/scene1_vincent_10_SPILL_B/0004.png")) / 255
    image_gt = np.array(Image.open("/home/jannes/uni/jk-masterarbeit/pascales-thesis/stick/DataSet/TrainingsSet/ImagesScenes/scene1_vincent_10_GT/0004.png")) / 255
    # image = np.array(Image.open("/home/jannes/uni/jk-masterarbeit/green_benchmark/benchmark/green/25_colors3.png")) / 255
    # image_gt = np.array(Image.open("/home/jannes/uni/jk-masterarbeit/green_benchmark/benchmark/ground_truth/25_colors3.png")) / 255


    # brightness_analysis(image, image_gt)

    color_diff(image, image_gt, image_blue)
