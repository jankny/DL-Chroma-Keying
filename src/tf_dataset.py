import cv2
import numpy as np
import tensorflow as tf
import tensorflow_addons as tfa
from tensorflow import keras

import trimap
from config import NNConfig
import config
from PerlinBackground import generate_perlin_background, merge_fg_bg
# import ColorAugmentation

tf.random.set_seed(config.RANDOM_SEED)


def load_image(image_path, channels, image_size, seed=None):
    image = tf.io.read_file(image_path)
    image = tf.image.decode_png(image, channels=channels)
    image = tf.image.convert_image_dtype(image, dtype=tf.float32)
    if seed is not None: # crop augmentation
        image_shape = tf.cast(tf.shape(image), tf.int32)
        random_dim = tf.random.stateless_uniform([1], seed, minval=tf.cast(image_size[0], tf.int32), maxval=tf.minimum(image_shape[0], image_shape[1]), dtype=tf.int32)
        image = tf.image.resize(image, tf.broadcast_to(random_dim, [2]))
        target_shape = (*image_size, channels)
        image = tf.image.stateless_random_crop(image, target_shape, seed)
    else:
        image = tf.image.resize(image, image_size)
    return image


def load_sample(image_path, nn_config:NNConfig, regex_replace=("SPILL_G", "GT", "SPILL_B"), crop_augmentation=False):
    seed = tf.random.uniform([2], maxval=tf.int32.max, dtype=tf.int32) if crop_augmentation else None

    image = load_image(image_path, channels=3, image_size=nn_config.IMAGE_SIZE, seed=seed)
    image_gt_path = tf.strings.regex_replace(image_path, regex_replace[0], regex_replace[1])
    image_gt = load_image(image_gt_path, channels=4, image_size=nn_config.IMAGE_SIZE, seed=seed)

    if not nn_config.TRIMAP_OUTPUT:
        return image, image_gt

    image_trimap_gt = tf.py_function(trimap.get_trimap_tf, [image_gt[:,:,3:4], nn_config.TRIMAP_KERNEL_SIZE], tf.float32)
    image_trimap_gt = tf.expand_dims(image_trimap_gt, -1)
    image_trimap_gt = tf.reshape(image_trimap_gt, (nn_config.IMAGE_SIZE[0], nn_config.IMAGE_SIZE[1], 1))

    image_gt = {"trimap": image_trimap_gt, "rgba": image_gt}

    if nn_config.AIM_RGB_ADDON:
        second_color_path = tf.strings.regex_replace(image_path, regex_replace[0], regex_replace[2])
        if regex_replace[1] != regex_replace[2]:
            second_color_image = load_image(second_color_path, channels=3, image_size=nn_config.IMAGE_SIZE, seed=seed)
        else:
            ones = tf.ones_like(image)
            second_color_image = merge_fg_bg(image_gt["rgba"], ones)
        alpha_neq_zero = tf.not_equal(image_gt["rgba"][:,:,3:4], 0)
        rgb_diff = image[:,:,:3] - second_color_image[:,:,:3]
        rgb_diff = tf.abs(rgb_diff) + 1e-6
        rgb_diff = tf.where(alpha_neq_zero, rgb_diff, tf.zeros_like(rgb_diff))
        rgb_max_diff = tf.reduce_max(rgb_diff, axis=-1, keepdims=True)

        image_gt["rgb_addon"] = rgb_max_diff

    return image, image_gt


def augment_layers(image, seed=None):
    aug_layers = tf.keras.Sequential([
        keras.layers.RandomRotation(0.041, seed=seed),
        keras.layers.RandomTranslation(0.2, 0.2, seed=seed),
        # keras.layers.RandomZoom((0.7, 1), seed=seed),
        keras.layers.RandomFlip("horizontal", seed=seed),
    ])
    return aug_layers(image)


def augment_sample(image, image_gt, nn_config:NNConfig):
    # seed = tf.random.uniform([1], maxval=tf.int32.max, dtype=tf.int32)

    if nn_config.TRIMAP_OUTPUT:
        stack = [image, image_gt["trimap"], image_gt["rgba"]]
        if nn_config.AIM_RGB_ADDON:
            stack.append(image_gt["rgb_addon"])

        images_stacked = tf.concat(stack, -1)
        images_augmented = augment_layers(images_stacked, config.RANDOM_SEED)
        image_gt_rgba = images_augmented[:,:,4:8]
        image_trimap_gt = images_augmented[:,:,3:4]
        if nn_config.AIM_RGB_ADDON:
            return images_augmented[:,:,0:3], {"trimap": image_trimap_gt, "rgba": image_gt_rgba, "rgb_addon": images_augmented[..., 8:9]}
        return images_augmented[:,:,0:3], {"trimap": image_trimap_gt, "rgba": image_gt_rgba}
    else:
        images_stacked = tf.concat([image, image_gt], -1)
        images_augmented = augment_layers(images_stacked, config.RANDOM_SEED)
        return images_augmented[:,:,0:3], images_augmented[:,:,3:]


def finalize(image, image_gt, nn_config:NNConfig):
    if nn_config.TRIMAP_OUTPUT:
        image_gt["trimap"] = tf.math.round(image_gt["trimap"] / 127.5)

    if getattr(nn_config, "SEMANTIC_BRANCH", False):
        if not nn_config.TRIMAP_OUTPUT:
            raise NotImplementedError("No trimap generated")
        image_gt["semantic_branch"] = tf.concat([image_gt["trimap"], image_gt["rgba"][..., 3:4]], -1)

    if getattr(nn_config, "AIM_RGB_ADDON", False):
        image_gt["rgb_addon"] = tf.py_function(dilate_max_diff, [image_gt["rgb_addon"]], tf.float32)
        image_gt["rgb_addon"] = tfa.image.gaussian_filter2d(image_gt["rgb_addon"], (3, 3), 1.)

        scale_min = tf.reduce_min(image_gt["rgb_addon"])
        scale_max = tf.reduce_max(image_gt["rgb_addon"])
        image_gt["rgb_addon"] = (image_gt["rgb_addon"] - scale_min) / (scale_max - scale_min)

        image_gt["rgb_addon"] = tf.concat([image_gt["rgb_addon"], image_gt["rgba"]], -1)

    if nn_config.AIM_RGB_BRANCH:
        image_gt["rgb_max_diff"] = image_gt["rgb_addon"]

    image_gt["trimap"] = tf.concat([image_gt["trimap"], image_gt["rgba"][..., 3:4]], -1)
    return image, image_gt


def dilate_max_diff(max_diff, kernel_size=5):
    max_diff = max_diff.numpy()
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (kernel_size, kernel_size))
    result = cv2.dilate(max_diff, kernel=kernel)
    return tf.expand_dims(result, -1)


def append_input_to_gt(image, image_gt, nn_config:NNConfig):
    if nn_config.TRIMAP_OUTPUT:
        image_gt["rgba"] = tf.concat([image_gt["rgba"], image], -1)
    else:
        image_gt = tf.concat([image_gt, image], -1)

    return image, image_gt


def add_sample_weight(image, image_gt, sample_weight):
    weights = {k: tf.reshape(sample_weight[k], [1 for _ in v.shape]) for k, v in image_gt.items()}
    return image, image_gt, weights


def augment_background(image, image_gt, nn_config:NNConfig):
    fg = image_gt if not nn_config.TRIMAP_OUTPUT else image_gt["rgba"]
    shape = nn_config.IMAGE_SIZE
    bg = tf.py_function(generate_perlin_background, [*shape, config.RANDOM_SEED], tf.float32)

    image = merge_fg_bg(fg, bg)
    return image, image_gt


def augment_color(image, image_gt, nn_config:NNConfig):
    img_gt = image_gt[:,:,:4] if not nn_config.TRIMAP_OUTPUT else image_gt["rgba"]

    # [img, img_gt] = tf.py_function(ColorAugmentation.augment_color, [image, img_gt], [tf.float32, tf.float32])

    seed = tf.random.uniform([2], maxval=tf.int32.max, dtype=tf.int32)
    img = image
    img_gt_rgb = img_gt[:,:,:3]

    img = tf.image.stateless_random_hue(img, 0.045, seed)
    img_gt_rgb = tf.image.stateless_random_hue(img_gt_rgb, 0.045, seed)

    img = tf.image.adjust_brightness(img, 0.2)
    img_gt_rgb = tf.image.adjust_brightness(img_gt_rgb, 0.2)
    img = tf.image.stateless_random_brightness(img, 0.2, seed)
    img_gt_rgb = tf.image.stateless_random_brightness(img_gt_rgb, 0.2, seed)

    img = tf.clip_by_value(img, 0, 1)
    img_gt_rgb = tf.clip_by_value(img_gt_rgb, 0, 1)
    img_gt = tf.concat([img_gt_rgb, img_gt[..., 3:]], axis=-1)

    if nn_config.TRIMAP_OUTPUT:
        image_gt["rgba"] = img_gt
    else:
        image_gt = img_gt
    return img, image_gt


def diff_as_target(image, image_gt, nn_config:NNConfig):
    img_gt = image_gt[:,:,:4] if not nn_config.TRIMAP_OUTPUT else image_gt["rgba"][:,:,:4]
    diff = image[...,:3] - img_gt[...,:3]
    img_gt = tf.concat([diff, img_gt[...,3:4]], axis=-1)

    if nn_config.TRIMAP_OUTPUT:
        image_gt["rgba"] = img_gt
    else:
        image_gt = img_gt
    return image, image_gt
