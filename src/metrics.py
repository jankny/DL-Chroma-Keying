import numpy as np
import tensorflow as tf
import tensorflow_addons as tfa
from gradient_loss import gauss_gradient


def alpha_loss4D(y_true, y_pred): # for rgba images
    return tf.square(y_true[..., 3] - y_pred[..., 3])

def alpha_loss_root(y_true, y_pred):
    return tf.sqrt(alpha_loss4D(y_true, y_pred) + 1e-12)

def weighted_rgb_loss(y_true, y_pred):
    # return tf.reduce_mean(y_true[:,:,:,3:4] * (tf.square(y_true[:,:,:,:3]-y_pred[:,:,:,:3])), axis=-1)
    loss = weighted_mse(y_true[:,:,:,:3], y_pred[:,:,:,:3], y_true[:,:,:,3:4])
    return loss

def weighted_rgb_loss_root(y_true, y_pred):
    # return tf.reduce_mean(y_true[:,:,:,3:4] * tf.sqrt(tf.square(y_true[:,:,:,:3]-y_pred[:,:,:,:3]) + 1e-12), axis=-1)
    loss = weighted_mse(y_true[:,:,:,:3], y_pred[:,:,:,:3], y_true[:,:,:,3:4], root=True)
    return loss

def wmse(y_true, y_pred, root=False):
    if root:
        return alpha_loss_root(y_true, y_pred) + weighted_rgb_loss_root(y_true, y_pred)
    else:
        return alpha_loss4D(y_true, y_pred) + weighted_rgb_loss(y_true, y_pred)

def wmae(y_true, y_pred):
    return wmse(y_true, y_pred, root=True)

def wmse_cs_mp(y_true, y_pred, root=False): # add more weight to color spill pixels and mixed pixels
    x = y_true[:,:,:,4:7]
    y_true_rgba = y_true[:,:,:,:4]
    y_true_alpha = y_true[:,:,:,3:4]

    wmse_loss = tf.expand_dims(wmse(y_true_rgba, y_pred, root=root), -1)

    alpha_eq_one = tf.equal(y_true_alpha, 1)
    color_spill_weight = tf.abs(y_true[:,:,:,:3] - x) * tf.where(alpha_eq_one, y_true_alpha, tf.zeros_like(y_true_alpha))
    color_spill_loss = wmse_loss * color_spill_weight

    mixed_pixels_weight = tf.where(alpha_eq_one, tf.zeros_like(y_true_alpha), y_true_alpha)
    mixed_pixels_loss = wmse_loss * mixed_pixels_weight

    return color_spill_loss + mixed_pixels_loss + wmse_loss

def weighted_mse(y_true, y_pred, weights, root=False):
    diff = y_pred - y_true
    diff = diff * weights
    loss = tf.math.square(diff)
    if root: loss = tf.math.sqrt(loss + 1e-12)
    loss = tf.math.reduce_sum(loss) / (tf.math.reduce_sum(weights) + 1)
    return loss

def semantic_branch_loss(y_true, y_pred):
    trimap_true = y_true[..., :1]
    alpha_true = y_true[..., 1:]

    # guided alpha loss
    weights = tf.zeros_like(alpha_true)
    weights = tf.where(tf.equal(trimap_true, 1), tf.ones_like(weights), weights)
    alpha_loss = weighted_mse(alpha_true, y_pred[..., 3:], weights, root=True)

    # semantic loss
    semantic_loss = tf.keras.losses.SparseCategoricalCrossentropy(from_logits=False)(trimap_true, y_pred[..., :3] + 1e-12)

    return alpha_loss + semantic_loss

def rgb_max_diff_loss(y_true, y_pred): # rgb_max_diff and alpha_true needed
    # whole map
    loss = tf.square(y_true[..., :1] - y_pred)
    return loss

def rgb_correction_loss(y_true, y_pred): # rgb_max_diff_true, rgba_true
    alpha = y_true[...,4:]
    alpha_not_zero = tf.cast(alpha > 0, tf.float32)
    weights = y_true[..., :1] * alpha_not_zero
    loss = weighted_mse(y_true[..., 1:4], y_pred[..., :3], weights, root=True)
    return loss

def rgb_constant_loss(y_true, y_pred): # rgb_max_diff_true, rgba_true
    alpha = y_true[...,4:]
    alpha_not_zero = tf.cast(alpha > 0, tf.float32)
    weights = (1 - y_true[..., :1]) * alpha_not_zero
    loss = weighted_mse(y_true[..., 1:4], y_pred[..., :3], weights, root=True)
    return loss


def rgb_addon_loss(y_true, y_pred): # rgb_max_diff, rgba_true; guided_rgb, rgb_max_diff
    rgb_max_diff_true, rgba_true = y_true[..., :1], y_true[..., 1:]
    rgb_true, alpha_true = rgba_true[..., :3], rgba_true[..., 3:4]

    guided_rgb, rgb_max_diff_pred, rgb_diff, inputs = y_pred[..., :3], y_pred[..., 3:4], y_pred[..., 4:7], y_pred[..., 7:]

    max_diff_loss = rgb_max_diff_loss(tf.concat([rgb_max_diff_true, alpha_true], axis=-1), rgb_max_diff_pred)
    rgb_loss = rgb_correction_loss(tf.concat([rgb_max_diff_true, rgba_true], axis=-1), guided_rgb)

    # alpha_not_zero = tf.cast(alpha_true > 0, tf.float32)
    # diff_true = rgb_true - inputs
    # rgb_diff_loss = weighted_mse(rgb_diff, diff_true, alpha_not_zero, root=True)
    # rgb_diff_loss = weighted_mse(rgb_diff, diff_true, rgb_max_diff_true, root=True)

    return max_diff_loss + rgb_loss

def rgb_addon_rgb_only(y_true, y_pred): # rgb_max_diff, rgba_true; guided_rgb, rgb_max_diff
    rgb_max_diff_true, rgba_true = y_true[..., :1], y_true[..., 1:]

    rgb_pred = y_pred[..., :3]
    loss = weighted_rgb_loss_root(rgba_true, rgb_pred)
    return loss

def alpha_gradient_loss(y_true, y_pred, root=False):
    if root:
        alpha_loss = alpha_loss_root(y_true, y_pred)
    else:
        alpha_loss = alpha_loss4D(y_true, y_pred)

    grad_loss = gradient_loss(y_true, y_pred)
    return tf.expand_dims(alpha_loss, axis=-1) + grad_loss

def gradient_loss(y_true, y_pred):
    alpha_true = y_true[:,:,:,3:4]
    alpha_pred = y_pred[:,:,:,3:4]
    sigma = tf.constant(1.4, tf.float32)

    grad_true = gauss_gradient(alpha_true, sigma)
    grad_pred = gauss_gradient(alpha_pred, sigma)
    loss = tf.square(grad_pred - grad_true)
    loss = tf.sqrt(loss + 1e-12)

    return loss

def alpha_loss_bg(y_true, y_pred): #trimap_true, alpha_true; trimap_pred, alpha_pred
    trimap_true = y_true[..., :1]
    alpha_true = y_true[..., 1:]

    weights = tf.cast(tf.equal(trimap_true, 0), tf.float32)
    loss = weighted_mse(alpha_true, y_pred[..., 3:], weights, root=True)
    return loss

def alpha_loss_trans(y_true, y_pred): #trimap_true, alpha_true; trimap_pred, alpha_pred
    trimap_true = y_true[..., :1]
    alpha_true = y_true[..., 1:]

    weights = tf.cast(tf.equal(trimap_true, 1), tf.float32)
    loss = weighted_mse(alpha_true, y_pred[..., 3:], weights, root=True)
    return loss

def alpha_loss_fg(y_true, y_pred): #trimap_true, alpha_true; trimap_pred, alpha_pred
    trimap_true = y_true[..., :1]
    alpha_true = y_true[..., 1:]

    weights = tf.cast(tf.equal(trimap_true, 2), tf.float32)
    loss = weighted_mse(alpha_true, y_pred[..., 3:], weights, root=True)
    return loss

def ssim(y_true, y_pred):
    ssim = tf.image.ssim(y_true[..., :3]*y_true[..., 3:4], y_pred[..., :3]*y_true[..., 3:4], max_val=1.0, filter_size=11,
                          filter_sigma=1.5, k1=0.01, k2=0.03)
    return ssim

