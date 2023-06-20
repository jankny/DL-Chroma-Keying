import json
import numpy as np
from PIL import Image
# import tensorflow as tf
import re


def load_json(file_path):
    with open(file_path, 'r') as f:
        file = json.load(f)
    return file


def load_image(img_path, img_shape, mode="RGB", alpha_path=None):
    assert (mode=="RGB") or (mode=="RGBA"), "incorrect mode"
    mode_idx =  3 if (mode=="RGB") else 4

    img = Image.open(img_path)
    img.thumbnail(img_shape, Image.ANTIALIAS)
    img = np.array(img, dtype=np.float32)
    if alpha_path:
        alpha = Image.open(alpha_path)
        alpha.thumbnail(img_shape, Image.ANTIALIAS)
        alpha = np.expand_dims(np.array(alpha, dtype=np.float32), -1)
        if img.shape[-1] == 4:
            img[:,:,3] = alpha
        else:
            img = np.dstack((img, alpha))
    return img[:,:,:mode_idx]


def get_gt(path: str):
    return re.sub(r"SPILL_[GB]", "GT", path)


def merge_fg_bg(fg, bg):
    image = fg[:,:,:3] * fg[:,:,3:4] + bg * (1 - fg[:,:,3:4])
    return image


def rgb_to_gray(rgb_tensor):
    return 0.299 * rgb_tensor[..., 0:1] + 0.587 * rgb_tensor[..., 1:2] + 0.114 * rgb_tensor[..., 2:3]
