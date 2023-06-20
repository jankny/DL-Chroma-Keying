import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
from PIL import Image
from PIL import ImageColor
from randomcolor import RandomColor # https://github.com/davidmerfield/randomColor

from utils import merge_fg_bg
from config import RANDOM_SEED
from perlin2d import generate_perlin_noise_2d, generate_fractal_noise_2d

np.random.seed(RANDOM_SEED)


def generate_perlin_numpy(xpix, ypix, seed=None):
    shape = (xpix, ypix)
    res = (2, 2)
    noise = generate_fractal_noise_2d(shape, res, octaves=4, persistence=0.5, lacunarity=2, seed=seed)
    return noise


def generate_perlin_background(xpix, ypix, seed=None, hue="green"):
    color_seed = np.random.randint(0, np.iinfo(np.int32).max)
    randColor = RandomColor(seed=color_seed)
    color = randColor.generate(hue=hue)[0]
    rgb = ImageColor.getcolor(color, "RGB")

    shape = [xpix.numpy(), ypix.numpy(), 3]

    img = np.full(shape, rgb)

    noise = generate_perlin_numpy(shape[0], shape[1])

    img_noise = (img / 255 + 0.2 * np.expand_dims(noise, -1))
    img_noise = np.clip(img_noise, 0, 1).astype(np.float32)

    # plt.imshow(noise, cmap='gray')
    # plt.show()
    # plt.imshow(img)
    # plt.show()
    # plt.imshow(img_noise)
    # plt.show()
    return img_noise



if __name__ == "__main__":
    fg = Image.open("/home/jannes/uni/jk-masterarbeit/data/Adobe_Deep_Matting_Dataset/Combined_Dataset/Training_set/gt/5149410930_3a943dc43f_b.png")
    # fg = fg.resize((256, 256))
    fg = np.array(fg) / 255
    shape = tf.convert_to_tensor(fg.shape)

    pic = generate_perlin_background(shape[0], shape[1])
    im = merge_fg_bg(fg, pic)
    plt.imshow(im)
    plt.axis("off")
    plt.tight_layout(pad=0.0)
    plt.show()

    # noise = generate_perlin_numpy(256, 256)
    # noise = generate_perlin_numpy(512, 512)
    # plt.imshow(noise, cmap="gray")
    # plt.show()