import os
import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
from os.path import join
from imutils import paths

import config
import plots
from Setup import Setup
# from PascaleModel import PascaleModel
from UNetAdvanced import UNetAdvanced

SAVE_IMAGES = True
SAVE_PATH = "/home/jannes/uni/jk-masterarbeit/arbeit/Evaluation"

BENCHMARK_PATH = config.BENCHMARK_PATH

BLUE_PATHS = list(sorted(paths.list_images(join(BENCHMARK_PATH, "blue"))))
GREEN_PATHS = list(sorted(paths.list_images(join(BENCHMARK_PATH, "green"))))
regex_replace = config.BENCHMARK_REGEX_REPLACE

models = [
        ([*GREEN_PATHS], UNetAdvanced(path="/home/jannes/uni/jk-masterarbeit/saves_logs/0608-1536-UAdv_WA").load_model().set_name("plain")),
        # ([*GREEN_PATHS], UNetAdvanced(path="/home/jannes/uni/jk-masterarbeit/saves_logs/0608-1728-UAdv_SL_WA").load_model().set_name("SL")),
        # ([*GREEN_PATHS], UNetAdvanced(path="/home/jannes/uni/jk-masterarbeit/saves_logs/0608-1932-UAdv_TC_WA").load_model().set_name("TC")),
        # ([*GREEN_PATHS], UNetAdvanced(path="/home/jannes/uni/jk-masterarbeit/saves_logs/0608-2138-UAdv_WA_4").load_model().set_name("MP4")),
        # ([*GREEN_PATHS], UNetAdvanced(path="/home/jannes/uni/jk-masterarbeit/saves_logs/0609-0001-UAdv5_WA").load_model().set_name("MP5")),
        # ([*GREEN_PATHS], UNetAdvanced(path="/home/jannes/uni/jk-masterarbeit/saves_logs/0609-0139-UAdv4_TC_WA").load_model().set_name("MP4+TC")),
        # ([*GREEN_PATHS], UNetAdvanced(path="/home/jannes/uni/jk-masterarbeit/saves_logs/0609-0337-UAdv4_L_WA").load_model().set_name("MP4+LPRED")),

        ([*GREEN_PATHS], UNetAdvanced(path="/home/jannes/uni/jk-masterarbeit/saves_logs/0609-1610-UAdv4_MIX_DBG_WA").load_model().set_name("ND")),
        ([*GREEN_PATHS], UNetAdvanced(path="/home/jannes/uni/jk-masterarbeit/saves_logs/0510-0105-UAdv_SL_DIM_TC_L_DBG_W_AIM").load_model().set_name("SB")),
        ([*GREEN_PATHS], UNetAdvanced(path="/home/jannes/uni/jk-masterarbeit/saves_logs/0514-2147-UAdv_SL_MIX_TC_L_DBG_W_AIMA").load_model().set_name("DIFF")),
        ([*GREEN_PATHS], UNetAdvanced(path="/home/jannes/uni/jk-masterarbeit/saves_logs/0607-1124-UAdv_SL_MIX_L_DBG_WA_AIMAG").load_model().set_name("T")),
]

# img_name = "0_cactus.png"
# img_name = "2_teddy.png"
# img_name = "25_colors3.png"
# img_name = "26_car.png"
# img_name = "28_leaves2.png"
# img_name = "29_fox_spill.png"
img_name = "30_fox.png"
eval_paths = [join(BENCHMARK_PATH, "green", img_name)]

rgba = None
save_name = ""
model_names = []

model: Setup
for _, model in models:
    name = model.name
    save_name = save_name + f"__{name}"
    model_names.append(name)
    print(f"Evaluating {name}")

    eval_ds = model.get_test_dataset(eval_paths, regex_replace)
    eval_ds = eval_ds.unbatch().batch(len(eval_paths))

    [(eval_x, eval_y)] = list(eval_ds.take(len(eval_paths)).as_numpy_iterator())

    eval_pred = model.predict(eval_x)
    outputs = UNetAdvanced.outputs_to_img(eval_x, eval_pred, model.nn_config)

    rgba = outputs[0] if rgba is None else tf.concat([rgba, outputs[0]], 0)

fig = plots.model_comparison(eval_x, eval_y["rgba"], rgba, model_names=model_names, show=True)
fig.savefig(join(SAVE_PATH, f"{img_name[:-4]}{save_name}.png"))
