import os
import re
import pylab
import numpy as np
import pandas as pd
import tensorflow as tf
import matplotlib.pyplot as plt
from tensorflow import keras
from os.path import join
from imutils import paths

import config
import plots
import utils
import metrics
from Setup import Setup
from PascaleModel import PascaleModel
from UNetAdvanced import UNetAdvanced

SAVE_IMAGES = True
TEST_EVAL = False

RESULT_PATH = "/home/jannes/uni/jk-masterarbeit/evaluation"

BLUE_PATHS = config.BLUE_PATHS
GREEN_PATHS = config.GREEN_PATHS
regex_replace = config.BENCHMARK_REGEX_REPLACE


PASCALE_THESIS = config.P_EVAL_DIR
P_EVALUATION_PATHS = [
    *list(sorted(paths.list_images(join(PASCALE_THESIS, "Prediction Motion Blur Video", "Original")))),
    *list(sorted(paths.list_images(join(PASCALE_THESIS, "Prediction Video", "Original Frames")))),
    join(PASCALE_THESIS, "Predictions Natural Images", "GSP Predictions", "GrScPl0218_original.png"),
    join(PASCALE_THESIS, "Predictions Natural Images", "TOS Predictions", "tosB1_original.png"),
    join(PASCALE_THESIS, "PredictionExamples_SynData", "BSP-caterpillar-original.png"),
    join(PASCALE_THESIS, "PredictionExamples_SynData", "BSP-victor-original.png"),
    join(PASCALE_THESIS, "PredictionExamples_SynData", "GSP-caterpillar-original.png"),
    join(PASCALE_THESIS, "PredictionExamples_SynData", "GSP-victor-original.png"),
]

TEST_PATHS = utils.load_json(config.P_TEST_PATHS_GREEN)
p_regex = config.P_REGEX_REPLACE

EVAL_COLS = [
    "rgba_wmae",
    "rgba_alpha_loss_root",
    "trimap_alpha_loss_bg",
    "trimap_alpha_loss_fg",
    "trimap_alpha_loss_trans",
    "rgba_gradient_loss",
    "rgba_weighted_rgb_loss_root",
    "rgb_addon_rgb_correction_loss",
    "rgb_addon_rgb_constant_loss",
    "rgba_ssim",
]


if __name__ == "__main__":

    models = [
        # ([*GREEN_PATHS], PascaleModel(name="PascaleFinalGSP", path="/home/jannes/uni/jk-masterarbeit/pascales-thesis/stick/Trained Models/Finetuned GSP")),

        # ([*GREEN_PATHS], UNetAdvanced(path="/home/jannes/uni/jk-masterarbeit/saves_logs/0608-1536-UAdv_WA").load_model().set_name("plain")),
        # ([*GREEN_PATHS], UNetAdvanced(path="/home/jannes/uni/jk-masterarbeit/saves_logs/0608-1728-UAdv_SL_WA").load_model().set_name("SL")),
        # ([*GREEN_PATHS], UNetAdvanced(path="/home/jannes/uni/jk-masterarbeit/saves_logs/0608-1932-UAdv_TC_WA").load_model().set_name("TC")),
        # ([*GREEN_PATHS], UNetAdvanced(path="/home/jannes/uni/jk-masterarbeit/saves_logs/0608-2138-UAdv_WA_4").load_model().set_name("MP4")),
        # ([*GREEN_PATHS], UNetAdvanced(path="/home/jannes/uni/jk-masterarbeit/saves_logs/0609-0001-UAdv5_WA").load_model().set_name("MP5")),
        # ([*GREEN_PATHS], UNetAdvanced(path="/home/jannes/uni/jk-masterarbeit/saves_logs/0609-0139-UAdv4_TC_WA").load_model().set_name("MP4+TC")),
        # ([*GREEN_PATHS], UNetAdvanced(path="/home/jannes/uni/jk-masterarbeit/saves_logs/0609-0337-UAdv4_L_WA").load_model().set_name("MP4+LPRED")),

        # ([*GREEN_PATHS], UNetAdvanced(path="/home/jannes/uni/jk-masterarbeit/saves_logs/0609-1610-UAdv4_MIX_DBG_WA").load_model().set_name("ND")),
        # ([*GREEN_PATHS], UNetAdvanced(path="/home/jannes/uni/jk-masterarbeit/saves_logs/0607-1124-UAdv_SL_MIX_L_DBG_WA_AIMAG").load_model().set_name("T")),
        # ([*GREEN_PATHS], UNetAdvanced(name=None, path="/home/jannes/uni/jk-masterarbeit/saves_logs/0510-0105-UAdv_SL_DIM_TC_L_DBG_W_AIM").load_model().set_name("SD")),
        ([*GREEN_PATHS], UNetAdvanced(name=None, path="/home/jannes/uni/jk-masterarbeit/saves_logs/0514-2147-UAdv_SL_MIX_TC_L_DBG_W_AIMA").load_model().set_name("DIFF")),
    ]

    if not TEST_EVAL:
        scores = pd.DataFrame(columns=EVAL_COLS)
        model: Setup
        for eval_paths, model in models:
            name = model.name
            print(f"Evaluating {name}")

            model.nn_config.TRIMAP_KERNEL_SIZE = 15
            eval_ds = model.get_test_dataset(eval_paths, regex_replace)
            eval_ds = eval_ds.unbatch().batch(len(eval_paths))

            [(eval_x, eval_y)] = list(eval_ds.take(len(eval_paths)).as_numpy_iterator())

            score = model.evaluate(eval_x, eval_y, return_dict=True)
            eval_score = {key: (np.nan if key not in score else score[key]) for key in EVAL_COLS}
            scores.loc[name] = eval_score

            eval_pred = model.predict(eval_x)

            if type(eval_y) == dict:
                eval_y = eval_y["rgba"]
                eval_pred = eval_pred[0]

            for x, y, pred, path in zip(eval_x, eval_y, eval_pred, eval_paths):
                object_name = os.path.splitext(os.path.basename(path))[0]

                # wmse_loss = tf.math.reduce_mean(metrics.wmae(tf.expand_dims(y, 0), tf.expand_dims(pred, 0)))
                # print(f"{object_name} -- wmse: {wmse_loss}")

                if SAVE_IMAGES:
                    bg_color = path.split("/")[-2]
                    save_dir = join(RESULT_PATH, name, object_name)
                    os.makedirs(save_dir, exist_ok=True)
                    pylab.imsave(join(save_dir, f"{bg_color}.png"), x)
                    pylab.imsave(join(save_dir, f"{bg_color}_gt.png"), y[:,:,:4])
                    pylab.imsave(join(save_dir, f"{bg_color}_pred.png"), pred)
                    pylab.imsave(join(save_dir, f"{bg_color}_alpha_gt.png"), y[:,:,3], cmap=pylab.cm.gray)
                    pylab.imsave(join(save_dir, f"{bg_color}_alpha_pred.png"), pred[:,:,3], cmap=pylab.cm.gray)
                    pylab.imsave(join(save_dir, f"{bg_color}_alpha_diff.png"), np.abs(pred[:,:,3] - y[:,:,3]), cmap=pylab.cm.gray)
                    pylab.imsave(join(save_dir, f"{bg_color}_rgb_only_pred.png"), pred[:,:,:3])
                    pylab.imsave(join(save_dir, f"{bg_color}_rgb_diff.png"), np.concatenate((np.abs(pred[:,:,:3] - y[:,:,:3]), pred[:,:,3:]), -1))


        table_latex = re.sub(r' +', ' ', scores.to_latex())
        table_latex = re.sub(r'(\.0*[1-9]\d)\d*', r'\1', table_latex)
        print(table_latex)


    if TEST_EVAL:
        models = [(TEST_PATHS, m) for _, m in models]
        scores = pd.DataFrame(columns=EVAL_COLS)
        model: Setup
        for eval_paths, model in models:
            name = model.name
            print(f"Testing {name}")

            eval_ds = model.get_test_dataset(eval_paths, p_regex)
            eval_ds = eval_ds.unbatch().batch(len(eval_paths))

            [(eval_x, eval_y)] = list(eval_ds.take(len(eval_paths)).as_numpy_iterator())

            score = model.evaluate(eval_x, eval_y, return_dict=True)
            eval_score = {key: (np.nan if key not in score else score[key]) for key in EVAL_COLS}
            scores.loc[name] = eval_score

        table_latex = re.sub(r' +', ' ', scores.to_latex())
        table_latex = re.sub(r'(\.0*[1-9]\d)\d*', r'\1', table_latex)
        print(table_latex)


    if SAVE_IMAGES:
        models = [(P_EVALUATION_PATHS, m) for _, m in models]

        model: Setup
        for eval_paths, model in models:
            name = model.name
            print(f"Evaluating {name}")

            eval_ds = model.get_test_dataset(eval_paths, ("", "", ""))
            eval_ds = eval_ds.unbatch().batch(len(eval_paths))

            [(eval_x, eval_y)] = list(eval_ds.take(len(eval_paths)).as_numpy_iterator())

            eval_pred = model.predict(eval_x)

            if type(eval_y) == dict:
                eval_y = eval_y["rgba"]
                eval_pred = eval_pred[0]

            if SAVE_IMAGES:
                for x, y, pred, path in zip(eval_x, eval_y, eval_pred, eval_paths):
                    file_name = os.path.splitext(os.path.basename(path))[0]
                    dir_name = path.split("/")[-3]
                    save_dir = join(RESULT_PATH, name, dir_name, file_name)
                    os.makedirs(save_dir, exist_ok=True)
                    pylab.imsave(join(save_dir, f"input.png"), x)
                    pylab.imsave(join(save_dir, f"pred.png"), pred)
                    pylab.imsave(join(save_dir, f"alpha_pred.png"), pred[:,:,3], cmap=pylab.cm.gray)
                    pylab.imsave(join(save_dir, f"rgb_only_pred.png"), pred[:,:,:3])
