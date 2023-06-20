import os
import json
import random
import shutil
import numpy as np
import tensorflow as tf # https://stackoverflow.com/questions/55549257/how-to-run-tensorflow-gpu-in-pycharm
from imutils import paths
from datetime import datetime
from contextlib import redirect_stdout

import utils
import plots
import config
from UNetAdvanced import UNetAdvanced
from tensorboard_utils import TensorboardPlotsCallback, EvaluationSetCallback, LearningRateLogger

tf.random.set_seed(config.RANDOM_SEED)
random.seed(config.RANDOM_SEED)
np.random.seed(config.RANDOM_SEED)


def shuffle(l):
    random.shuffle(l)
    return l


if __name__ == "__main__":
    physical_devices = tf.config.list_physical_devices('GPU')
    try:
        tf.config.experimental.set_memory_growth(physical_devices[0], True)
    except:
        # Invalid device or cannot modify virtual devices once initialized.
        pass

    # remain_training = "/home/jannes/uni/remote-jk-masterarbeit/saves_logs/0510-0105-UAdv_SL_DIM_TC_L_DBG_W_AIM"
    remain_training = None
    model = UNetAdvanced(path=remain_training)
    if remain_training:
        model = model.load_model()
    nn_config = model.nn_config

    now = datetime.now()
    log_name = f"{now.strftime('%m%d-%H%M')}-{model.name}"
    log_dir = os.path.join(config.LOG_DIR, log_name)


    p_regex = config.P_REGEX_REPLACE
    train_paths = [] \
                  + utils.load_json(config.P_TRAIN_PATHS_GREEN) \
                  + shuffle(utils.load_json(config.P_PRETRAIN_PATHS_GREEN))[:2000] \
                  # + utils.load_json(config.P_PRETRAIN_PATHS_GREEN) \

    val_paths = [] \
                + utils.load_json(config.P_VAL_PATHS_GREEN) \
                + shuffle(utils.load_json(config.P_PRETRAIN_VAL_PATHS_GREEN))[:500] \
                # + utils.load_json(config.P_PRETRAIN_VAL_PATHS_GREEN) \

    test_paths = utils.load_json(config.P_TEST_PATHS_GREEN) \

    dim_train_paths = list(paths.list_images(config.DIM_TRAIN_PATHS))
    dim_test_paths = list(paths.list_images(config.DIM_TEST_PATHS))
    dim_gt_regex = config.DIM_REGEX_REPLACE

    benchmark_paths = config.GREEN_PATHS

    print("Loading DIM DS..")
    if nn_config.MIXED_TRAIN_SET: nn_config.DIM_TRAIN_WEIGHTS = {"rgba": 0., "trimap": 1., "semantic_branch": 1., "rgb_addon": 0., "rgb_max_diff": 0.}
    dim_train_ds = model.get_train_dataset(dim_train_paths, dim_gt_regex, nn_config.DIM_BG_AUGMENTATION,
                                           sample_weight=nn_config.DIM_TRAIN_WEIGHTS, crop_augmentation=nn_config.CROP_AUGMENTATION)
    dim_test_ds = model.get_test_dataset(dim_test_paths, dim_gt_regex, nn_config.DIM_BG_AUGMENTATION)
    train_samples = list(dim_train_ds.take(1).as_numpy_iterator())[0]

    if nn_config.AIM_RGB_ADDON:
        plots.evaluation_plt(train_samples[0], train_samples[1]["rgba"], trimap_gt=train_samples[1]["trimap"], rgb_max_diff_gt=train_samples[1]["rgb_addon"][...,:1])
    else:
        plots.plot_samples(train_samples[0], train_samples[1])


    print("Loading Train DS..")
    if nn_config.MIXED_TRAIN_SET and nn_config.AIM_RGB_ADDON: nn_config.TRAIN_WEIGHTS = {"rgba": 1., "trimap": 1., "semantic_branch": 1., "rgb_addon": 1., "rgb_max_diff": 1.}
    train_ds = model.get_train_dataset(train_paths, p_regex, nn_config.BG_AUGMENTATION, nn_config.COLOR_AUGMENTATION,
                                       sample_weight=nn_config.TRAIN_WEIGHTS, crop_augmentation=nn_config.CROP_AUGMENTATION)
    train_samples = list(train_ds.take(1).as_numpy_iterator())[0]

    if nn_config.AIM_RGB_ADDON:
        plots.evaluation_plt(train_samples[0], train_samples[1]["rgba"], trimap_gt=train_samples[1]["trimap"], rgb_max_diff_gt=train_samples[1]["rgb_addon"][...,:1])
    else:
        plots.plot_samples(train_samples[0], train_samples[1])

    print("Loading Validation DS..")
    val_ds = model.get_test_dataset(val_paths, p_regex, nn_config.BG_AUGMENTATION)

    print("Loading Test DS..")
    test_ds = model.get_test_dataset(test_paths, p_regex, nn_config.BG_AUGMENTATION)

    if nn_config.MIXED_TRAIN_SET and nn_config.AIM_RGB_ADDON:
        train_ds = train_ds.concatenate(dim_train_ds)
        test_ds = test_ds.concatenate(dim_test_ds)
        train_ds.shuffle(tf.data.experimental.cardinality(train_ds))





    print("Finished Data loading")
    # -------------------------------------------

    epochs_to_train = nn_config.NUM_EPOCHS if model.epochs_trained is None else nn_config.NUM_EPOCHS - model.epochs_trained
    total_steps = (int(len(train_paths)/nn_config.BATCH_SIZE))*(epochs_to_train)
    model.create_model(total_steps=total_steps)
    model.model.summary()
    model.name = log_name

    print("Start Training")

    tensorboard_callback = tf.keras.callbacks.TensorBoard(log_dir=log_dir, histogram_freq=1)

    cp_callback = tf.keras.callbacks.ModelCheckpoint(
        filepath=os.path.join(log_dir, "checkpoints", "cp-{epoch:04d}.ckpt"),
        verbose=1,
        save_weights_only=True
    )

    random.shuffle(train_paths)
    random.shuffle(val_paths)
    random.shuffle(test_paths)
    benchmark_samples = [p for p in benchmark_paths if any(s in p for s in ["cactus", "colors3", "teddy", "9_doll", "fox_spill"])]
    plot_samples = [
        *model.get_test_dataset(train_paths[:4], p_regex, unbatch=True).take(4).as_numpy_iterator(),
        *model.get_test_dataset(val_paths[:4], p_regex, unbatch=True).take(4).as_numpy_iterator(),
        *model.get_test_dataset(test_paths[:4], p_regex, unbatch=True).take(4).as_numpy_iterator(),
        *model.get_test_dataset(benchmark_samples, config.BENCHMARK_REGEX_REPLACE, unbatch=True).take(len(benchmark_samples)).as_numpy_iterator(),
    ]
    tensorboardPlotsCallback = TensorboardPlotsCallback(
        predict_samples=plot_samples,
        logdir=os.path.join(log_dir, "plots"),
        title="val samples",
        nn_config=nn_config,
    )

    benchmark_callback = EvaluationSetCallback(
        eval_ds=model.get_test_dataset(benchmark_paths, config.BENCHMARK_REGEX_REPLACE),
        logdir=os.path.join(log_dir, "benchmark")
    )

    test_callback = EvaluationSetCallback(
        eval_ds=test_ds,
        logdir=os.path.join(log_dir, "test")
    )

    learningRateLogger = LearningRateLogger()

    if config.MODE == config.MODES.train:
        model_log_dir = os.path.join(log_dir, "model", "unet")
        os.makedirs(model_log_dir, exist_ok=True)
        model.save(model_log_dir)
        tf.keras.utils.plot_model(model.model,
                                  to_file=os.path.join(log_dir, "model", "summary.png"),
                                  show_shapes=True,
                                  show_layer_names=True,
                                  show_layer_activations=True,
                                  )
        with open(os.path.join(log_dir, 'model', 'modelsummary.txt'), 'w') as f:
            with redirect_stdout(f):
                model.model.summary()

        if nn_config.DIM_PRETRAINIG and (model.epochs_trained or -1) < 0:
            h_pre = model.fit(
                x=dim_train_ds,
                validation_data=dim_test_ds,
                epochs=0,
                initial_epoch=-1*nn_config.DIM_NUM_EPOCHS,
                callbacks=[
                    learningRateLogger,
                    tensorboard_callback,
                    benchmark_callback,
                    test_callback,
                    tensorboardPlotsCallback,
                ]
            )

        callbacks = [
            # learningRateSchedule,
            # learningRateLogger,
            tensorboard_callback,
            cp_callback,
            tensorboardPlotsCallback,
            benchmark_callback,
            test_callback,
        ]

        H = model.fit(
            x=train_ds,
            validation_data=val_ds,
            epochs=nn_config.NUM_EPOCHS,
            callbacks=callbacks,
            initial_epoch=model.epochs_trained or 0
        )

    else:
        latest = tf.train.latest_checkpoint("../saves_logs/20221111-175454/checkpoints")
        # model = tf.keras.models.load_model("../saves_logs/20221111-175454/model/unet")
        model.load_weights(latest)

    test_samples = test_ds.take(1)
    test_outputs = model.predict(test_samples)
    test_samples = list(test_samples.as_numpy_iterator())[0]

    val_samples = val_ds.take(1)
    val_outputs = model.predict(val_samples)
    val_samples = list(val_samples.as_numpy_iterator())[0]

    print("finished")
