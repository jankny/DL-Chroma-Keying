import re
import numpy as np
import tensorflow as tf
import tensorflow_addons as tfa
import tensorflow_models as tfm
from os.path import join

import metrics
from UNet import UNet
from Setup import Setup
from config import NNConfig
from WarmUpScheduler import WarmUpScheduler


class UNetAdvanced(Setup):
    def __init__(self, name=None, path=None, **kwargs):
        self.nn_config = NNConfig()
        self.nn_config.TRIMAP_OUTPUT = True
        self.nn_config.TRIMAP_KERNEL_SIZE = 15
        self.nn_config.DIFFERENCE_OUTPUT = False
        self.nn_config.INPUT_SKIP_LINK = True
        self.nn_config.DIM_PRETRAINIG = False
        self.nn_config.DIM_BG_AUGMENTATION = True
        self.nn_config.DIM_NUM_EPOCHS = 40
        self.nn_config.NUM_EPOCHS = 80
        self.nn_config.TRANSPOSED_CONVOLUTIONS = True
        self.nn_config.DATA_AUGMENTATION = False
        self.nn_config.COLOR_AUGMENTATION = False
        self.nn_config.CROP_AUGMENTATION = False

        self.nn_config.RADAM = False
        self.nn_config.RADAM_LR = 5e-4
        self.nn_config.RADAM_MIN_LR = 5e-5

        self.nn_config.A_WARMUP = True
        self.nn_config.A_LR = 5e-4
        self.nn_config.A_MIN_LR = 5e-5

        self.nn_config.L1_LOSS = True
        self.nn_config.COLOR_SPILL_MIXED_PIXELS_LOSS = False
        self.nn_config.INITIAL_CHANNEL_NUM = 20
        self.nn_config.LEVELS = 3
        self.nn_config.CHANNEL_FACTOR = 2

        self.nn_config.MIXED_TRAIN_SET = True
        self.nn_config.SEMANTIC_BRANCH = True
        self.nn_config.AIM_RGB_ADDON = True
        self.nn_config.AIM_RGB_BRANCH = True
        self.nn_config.BATCH_SIZE = 16
        self.nn_config.GRADIENT_LOSS = True

        if self.nn_config.AIM_RGB_BRANCH: self.nn_config.LOSS_WEIGHTS = {
                "trimap": 0,
                "rgba": 0.2,
                "semantic_branch": 0.4,
                "rgb_addon": 0.4,
                "rgb_max_diff": 0,
            }

        self.nn_config.COMMENT = ""

        self.path = path
        self.kwargs = kwargs
        self.epochs_trained = None

        if not self.nn_config.SEMANTIC_BRANCH: self.nn_config.AIM_RGB_BRANCH = False
        if self.nn_config.MIXED_TRAIN_SET: self.nn_config.DIM_PRETRAINIG = False
        if not self.nn_config.DIM_PRETRAINIG and not self.nn_config.MIXED_TRAIN_SET: self.nn_config.DIM_BG_AUGMENTATION = False
        self.name = name if name is not None else "UAdv"
        self.name = self.name + self.get_model_flags()
        super().__init__(name=self.name, model=None, nn_config=self.nn_config)

    def get_model_flags(self):
        flags = ""
        flags = flags + f"{self.nn_config.LEVELS}"
        if self.nn_config.INPUT_SKIP_LINK: flags = flags + "_SL"
        if self.nn_config.DIM_PRETRAINIG: flags = flags + "_DIM"
        if self.nn_config.MIXED_TRAIN_SET: flags = flags + "_MIX"
        if self.nn_config.TRANSPOSED_CONVOLUTIONS: flags = flags + "_TC"
        if self.nn_config.DATA_AUGMENTATION: flags = flags + "_DA"
        if self.nn_config.COLOR_SPILL_MIXED_PIXELS_LOSS: flags = flags + "_L"
        if self.nn_config.DIM_BG_AUGMENTATION: flags = flags + "_DBG"
        if self.nn_config.BG_AUGMENTATION: flags = flags + "_BG"
        if self.nn_config.COLOR_AUGMENTATION: flags = flags + "_CA"
        if self.nn_config.CROP_AUGMENTATION: flags = flags + "_PA" # patches
        if self.nn_config.RADAM: flags = flags + "_W"
        if self.nn_config.A_WARMUP: flags = flags + "_WA"
        if self.nn_config.DIFFERENCE_OUTPUT: flags = flags + "_DIFF"
        if self.nn_config.SEMANTIC_BRANCH: flags = flags + f"_AIM{'A' if self.nn_config.AIM_RGB_BRANCH else ''}{'G' if self.nn_config.GRADIENT_LOSS else ''}"
        return flags

    def load_model(self, path=None):
        if path is None: path = self.path
        model_path = join(path, "model", "unet")
        self.load_nn_config(model_path)

        latest = tf.train.latest_checkpoint(join(path, "checkpoints"))
        model = tf.keras.models.load_model(model_path, compile=False, custom_objects={'tf': tf})
        model.load_weights(latest)

        model = self.compile_model(model)
        self.set_model(model)

        self.epochs_trained = int(re.search(r"cp-(\d+)", latest).group(1))

        return self

    def create_model(self, total_steps=0, compile=True):
        image_size = (*self.nn_config.IMAGE_SIZE, 3)
        model = UNet(
            image_size,
            input_skip_link=self.nn_config.INPUT_SKIP_LINK,
            transposed_convolutions=self.nn_config.TRANSPOSED_CONVOLUTIONS,
            trimap_output=self.nn_config.TRIMAP_OUTPUT,
            difference_output=self.nn_config.DIFFERENCE_OUTPUT,
            initial_channel_num=self.nn_config.INITIAL_CHANNEL_NUM,
            levels=self.nn_config.LEVELS,
            channel_factor=self.nn_config.CHANNEL_FACTOR,
            final_layer_activation="sigmoid",
            semantic_branch=self.nn_config.SEMANTIC_BRANCH,
            aim_rgb_addon=self.nn_config.AIM_RGB_BRANCH,
            **self.kwargs
        )

        if compile:
            model = self.compile_model(model, total_steps)
        self.set_model(model)
        return self

    def compile_model(self, model, total_steps=0):
        root = self.nn_config.L1_LOSS
        rgba_loss = (lambda x, y: metrics.wmse_cs_mp(x, y, root)) if self.nn_config.COLOR_SPILL_MIXED_PIXELS_LOSS else \
            (lambda x, y: metrics.wmse(x, y, root))

        optimizer = "adam"
        if self.nn_config.RADAM:
            optimizer = tfa.optimizers.RectifiedAdam(
                total_steps=total_steps,
                warmup_proportion=self.nn_config.RADAM_WARMUP,
                learning_rate=self.nn_config.RADAM_LR,
                min_lr=self.nn_config.RADAM_MIN_LR,
            )

        if self.nn_config.A_WARMUP:
            # linear_decay = tf.keras.optimizers.schedules.PolynomialDecay(
            #     initial_learning_rate=self.nn_config.A_LR,
            #     end_learning_rate=self.nn_config.A_MIN_LR,
            #     decay_steps=total_steps)
            # warmup_schedule = tfm.optimization.lr_schedule.LinearWarmup(
            #     warmup_learning_rate=0,
            #     after_warmup_lr_sched=linear_decay,
            #     warmup_steps=int(total_steps * self.nn_config.A_WARMUP_STEPS)
            # )
            warmup_schedule = WarmUpScheduler(
                initial_learning_rate=0,
                decay_steps=int((1 - self.nn_config.A_WARMUP_STEPS) * total_steps),
                alpha=self.nn_config.A_MIN_LR / self.nn_config.A_LR,
                warmup_target=self.nn_config.A_LR,
                warmup_steps=int(self.nn_config.A_WARMUP_STEPS * total_steps),
            )
            optimizer = tf.keras.optimizers.Adam(
                learning_rate=warmup_schedule
            )

        if getattr(self.nn_config, "AIM_RGB_BRANCH", False):
            alpha_loss = lambda x, y: metrics.alpha_gradient_loss(x, y, root) if self.nn_config.GRADIENT_LOSS else \
                (metrics.alpha_loss_root if root else metrics.alpha_loss4D)
            self.nn_config.LOSS_WEIGHTS = getattr(self.nn_config, "LOSS_WEIGHTS", None) or {
                "trimap": 0,
                "rgba": 0.2,
                "semantic_branch": 0.4,
                "rgb_addon": 0.2,
                "rgb_max_diff": 0.2,
            }

            model.compile(optimizer=optimizer, loss={
                "rgba": alpha_loss,
                "semantic_branch": metrics.semantic_branch_loss,
                "rgb_addon": metrics.rgb_addon_loss,
                "rgb_max_diff": metrics.rgb_max_diff_loss
            }, metrics={
                "trimap": [metrics.alpha_loss_bg, metrics.alpha_loss_trans, metrics.alpha_loss_fg],
                "rgba": [metrics.wmse, metrics.wmae, metrics.alpha_loss4D, metrics.weighted_rgb_loss, metrics.gradient_loss, metrics.alpha_loss_root, metrics.weighted_rgb_loss_root, metrics.ssim],
                "semantic_branch": metrics.semantic_branch_loss,
                "rgb_addon": [metrics.rgb_addon_loss, metrics.rgb_correction_loss, metrics.rgb_constant_loss],
            }, loss_weights=self.nn_config.LOSS_WEIGHTS)
        elif getattr(self.nn_config, "SEMANTIC_BRANCH", False):
            model.compile(optimizer=optimizer, loss={
                "rgba": rgba_loss,
                "semantic_branch": metrics.semantic_branch_loss,
            }, metrics={
                "trimap": [metrics.alpha_loss_bg, metrics.alpha_loss_trans, metrics.alpha_loss_fg],
                "rgba": [metrics.wmse, metrics.wmae, metrics.alpha_loss4D, metrics.weighted_rgb_loss, metrics.gradient_loss, metrics.alpha_loss_root, metrics.weighted_rgb_loss_root, metrics.ssim],
                "semantic_branch": metrics.semantic_branch_loss,
                "rgb_addon": [metrics.rgb_correction_loss, metrics.rgb_constant_loss],
            }, loss_weights={
                "trimap": 0,
                "rgba": 0.5,
                "semantic_branch": 0.5,
            })
        elif self.nn_config.TRIMAP_OUTPUT:
            if self.nn_config.MIXED_TRAIN_SET:
                loss = {
                    "rgba": metrics.alpha_loss_root if root else metrics.alpha_loss4D,
                    "rgb_addon": metrics.rgb_addon_rgb_only
                }
            else:
                loss = {
                    "rgba": rgba_loss,
                }
            model.compile(optimizer=optimizer, loss=loss, metrics={
                "trimap": [metrics.alpha_loss_bg, metrics.alpha_loss_trans, metrics.alpha_loss_fg],
                "rgba": [metrics.wmse, metrics.wmae, metrics.alpha_loss4D, metrics.weighted_rgb_loss, metrics.gradient_loss, metrics.alpha_loss_root, metrics.weighted_rgb_loss_root, metrics.ssim],
                "rgb_addon": [metrics.rgb_correction_loss, metrics.rgb_constant_loss],
            }, loss_weights={
                "trimap": 0,
                "rgba": 1,
                "rgb_addon": 1,
            })
        else:
            model.compile(
                optimizer='adam',
                loss=rgba_loss,
                metrics=[metrics.wmse, metrics.wmae, metrics.alpha_loss4D, metrics.weighted_rgb_loss, metrics.gradient_loss]
            )
        return model

    def predict(self, x, **kwargs):
        outputs = self.model.predict(x)
        return self.outputs_to_img(x, outputs, self.nn_config)

    @staticmethod
    def outputs_to_img(inputs, outputs, nn_config:NNConfig):
        images = np.array(outputs) if not nn_config.TRIMAP_OUTPUT else np.array(outputs[0])
        if nn_config.DIFFERENCE_OUTPUT:
            images[...,:3] = inputs + images[...,:3]
            images = np.clip(images, 0, 1)

        if nn_config.TRIMAP_OUTPUT:
            outputs[0] = images
        else:
            outputs = images
        return outputs


def model_build_func(input_shape):
    unet = UNetAdvanced(conv_activation="linear")
    unet.nn_config.IMAGE_SIZE = (input_shape[0], input_shape[1])
    unet.create_model(compile=False)
    return unet.model

if __name__ == "__main__":
    unet = UNetAdvanced()

