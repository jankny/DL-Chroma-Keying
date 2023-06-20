import io
import plots
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt

from UNetAdvanced import UNetAdvanced
from config import NNConfig


class TensorboardPlotsCallback(tf.keras.callbacks.Callback):
    def __init__(self, predict_samples, logdir, title=None, nn_config:NNConfig=None):
        super(TensorboardPlotsCallback, self).__init__()
        self.predict_samples = predict_samples
        self.logdir = logdir
        self.title = title
        self.file_writer = tf.summary.create_file_writer(logdir)
        self.inputs = [i for i, _ in predict_samples]
        self.ground_truth = [g["rgba"] for _, g in predict_samples]
        self.nn_config = nn_config

    def on_epoch_end(self, epoch, logs=None):
        outputs = self.model.predict(tf.convert_to_tensor(self.inputs))
        outputs = UNetAdvanced.outputs_to_img(self.inputs, outputs, self.nn_config)
        title = "{} at epoch {}".format(self.title, epoch)

        if self.nn_config.SEMANTIC_BRANCH:
            trimap_true = [g["trimap"] for _, g in self.predict_samples]

            rgba = outputs[0]
            semantic_output = outputs[1]
            trimap_pred = semantic_output[..., :3]

            if self.nn_config.AIM_RGB_BRANCH:
                rgb_max_diff_true = [g["rgb_addon"][..., :1] for _, g in self.predict_samples]
                rgb_addon = outputs[2]
                rgb_max_diff_pred = rgb_addon[..., 3:4]
            else:
                rgb_max_diff_true = None
                rgb_max_diff_pred = None


            fig = plots.evaluation_plt(self.inputs, self.ground_truth, rgba,
                                       trimap_gt=trimap_true, trimap_pred=trimap_pred,
                                       rgb_max_diff_gt=rgb_max_diff_true, rgb_max_diff_pred=rgb_max_diff_pred,
                                       title=title, show=False,
                                       n=len(self.inputs))
        else:
            fig = plots.evaluation_plt(self.inputs, self.ground_truth, outputs[0], title=title, show=False, n=len(self.inputs))

        with self.file_writer.as_default():
            tf.summary.image(title, plot_to_image(fig), step=epoch)


class EvaluationSetCallback(tf.keras.callbacks.Callback):
    def __init__(self, eval_ds, logdir):
        super(EvaluationSetCallback, self).__init__()
        self.eval_ds = eval_ds
        self.logdir = logdir
        self.file_writer = tf.summary.create_file_writer(logdir)

    def on_epoch_end(self, epoch, logs=None):
        results = self.model.evaluate(self.eval_ds)
        for score, metric in zip(results, self.model.metrics_names):
            with self.file_writer.as_default():
                tf.summary.scalar(f"epoch_{metric}", score, step=epoch)


class LearningRateLogger(tf.keras.callbacks.Callback):
    def __init__(self):
        super().__init__()
        self._supports_tf_logs = True

    def on_epoch_end(self, epoch, logs=None):
        if logs is None or "learning_rate" in logs:
            return
        logs["learning_rate"] = self.model.optimizer.lr


def plot_to_image(figure):
    """
    Copyright 2019 The TensorFlow Authors.

    Licensed under the Apache License, Version 2.0 (the "License");
    you may not use this file except in compliance with the License.
    You may obtain a copy of the License at

        http://www.apache.org/licenses/LICENSE-2.0

    Unless required by applicable law or agreed to in writing, software
    distributed under the License is distributed on an "AS IS" BASIS,
    WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
    See the License for the specific language governing permissions and
    limitations under the License.

    Converts the matplotlib plot specified by 'figure' to a PNG image and
    returns it. The supplied figure is closed and inaccessible after this call."""

    # Save the plot to a PNG in memory.
    buf = io.BytesIO()
    plt.savefig(buf, format='png')
    # Closing the figure prevents it from being displayed directly inside
    # the notebook.
    plt.close(figure)
    buf.seek(0)
    # Convert PNG buffer to TF image
    image = tf.image.decode_png(buf.getvalue(), channels=4)
    # Add the batch dimension
    image = tf.expand_dims(image, 0)
    return image
