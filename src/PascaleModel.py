import os
import json
import keras
import pylab
import numpy as np
from pathlib import Path
from os.path import join
from PIL import Image
from typing import Tuple
from matplotlib import pyplot as plt

from pascale import final_training_setup as setup, versioning, dl_utils as utils
from config import NNConfig
from Setup import Setup
from tf_dataset import load_sample, load_image

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
import tensorflow as tf
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '0'


class PascaleModel(Setup):
    name = "PascaleModel"

    def __init__(self, path, name=None):
        self.path = path
        self.model, self.params = load_pascale_model(path)
        self.nn_config = self.pascale_config()
        super().__init__(name=name, model=self.model, nn_config=self.nn_config)

    def pascale_config(self) -> NNConfig:
        nn_config = NNConfig(trimap=True)
        nn_config.IMAGE_SIZE = (self.params["imageSize"], self.params["imageSize"])
        return nn_config

    def get_test_dataset(self, path_list, gt_replace_regex, **kwargs) -> tf.data.Dataset:
        test_ds = tf.data.Dataset.from_tensor_slices(path_list)
        test_ds = (test_ds
                   .map(lambda x: load_sample(x, nn_config=self.nn_config, regex_replace=gt_replace_regex),
                        num_parallel_calls=tf.data.AUTOTUNE)
                   .batch(len(path_list))
                   .prefetch(tf.data.AUTOTUNE)
        )
        return test_ds

def load_pascale_model(model_path) -> Tuple[keras.Model, dict]:

    params_file = join(model_path, "input_params.json")
    params = versioning.get_params(params_file)


    metric_mapping = setup.get_metrics(params)
    model = utils.load_model(model_path)

    # adam = tf.keras.optimizers.Adam(**params["optimizer_spec"])
    adam = tf.keras.optimizers.Adam()

    if "wmse_acc" in metric_mapping:
        del metric_mapping["wmse_acc"]

    model.compile(optimizer=adam,
                  loss=getattr(utils, params["loss_type"]),
                  metrics=list(metric_mapping.values()))

    return model, params

if __name__ == "__main__":
    # model = load_pascale_model("/home/jannes/uni/jk-masterarbeit/pascales-thesis/stick/Trained Models/Finetuned BSP")
    pascaleModel = PascaleModel("/home/jannes/uni/jk-masterarbeit/pascales-thesis/stick/Trained Models/Finetuned GSP")
    pascaleModel.model.summary()

    test_img = Image.open("/home/jannes/uni/jk-masterarbeit/pascales-thesis/System Evaluation/Predictions Natural Images/GSP Predictions/GrScPl0218_original.png")
    test_img = np.expand_dims(np.array(test_img)[:,:,0:3], axis=0) / 255
    test_pred = pascaleModel.predict(test_img)[0]
    plt.imshow(test_pred)
    plt.show()
