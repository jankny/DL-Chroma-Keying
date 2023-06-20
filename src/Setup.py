import os
import json
import tensorflow as tf
from tensorflow import keras

from config import NNConfig
from tf_dataset import load_sample, augment_sample, finalize, append_input_to_gt, \
    augment_background, augment_color, diff_as_target, add_sample_weight


class Setup():
    def __init__(self, name=None, model=None, nn_config=None):
        self.model: keras.Model = model if model is not None else keras.Model()
        self.nn_config: NNConfig = nn_config if nn_config is not None else NNConfig()
        self.name = name if name is not None else "Default"
        self.nn_config.NAME = self.name

        self.fit = self.model.fit
        self.evaluate = self.model.evaluate
        self.load_weights = self.model.load_weights

    def predict(self, x, **kwargs):
        return self.model.predict(x, **kwargs)

    def set_model(self, model:keras.Model):
        self.model = model
        self.fit = self.model.fit
        self.evaluate = self.model.evaluate
        self.load_weights = self.model.load_weights

    def set_name(self, name:str):
        self.name = name
        return self

    def create_model(self):
        return self.model

    def get_model(self) -> keras.Model:
        return self.model

    def get_config(self) -> NNConfig:
        return self.nn_config

    def get_test_dataset(self, path_list, gt_replace_regex, bg_augmentation=False, unbatch=False) -> tf.data.Dataset:
        test_ds = tf.data.Dataset.from_tensor_slices(path_list)
        test_ds = test_ds.map(lambda x: load_sample(x, nn_config=self.nn_config, regex_replace=gt_replace_regex),
                        num_parallel_calls=tf.data.AUTOTUNE)
        test_ds = test_ds.map(lambda x, y: finalize(x, y, self.nn_config), num_parallel_calls=tf.data.AUTOTUNE)

        if bg_augmentation:
            test_ds = test_ds.map(lambda x, y: augment_background(x, y, self.nn_config), num_parallel_calls=1)

        if self.nn_config.DIFFERENCE_OUTPUT:
            test_ds = test_ds.map(lambda x, y: diff_as_target(x, y, self.nn_config), num_parallel_calls=tf.data.AUTOTUNE)

        if self.nn_config.COLOR_SPILL_MIXED_PIXELS_LOSS:
            test_ds = test_ds.map(lambda x, y: append_input_to_gt(x, y, self.nn_config), num_parallel_calls=tf.data.AUTOTUNE)

        test_ds = test_ds.batch(self.nn_config.BATCH_SIZE)
        test_ds = test_ds.prefetch(tf.data.AUTOTUNE)

        if unbatch:
            test_ds = test_ds.unbatch()

        return test_ds

    def get_train_dataset(self, path_list, gt_replace_regex, bg_augmentation=False, color_augmentation=False,
                          sample_weight=None, crop_augmentation=False) -> tf.data.Dataset:
        train_ds = tf.data.Dataset.from_tensor_slices(path_list)
        train_ds = train_ds.shuffle(len(path_list))
        train_ds = train_ds.map(lambda x: load_sample(x, nn_config=self.nn_config, regex_replace=gt_replace_regex, crop_augmentation=crop_augmentation),
                                num_parallel_calls=1 if crop_augmentation else tf.data.AUTOTUNE)

        if self.nn_config.DATA_AUGMENTATION:
            train_ds = train_ds.map(lambda x, y: augment_sample(x, y, self.nn_config), num_parallel_calls=1)

        if bg_augmentation:
            train_ds = train_ds.map(lambda x, y: augment_background(x, y, self.nn_config), num_parallel_calls=1)

        if color_augmentation:
            train_ds = train_ds.map(lambda x, y: augment_color(x, y, self.nn_config), num_parallel_calls=1)

        if self.nn_config.DIFFERENCE_OUTPUT:
            train_ds = train_ds.map(lambda x, y: diff_as_target(x, y, self.nn_config), num_parallel_calls=tf.data.AUTOTUNE)

        train_ds = train_ds.map(lambda x, y: finalize(x, y, self.nn_config), num_parallel_calls=tf.data.AUTOTUNE)

        if self.nn_config.COLOR_SPILL_MIXED_PIXELS_LOSS:
            train_ds = train_ds.map(lambda x, y: append_input_to_gt(x, y, self.nn_config), num_parallel_calls=tf.data.AUTOTUNE)

        if sample_weight:
            train_ds = train_ds.map(lambda x, y,: add_sample_weight(x, y, sample_weight), num_parallel_calls=tf.data.AUTOTUNE)

        # train_ds = train_ds.cache()
        train_ds = train_ds.batch(self.nn_config.BATCH_SIZE)
        train_ds = train_ds.prefetch(tf.data.AUTOTUNE)
        return train_ds

    def save_nn_config(self, path):
        self.nn_config.NAME = self.name
        with open(os.path.join(path, 'nn_config.json'), 'w') as fp:
            json.dump(self.nn_config.__dict__, fp, indent=4)

    def save(self, path, **kwargs):
        self.model.save(path, **kwargs)
        self.save_nn_config(path)

    def load_nn_config(self, path): # path is the path to the directory of the nn_config.json
        with open(os.path.join(path, 'nn_config.json'), 'r') as fp:
            nn_config_dict = json.load(fp)
        default_config = NNConfig()
        self.nn_config.__dict__ = {**default_config.__dict__, **nn_config_dict}
        self.name = self.nn_config.NAME
