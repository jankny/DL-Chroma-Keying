import numpy as np
import tensorflow as tf
from tensorflow import keras
from keras.models import Model
from keras_pyramid_pooling_module import PyramidPoolingModule
from keras.layers import Input, Conv2D, MaxPooling2D, UpSampling2D, Concatenate, BatchNormalization, Conv2DTranspose


def add_conv_block(x, channel_num, bn, conv_activation, transposed_convolutions=None):
    x = Conv2D(channel_num, 3, activation=conv_activation, padding="same",
               kernel_initializer='he_normal',
               bias_initializer='zeros')(x)
    x = BatchNormalization()(x) if bn else x
    x = Conv2D(channel_num, 3, activation=conv_activation, padding="same",
               kernel_initializer='he_normal',
               bias_initializer='zeros')(x)
    x = BatchNormalization()(x) if bn else x
    return x


def add_down_block(x, channel_num, bn, conv_activation):
    x = add_conv_block(x, channel_num, bn, conv_activation)
    skip_conn = x
    x = MaxPooling2D(pool_size=2, padding="valid")(x)
    return [x, skip_conn]


def add_up_block(x, skip_conn, channel_num, bn, conv_activation, transposed_convolutions=False):
    if transposed_convolutions:
        x = Conv2DTranspose(channel_num, (1,1), strides=(2,2))(x)
        # x = Conv2DTranspose(channel_num, (3,3), strides=(2,2),padding="same", activation=conv_activation, kernel_initializer='he_normal')(x)
    else:
        x = UpSampling2D(size=2)(x)
    x = Concatenate(axis=-1)([x, skip_conn])
    return add_conv_block(x, channel_num, bn, conv_activation)


def SemanticBranch(x, channel_num, levels, **kwargs):
    ppm = PyramidPoolingModule(num_filters=int(channel_num), kernel_size=(1,1))(x)
    x = Concatenate(axis=-1)([x, ppm])
    x = add_conv_block(x, channel_num, **kwargs)
    x = SqueezeAndExcitation(x)
    for level in range(levels):
        channel_num = int(channel_num / 2)
        ppm = conv_up_ppm(ppm, channel_num)
        up = add_up_block(x, ppm, channel_num, **kwargs)
        x = SqueezeAndExcitation(up)

    x_mean = tf.math.reduce_mean(up, axis=-1, keepdims=True)
    x_max = tf.math.reduce_max(up, axis=-1, keepdims=True)
    features = Concatenate(axis=-1)([x_mean, x_max])
    attention_map = Conv2D(1, 7, padding="same", activation="sigmoid",
                         kernel_initializer='glorot_normal', bias_initializer='zeros', name='attention_map')(features)

    semantic_output = Conv2D(3, 1, padding="same", activation="softmax",
                         kernel_initializer='glorot_normal', bias_initializer='zeros', name='semantic_output')(x)
    return attention_map, semantic_output


def conv_up_ppm(x, out_channels):
    x = Conv2D(out_channels, 3, padding="same", activation="relu",
               kernel_initializer='he_normal', bias_initializer='zeros')(x)
    x = BatchNormalization()(x)
    x = UpSampling2D(size=2, interpolation="bilinear")(x)
    return x


def SqueezeAndExcitation(inputs, ratio=16):
    b, _, _, c = inputs.shape
    x = keras.layers.GlobalAveragePooling2D()(inputs)
    x = keras.layers.Dense(c//ratio, activation="relu", use_bias=False, kernel_initializer='he_normal')(x)
    x = keras.layers.Dense(c, activation="sigmoid", use_bias=False)(x)
    x = inputs * tf.reshape(x, [-1, 1, 1, c])
    return x


def RGBBranch(x, channel_num, skip_connections, **kwargs):
    rgb_x = x
    for skip_connection in skip_connections[::-1]:
        channel_num = int(channel_num / 2)
        rgb_x = add_up_block(rgb_x, skip_connection, channel_num, **kwargs)

    return rgb_x


def UNet(input_shape,
              output_channels=4,
              initial_channel_num=11,
              levels=3, bn=True,
              final_layer_activation="sigmoid",
              channel_factor=1.5,
              input_skip_link=False,
              transposed_convolutions=False,
              trimap_output=False,
              difference_output=False,
              conv_activation="relu",
              semantic_branch=False,
              aim_rgb_addon=False,
         ):
    skip_connections = list()
    inputs = Input(shape=input_shape, name="input")
    x = inputs
    channel_num = initial_channel_num
    for level in range(levels):
        x, skip = add_down_block(x, channel_num, bn, conv_activation)
        skip_connections.append(skip)
        channel_num = int(np.ceil(channel_num * channel_factor))
    x = add_conv_block(x, channel_num, bn, conv_activation)
    bottleneck = x
    channel_num_max = channel_num

    for skip_connection in skip_connections[::-1]:
        channel_num = int(channel_num / 2)
        x = add_up_block(x, skip_connection, channel_num, bn, conv_activation, transposed_convolutions=transposed_convolutions)

    if input_skip_link:
        x = Concatenate(axis=-1)([x, inputs])

    if final_layer_activation == "linear":
        final_layer_activation = tf.keras.layers.ReLU(max_value=1.0)

    outputs = []

    if difference_output:
        predictions_diff = Conv2D(3, 1, padding="same", activation="tanh",
                         kernel_initializer='he_normal', bias_initializer='zeros', name='diff')(x)
        predictions_alpha = Conv2D(1, 1, padding="same", activation=final_layer_activation,
                         kernel_initializer='glorot_normal', bias_initializer='zeros', name='alpha')(x)
        predictions_rgba = Concatenate(axis=-1, name="rgba")([predictions_diff, predictions_alpha])
        outputs.append(predictions_rgba)
    elif semantic_branch:
        attention_map, semantic_output = SemanticBranch(bottleneck, channel_num_max, levels=levels, bn=bn,
                    conv_activation=conv_activation, transposed_convolutions=transposed_convolutions)

        x = x + x * attention_map

        guided_alpha = Conv2D(1, 3, padding="same", activation="sigmoid",
                              kernel_initializer='glorot_normal', bias_initializer='zeros', name='guided_alpha')(x)

        index = tf.cast(tf.expand_dims(tf.math.argmax(semantic_output, axis=-1), -1), tf.float32)
        between_mask = tf.identity(index)
        between_mask = tf.where(tf.equal(between_mask, 2), tf.zeros_like(between_mask), between_mask)
        fg_mask = tf.identity(index)
        fg_mask = tf.where(tf.equal(fg_mask, 1), tf.zeros_like(fg_mask), fg_mask)
        fg_mask = tf.where(tf.equal(fg_mask, 2), tf.ones_like(fg_mask), fg_mask)

        alphamap = guided_alpha * between_mask + fg_mask

        if aim_rgb_addon:
            rgb_x = x

            rgb_max_diff = Conv2D(1, 3, padding="same", activation="sigmoid",
                                  kernel_initializer='glorot_normal', bias_initializer='zeros', name='rgb_max_diff')(rgb_x)
            rgb_features = Concatenate(axis=-1)([rgb_x, *([inputs] if input_skip_link else []), rgb_max_diff])
            guided_rgb_diff = Conv2D(3, 3, padding="same", activation="tanh",
                                  kernel_initializer='he_normal', bias_initializer='zeros', name='guided_rgb_diff')(rgb_features)

            guided_rgb_diff = guided_rgb_diff * rgb_max_diff
            guided_rgb = inputs + guided_rgb_diff
            guided_rgb = tf.clip_by_value(guided_rgb, 0, 1, name="guided_rgb")
            rgb_addon = Concatenate(axis=-1, name="rgb_addon")([guided_rgb, rgb_max_diff, guided_rgb_diff, inputs])
        else:
            guided_rgb = Conv2D(3, 3, padding="same", activation="sigmoid",
                                      kernel_initializer='glorot_normal', bias_initializer='zeros', name='guided_rgb')(x)

        predictions_rgba = Concatenate(axis=-1, name="rgba")([guided_rgb, alphamap])
        semantic_out = Concatenate(axis=-1, name="semantic_branch")([semantic_output, guided_alpha])

        outputs.extend([predictions_rgba, semantic_out])
        if aim_rgb_addon:
            outputs.extend([rgb_addon, rgb_max_diff])
        else:
            outputs.extend([Concatenate(axis=-1, name="rgb_addon")([guided_rgb[..., :2], guided_rgb[..., 2:3]])])
    else:
        predictions_rgba = Conv2D(output_channels, 1, padding="same", activation=final_layer_activation,
                         kernel_initializer='glorot_normal', bias_initializer='zeros', name='rgba')(x)
        rgb = Concatenate(axis=-1, name="rgb_addon")([predictions_rgba[..., :2], predictions_rgba[..., 2:3]]) # tf.identity does not work for renaming
        outputs.extend([predictions_rgba, rgb])

    if trimap_output:
        predictions_trimap = Conv2D(3, 1, padding="same", activation="softmax",
                                    kernel_initializer='glorot_normal', bias_initializer='zeros', name='trimap_output')(x)
        predictions_trimap = Concatenate(axis=-1, name="trimap")([predictions_trimap, predictions_rgba[..., 3:4]])

        outputs.append(predictions_trimap)



    model = Model(inputs=inputs, outputs=outputs)
    return model
