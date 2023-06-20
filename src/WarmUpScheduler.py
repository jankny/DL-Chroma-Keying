import math
import numpy as np
import tensorflow as tf


class WarmUpScheduler(tf.keras.optimizers.schedules.LearningRateSchedule):
    def __init__(self,
                initial_learning_rate,
                decay_steps,
                alpha=0.0,
                name=None,
                warmup_target=None,
                warmup_steps=0):

        super(WarmUpScheduler, self).__init__()
        self.initial_learning_rate = initial_learning_rate
        self.decay_steps = decay_steps
        self.alpha = alpha
        self.name = name
        self.warmup_target = warmup_target
        self.warmup_steps = warmup_steps

        self.initial_decay_lr = self.warmup_target

    def __call__(self, step):

        lr = tf.cond(tf.cast(step, tf.float32) < tf.cast(self.warmup_steps, tf.float32),
                lambda :self.warmup_learning_rate(step),
                lambda :self.decayed_learning_rate(step))
        return lr

    def decayed_learning_rate(self, step):
        decay_steps = tf.cast(self.decay_steps, tf.float32)
        step = tf.minimum(tf.cast(step, tf.float32), decay_steps)
        cosine_decay = 0.5 * (1 + tf.math.cos(math.pi * step / decay_steps))
        decayed = (1 - self.alpha) * cosine_decay + self.alpha

        return self.initial_decay_lr * decayed

    def warmup_learning_rate(self, step):
        completed_fraction = step / self.warmup_steps
        total_delta = self.warmup_target - self.initial_learning_rate
        return completed_fraction * total_delta

    def get_config(self):
        return {
            "initial_learning_rate": self.initial_learning_rate,
            "decay_steps": self.decay_steps,
            "alpha": self.alpha,
            "name": self.name,
            "warmup_target": self.warmup_target,
            "warmup_steps": self.warmup_steps,
        }
