import tensorflow as tf
import keras
from keras.layers import Conv2D, Dense, Flatten
from keras.initializers import glorot_uniform, Constant

import numpy as np


class Agent(keras.Model):
    """
    Agent which play mario and learns
    """
    def __init__(self, actions):
        super(Agent, self).__init__()
        self.conv1 = Conv2D(32, (8, 8), strides=(4, 4), activation='relu')
        self.conv2 = Conv2D(64, (4, 4), strides=(2, 2), activation='relu')
        self.conv3 = Conv2D(64, (2, 2), strides=(1, 1), activation='relu')

        self.flatten = Flatten()

        self.process = Dense(512, activation='relu')
        self.q = Dense(actions, activation='linear')
        self.v = Dense(1, activation='linear')

        self.weights()

    def call(self, input: np.array) -> float:
        """
        The process of taking the input data and deciding on a Q value for the appropriate action
        :param input: The current observation
        :return:
        """
        input = self.conv1(input)
        input = self.conv2(input)
        input = self.conv3(input)
        input = self.flatten(input)
        input = self.process(input)

        adv = self.q(input)
        v = self.v(input)

        return tf.cast(v, dtype=tf.float32) + (tf.cast(adv, dtype=tf.float32) - 1 / tf.cast(tf.shape(adv)[-1], dtype=tf.float32) * tf.reduce_max(adv, axis=-1, keepdims=True))

    def weights(self) -> None:
        """
        Set preliminary weights for the Conv2D layers
        :return: None
        """
        for layer in self.layers:
            if isinstance(layer, Conv2D):
                kernel_initializer = glorot_uniform()
                bias_initializer = Constant(0.01)

                layer.kernel_initializer = kernel_initializer
                layer.bias_initializer = bias_initializer

