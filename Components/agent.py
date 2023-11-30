import tensorflow as tf
from keras.layers import Conv2D, Dense, Flatten
from keras.initializers import glorot_uniform, Constant
from keras import Sequential
import numpy as np


class Agent(Sequential):
    """
    Agent which play mario and learns
    """

    def __init__(self, actions: int, temperature: float):
        super(Agent, self).__init__()
        self.conv1 = Conv2D(32, (8, 8), strides=(4, 4), activation='relu')
        self.conv2 = Conv2D(64, (3, 3), strides=(1, 1))
        # self.conv2 = Conv2D(64, (4, 4), strides=(2, 2), activation='relu')
        # self.conv3 = Conv2D(64, (2, 2), strides=(1, 1), activation='relu')

        self.flatten = Flatten()

        self.process = Dense(512, activation='relu')
        self.q = Dense(actions, activation='linear')
        self.v = Dense(1, activation='linear')

        self.weights()

        self.temperature = temperature

    def call(self, input: np.array) -> np.array:
        """
        The process of taking the input data and deciding on a Q value for the appropriate action
        :param input: The current observation
        :return: A np.array of the values of each action
        """
        input = self.conv1(input)
        input = self.conv2(input)
        # input = self.conv3(input)
        input = self.flatten(input)
        input = self.process(input)

        adv = self.q(input)
        v = self.v(input)

        prob = v + (adv - 1 / (adv.shape[-1] * tf.reduce_max(adv, axis=-1, keepdims=True)))
        softmax_probs = tf.nn.softmax(prob / self.temperature)
        action = tf.squeeze(tf.random.categorical(softmax_probs, 1), axis=-1)
        return action.numpy()

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

    def get_weights(self) -> list:
        """
        Get the weights of the current layers
        :return: A list of layer weights
        """
        weights = [layer.get_weights() for layer in self.layers]
        return weights

    def set_weights(self, weights: list) -> None:
        """
        Update the current model using the provided weights
        :param weights: A list of weights
        :return: None
        """
        for layer, layer_weights in zip(self.layers, weights):
            layer.set_weights(layer_weights)
