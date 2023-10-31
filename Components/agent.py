import tensorflow as tf
import numpy as np

from keras import Sequential
from keras.layers import Conv2D, Dense, Input, Flatten, Activation
from keras.optimizers import Adam


class Agent:
    """
    DQN Agent

    The agent which learns how to play the game
    It does this through taking actions until an end point is reached. Then it reevaluates the actions and
    how they affected the end reward. If the end reward was very positive, it is likely to do it again. If it was very
    it will try to avoid that action when possible.
    """

    def __init__(self, shape: tuple, actions: int, batch_size: int, discount: float):
        self.q_network = self.build_dqn_model(shape, actions)
        self.target_network = self.build_dqn_model(shape, actions)

        self.batch_size = batch_size
        self.discount = discount

    def build_dqn_model(self, shape: tuple, actions: int):
        """
        Builds the neural network which predicts q values for determining actions based on the current state

        :param shape: the shape of the input
        :param actions: the amount of actions the agent can take
        :return: A Sequential Model
        """
        network = Sequential()
        # Define the input
        network.add(Input(shape=shape))

        # Process the frames / frame stack
        network.add(Conv2D(32, (8, 8), strides=(4, 4)))
        network.add(Activation('relu'))
        network.add(Conv2D(64, (4, 4), strides=(2, 2)))
        network.add(Activation('relu'))
        network.add(Conv2D(64, (2, 2), strides=(1, 1)))
        network.add(Activation('relu'))

        # Flatten for analysis
        network.add(Flatten())

        # Analyze the frames
        network.add(Dense(512, activation='relu', kernel_initializer='he_uniform'))
        # network.add(Dense(64, activation='relu', kernel_initializer='he_uniform'))

        # Decide the probabilities of each action
        network.add(Dense(actions, activation='linear', kernel_initializer='he_uniform'))  # End Layer
        network.compile(optimizer=Adam(learning_rate=0.001), loss='mse')

        return network

    def policy(self, state):
        """
        Look at a given state and return the action deemed the best
        based on the given Q values

        :param state: current state of the game
        :return: abn action
        """
        state_input = tf.convert_to_tensor(state, dtype=tf.float32)
        action_q = self.target_network(state_input)
        action = np.argmax(action_q.numpy()[0], axis=0)
        return action

    def update_target_network(self):
        """
        Updates the current target network with the q network

        :return: None
        """
        self.target_network.set_weights(self.q_network.get_weights())

    def train(self, batch: tuple):
        """
        Trains the network using gameplay experiences to help predict better
        outcomes

        :param batch: a batch of gameplay experiences
        :return: training loss
        """
        current_states = np.array(batch[0])
        current_qs_list = self.q_network.predict(current_states)

        new_current_states = np.array(batch[1])
        future_qs_list = self.q_network.predict(new_current_states)

        X = list()
        y = list()

        for i in range(len(batch)):
            current_state = batch[0][i]
            new_current_state = batch[1][i]
            reward = batch[2][i]
            action = batch[3][i]
            done = batch[4][i]

            if not done:
                max_future_q = np.max(future_qs_list[i])
                new_q = reward + self.discount * max_future_q
            else:
                new_q = reward

            current_qs = current_qs_list[i]
            current_qs[action] = new_q

            X.append(current_state)
            y.append(current_qs)

        history = self.q_network.fit(x=np.array(X), y=np.array(y), batch_size=len(batch))
        loss = history.history['loss']
        return loss

    def save_weights(self, path: str, episode: int) -> None:
        """
        Save the weights of the target and q networks
        :param path: Path to save the weights
        :return: None
        """

        self.q_network.save_weights(f'{path}-q-{episode}')
        self.target_network.save_weights(f'{path}-t-{episode}')
