import numpy as np
import tensorflow as tf
from Components.agent import Agent
from keras.optimizers import Adam


def preprocess_states(state: np.array) -> np.array:
    """
    Adds the batch dimension for the input of the Agent
    :param state: The state needing changing
    :return: The state with the batch dimension added
    """
    return np.expand_dims(state, 0)


def sync_weights(q_model: Agent, target_model: Agent) -> None:
    """
    Updating the target model to the q model's weights
    :param q_model: The Q model
    :param target_model: The Target Model
    :return: None
    """
    weights = q_model.get_weights()
    target_model.set_weights(weights)


def train(q_model: Agent, target_model: Agent, batch: list[tuple], DISCOUNT: float) -> float:
    optimizer = Adam(lr=0.0001)

    states = [item[0] for item in batch]
    observations = [item[1] for item in batch]
    rewards = [item[2] for item in batch]
    actions = [item[3] for item in batch]
    done = [item[4] for item in batch]

    # Convert to NumPy arrays
    states = np.array(states)
    observations = np.array(observations)
    actions = np.array(actions)
    rewards = np.array(rewards)
    done = np.array(done)

    # Squeeze the dimensions with size 1
    states = np.squeeze(states, axis=1)
    observations = np.squeeze(observations, axis=1)
    actions = np.squeeze(actions, axis=1)

    # Convert to TensorFlow tensors
    states = tf.convert_to_tensor(states, dtype=tf.float32)
    observations = tf.convert_to_tensor(observations, dtype=tf.float32)
    rewards = tf.convert_to_tensor(rewards, dtype=tf.float32)
    actions = tf.convert_to_tensor(actions, dtype=tf.int32)  # Assuming 'a' represents actions

    # Calculate Q-values for the current and next states
    q_values = q_model(states)
    q_prime_values = target_model(observations)

    # Find the action that maximizes Q-value for the next state
    a_max = tf.argmax(q_prime_values, axis=1, output_type=tf.int32)

    # Gather Q-values for the selected actions
    q_value = tf.gather(q_values, actions, axis=1)

    # Calculate the target Q-value based on the Bellman equation
    y = rewards + DISCOUNT * tf.gather(q_prime_values, a_max, axis=1) * done

    # Compute the loss using the Huber loss
    loss = tf.losses.huber(y, q_value)

    # Perform a gradient descent step
    gradients = tf.gradients(loss, q_model.trainable_variables)
    optimizer.apply_gradients(zip(gradients, q_model.trainable_variables))

    return loss
