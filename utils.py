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


def train(q_model: Agent, target_model: Agent, batch: tuple, discount: float) -> float:
    optimizer = Adam(learning_rate=0.0001)

    # Assigns variables to each list
    states = batch[0]
    observations = batch[1]
    rewards = batch[2]
    actions = batch[3]
    done = batch[4]

    # Squeeze the dimensions with size 1
    # states = np.squeeze(states, axis=1)
    # observations = np.squeeze(observations, axis=1)
    # actions = np.squeeze(actions, axis=1)

    # Convert to tensors
    states = tf.convert_to_tensor(states, dtype=tf.float32)
    observations = tf.convert_to_tensor(observations, dtype=tf.float32)
    rewards = tf.convert_to_tensor(rewards, dtype=tf.float32)
    actions = tf.convert_to_tensor(actions, dtype=tf.int32)  # Assuming 'a' represents actions

    with tf.GradientTape() as tape:
        # Calculate Q-values for the current and next states
        q_values = q_model(states)
        q_prime_values = target_model(observations)
        q_prime_values = tf.cast(q_prime_values, dtype=tf.float32)

        # Find the action that maximizes Q-value for the next state
        a_max = tf.argmax(q_prime_values, output_type=tf.int32)

        # Gather Q-values for the selected actions
        q_value = tf.gather(q_values, actions)

        # Calculate the target Q-value based on the Bellman equation
        y = rewards + discount * tf.gather(q_prime_values, a_max) * done

        # Compute the loss using the HuberBut wh loss
        loss = tf.reduce_mean(tf.losses.huber(y, q_value))
        # Compute gradients and update the model

    gradients = tape.gradient(loss, q_model.trainable_variables)
    optimizer.apply_gradients(zip(gradients, q_model.trainable_variables))
    return loss
