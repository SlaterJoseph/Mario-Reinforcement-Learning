import numpy as np

from collections import deque
import random


class ReplayBuffer:
    """
    Replay Buffer

    The memory of the DQN - Stores and retrieves results
    """

    def __init__(self, experience_storage_size: int, batch_size: int):
        self.experiences = deque(maxlen=experience_storage_size)
        self.batch_size = batch_size

    def store_experience(self, state: np.array, result: np.array, reward: float, action: int, done: int) -> None:
        """
        Stores a single transaction of gameplay
        :param state: current game state
        :param result: the result from the action taken to the game state
        :param reward: the reward of the action from teh game state
        :param action: the action taken to the game state
        :param done: a indication if the current episode is complete
        :return: None
        """
        # Store the recent state and all relevant info
        self.experiences.append((state, result, reward, action, done))

    def sample_batch(self) -> list[tuple]:
        """
        Samples a batch of experiences for training
        :return: a list of experiences
        """
        # Get a random sample if we have more than the length, otherwise take all the memory
        batch = random.sample(self.experiences, k=self.batch_size) if \
            len(self.experiences) > self.batch_size else self.experiences

        state = list()
        new_state = list()
        action = list()
        reward = list()
        done = list()

        # Add the samples into the lists for each category
        for sample in batch:
            state.append(sample[0])
            new_state.append(sample[1])
            reward.append(sample[2])
            action.append(sample[3])
            done.append(sample[4])

        print(np.array(state).shape, np.array(new_state).shape, np.array(reward).shape, np.array(action).shape,
              np.array(done).shape)
        print(len(state), len(new_state), len(reward), len(action), len(done))

        return np.array(state), np.array(new_state), np.array(reward), np.array(action), np.array(done)

    def __len__(self):
        return len(self.experiences)
