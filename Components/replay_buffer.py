import numpy as np
import random


class ReplayBuffer:
    """
    Replay Buffer

    The memory of the DQN - Stores and retrieves results
    """

    def __init__(self, experience_storage_size: int, batch_size: int):
        self.experiences = CustomDeque(experience_storage_size)
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
        self.experiences.add((state, result, reward, action, done))

    def sample_batch(self) -> tuple:
        """
        Samples a batch of experiences for training
        :return: a list of experiences
        """
        # Get a random sample if we have more than the length, otherwise take all the memory

        batch = list()
        i = 0
        while i < self.batch_size:
            batch.append(self.experiences.collect())
            i += 1

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

        return np.array(state), np.array(new_state), np.array(reward), np.array(action), np.array(done)

    def __len__(self):
        return len(self.experiences)


class CustomDeque:
    """
    Custom Deque in an attempt to save time
    """
    def __init__(self, max_len):
        self.deque = list()
        self.max_len = max_len

    def add(self, value) -> None:
        """
        Add a new element to the Deque
        :param value: The value to be added
        :return: None
        """
        if len(self.deque) == self.max_len:
            del self.deque[0]
        self.deque.append(value)

    def collect(self) -> any:
        """
        Collecting a single experience from the deque
        :return: The collected experience
        """
        index = int(random.randint(0, len(self.deque) - 1))
        return self.deque.pop(index)

    def __len__(self):
        return len(self.deque)
