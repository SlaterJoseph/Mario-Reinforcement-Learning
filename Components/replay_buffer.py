class ReplayBuffer:
    """
    Replay Buffer

    The memory of the DQN - Stores and retrieves results
    """
    def store_experience(self, state, result, reward, action, done):
        """
        Stores a single transaction of gameplay
        :param state: current game state
        :param result: the result from the action taken to the game state
        :param reward: the reward of the action from teh game state
        :param action: the action taken to the game state
        :param done: a indication if the current episode is complete
        :return: None
        """

        return

    def sample_batch(self):
        """
        Samples a batch of experiences for training
        :return:
        """
        return