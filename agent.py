import numpy as np
from keras.models import Sequential
from keras.layers import Dense, Flatten, Conv2D, Input
from keras.optimizers import Adam

from rl.agents import DQNAgent
from rl.policy import EpsGreedyQPolicy
from rl.memory import SequentialMemory


class Agent:
    def __init__(self, num_frames: int, actions: int, shape: tuple):
        self.num_frames = num_frames
        self.actions = actions
        self.shape = shape
        self.frame_stack = np.zeros((num_frames, *shape), dtype=np.float32)

        self.model = self.build_model()
        self.agent = self.build_agent()
        self.memory = self.build_memory()

    def build_model(self):
        """Create Model"""
        model = Sequential()
        model.add(Input(shape=self.shape))
        model.add(Conv2D(32, (8, 8), strides=(4, 4)))
        model.add(Conv2D(64, (4, 4), strides=(2, 2)))
        model.add(Conv2D(32, (2, 2), strides=(1, 1)))
        model.add(Flatten())
        model.add(Dense(64, activation='relu'))
        model.add(Dense(64, activation='relu'))
        model.add(Dense(self.actions, activation='linear'))
        return model

    def stack_frames(self, obs, is_new: bool):
        """Add Frames to Frame Stack"""
        if is_new:
            self.frame_stack = np.zeros_like(self.frame_stack)

        self.frame_stack[0: self.num_frames - 1] = self.frame_stack[1:]
        self.frame_stack[self.num_frames - 1] = obs

        return self.frame_stack

    def build_agent(self):
        memory = self.memory
        policy = EpsGreedyQPolicy(eps=0.1)
        dqn = DQNAgent(model=self.model, memory=memory, policy=policy, nb_actions=self.actions,
                       enable_double_dqn=True, target_model_update=1e-2)

        dqn.compile(Adam(lr=1e-3), metrics=['mae'])
        return dqn

    def build_memory(self):
        return SequentialMemory(limit=10000, window_length=self.num_frames)
