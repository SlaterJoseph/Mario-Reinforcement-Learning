from collections import deque

import gym_super_mario_bros
from nes_py.wrappers import JoypadSpace
from gym_super_mario_bros.actions import COMPLEX_MOVEMENT

import tensorflow as tf
import numpy as np

from keras import Sequential
from keras.layers import InputLayer, Dense
from keras.regularizers import l2
from keras.optimizers import Adam

from agent import Agent

##############################################################################################################
REPLAY_MEMORY_SIZE = 10_000
MIN_REPLAY_MEMORY_SIZE = 1_000
MODEL_NAME = 'V1'
MINIBATCH_SIZE = 64
DISCOUNT = 1
UPDATE_TARGET_EVERY = 200
RENDER_EVERY = 1000

EPISODE_COUNT = 10000

CHECKPOINT_DIR = './train/'
LOG_DIR = './logs/.'

############################################################################################################

env = gym_super_mario_bros.make('SuperMarioBros-v0')
env = JoypadSpace(env, COMPLEX_MOVEMENT)

num_frames = 10
actions = len(COMPLEX_MOVEMENT)
shape = env.action_space.shape

agent = Agent(num_frames=num_frames, actions=actions, shape=shape)

agent.agent.fit(env, nb_steps=100000, visualize=True)
agent.agent.save_weights(CHECKPOINT_DIR + MODEL_NAME)


# for episode in range(EPISODE_COUNT):
#     state = env.reset()
#     done = False
#
#     frame_stack = agent.stack_frames(state, True)
#
#     while not done:
#         action = agent.agent.forward(state)
#         new_state, reward, done, info = env.step(action)
#         agent.agent.backward(reward, done)
#
#         if len(agent.agent.memory) > agent.agent.batch_size:
#             agent.agent.training()
#
#     if done:
#         state = env.reset()
#     state, reward, done, info = env.step(env.action_space.sample())
#     print(state)
#     env.render()
#
# env.close()





