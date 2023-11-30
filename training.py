import os

import numpy as np
import tensorflow as tf

# Component Import
from Components.agent import Agent
from Components.replay_buffer import ReplayBuffer

# Environment Import
import gym_super_mario_bros
from nes_py.wrappers import JoypadSpace
from gym_super_mario_bros.actions import COMPLEX_MOVEMENT
from wrappers import wrap_mario
import random

# Utils imports
from utils import preprocess_states, sync_weights, train

# Plotting
import matplotlib.pyplot as plt

###################################################################################################
EPISODE_COUNT = 10_000
EXPERIENCE_STORAGE_SIZE = 50_000
BATCH_SIZE = 512

RENDER_EVERY = 100
UPDATE_EVERY = 100
PRINT_EVERY = 10
MONITOR_EVERY = 50
SAVE_EVERY = 100
TRAIN_EVERY = 25
ACTION_EVERY = 4

TEMPERATURE = 0.8
MIN_EPSILON = 0.001
EPSILON_DECAY = 0.9995
DISCOUNT = 0.99

VERSION = 'V4'
WEIGHT_PATH = f'./Models/{VERSION}'
###################################################################################################

env = gym_super_mario_bros.make('SuperMarioBros-v0')
env = JoypadSpace(env, COMPLEX_MOVEMENT)
env = wrap_mario(env)

actions = len(COMPLEX_MOVEMENT)

q_model = Agent(actions, TEMPERATURE)
target_model = Agent(actions, TEMPERATURE)
buffer = ReplayBuffer(EXPERIENCE_STORAGE_SIZE, BATCH_SIZE)
epsilon = 1

score_list = list()
stage_dict = dict()
total_score = 0.0
episode_score = 0.0
loss = 0.0
training_count = 0
rendering = False

for episode in range(EPISODE_COUNT):
    state = preprocess_states(env.reset())
    done = False
    level = (0, 0)
    level_number = 0

    # Exploration vs Exploitation
    while not done:
        if random.random() > epsilon:  # Use Boltzmann Exploration
            action = q_model(tf.convert_to_tensor(state, dtype=tf.float32))[0]
        else:  # Use Epsilon-Greedy Exploration
            action = env.action_space.sample()

        observation, reward, done, info = env.step(action)
        observation = preprocess_states(observation)

        total_score += reward

        reward = np.sign(reward) * (np.sqrt(abs(reward) + 1) - 1) + 0.001 * reward
        buffer.store_experience(state, observation, float(reward), action, int(1 - done))
        state = observation

        # Some stage tracking
        level = (info['world'], info['stage'])
        level_number = (level[0] * 4) - level[1]

        # # Render for visual validation that the training is working
        if episode % RENDER_EVERY == 0:
            env.render()
            rendering = True

        # Training
        if len(buffer) > 2000 and episode % UPDATE_EVERY != 0:
            batch = buffer.sample_batch()
            loss += train(q_model, target_model, batch, DISCOUNT)
            training_count += 1

        # Update after every UPDATE_EVERY training and save the weights
        if training_count % UPDATE_EVERY == 0:
            sync_weights(q_model, target_model)

    epsilon *= EPSILON_DECAY
    if epsilon < MIN_EPSILON:
        epsilon = MIN_EPSILON

    # More level tracking
    if level_number not in stage_dict:
        stage_dict[level_number] = 0
    stage_dict[level_number] += 1
    score_list.append(episode_score)

    # Print updates of training
    if episode % PRINT_EVERY == 0:
        print(f'Epoch: {episode} | Score : {total_score / PRINT_EVERY} |'
              f' Loss : {loss / PRINT_EVERY} | Level : {level} | Epsilon : {epsilon}')

        total_score = 0
        loss = 0.0

if rendering:
    env.close()
    rendering = False

os.makedirs(WEIGHT_PATH + '/Q')
q_model.save_weights(WEIGHT_PATH + '/Q')

os.makedirs(WEIGHT_PATH + '/T')
target_model.save_weights(WEIGHT_PATH + '/T')

# Show how far the bot made it and how many times
plt.bar(list(stage_dict.keys()), list(stage_dict.values()))
plt.xlabel('Levels')
plt.ylabel('Times Reached')
plt.title('Times an Episode Ended on a Level')

plt.plot(EPISODE_COUNT, score_list)
plt.xlabel('Episode Count')
plt.ylabel('Score')
plt.title('Score over Episodes')

plt.show()
