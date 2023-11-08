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

###################################################################################################
EPISODE_COUNT = 10000
EXPERIENCE_STORAGE_SIZE = 50000
BATCH_SIZE = 256

RENDER_EVERY = 50
UPDATE_EVERY = 50
PRINT_EVERY = 10
MONITOR_EVERY = 50
SAVE_EVERY = 100
TRAIN_EVERY = 25
ACTION_EVERY = 4

MIN_EPSILON = 0.01
EPSILON_DECAY = 0.9975
DISCOUNT = 0.99

VERSION = 'V4'
WEIGHT_PATH = f'/Models/{VERSION}/'
###################################################################################################

env = gym_super_mario_bros.make('SuperMarioBros-v0')
env = JoypadSpace(env, COMPLEX_MOVEMENT)
env = wrap_mario(env)

actions = len(COMPLEX_MOVEMENT)

q_model = Agent(actions)
target_model = Agent(actions)
buffer = ReplayBuffer(EXPERIENCE_STORAGE_SIZE, BATCH_SIZE)
epsilon = 1.00

score_list = list()
stage_list = list()
total_score = 0.0
loss = 0.0
training_count = 0

for episode in range(EPISODE_COUNT):
    state = preprocess_states(env.reset())
    done = False
    level = (0, 0)

    # Exploration vs Exploitation
    while not done:
        if random.random() > epsilon:
            action = np.argmax(q_model(tf.convert_to_tensor(state, dtype=tf.float32)).numpy())
        else:
            action = env.action_space.sample()

        observation, reward, done, info = env.step(action)
        observation = preprocess_states(observation)

        total_score += reward
        epsilon *= EPSILON_DECAY

        reward = np.sign(reward) * (np.sqrt(abs(reward) + 1) - 1) + 0.001 * reward
        buffer.store_experience(state, observation, float(reward), action, int(1 - done))

        state = observation

        level = (info['world'], info['stage'])
        stage_list.append((level[0] * 4) - level[1])

        # Render for visual validation that the training is working
        if episode % RENDER_EVERY == 0:
            env.render()

        # Training
        if len(buffer) > 2000 and episode != 0:
            batch = buffer.sample_batch()
            loss += train(q_model, target_model, batch, DISCOUNT)
            training_count += 1

        # Update after every UPDATE_EVERY training and save the weights
        if training_count % UPDATE_EVERY == 0 and training_count != 0:
            sync_weights(q_model, target_model)
            q_model.save_weights(WEIGHT_PATH + f'q_{episode}')
            target_model.save_weights(WEIGHT_PATH + f'target_{episode}')

    # Print updates of training
    if episode & PRINT_EVERY == 0:
        print(f'Epoch: {episode} | Score : {total_score / PRINT_EVERY} |'
              f' Loss : {loss / PRINT_EVERY} | Level : {level}')

        score_list.append(total_score / PRINT_EVERY)
        total_score = 0
        loss = 0.0

