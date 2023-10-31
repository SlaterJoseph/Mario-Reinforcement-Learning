import numpy as np
from gym import Env

# Component Import
from Components.agent import Agent
from Components.replay_buffer import ReplayBuffer
from Components.frame_stack import FrameStack

# Environment Import
import gym_super_mario_bros
from nes_py.wrappers import JoypadSpace
from gym_super_mario_bros.actions import COMPLEX_MOVEMENT
import random

# Data Tracking
import matplotlib.pyplot as plt

###################################################################################################
EPISODE_COUNT = 1000
EXPERIENCE_STORAGE_SIZE = 10000
BATCH_SIZE = 1000

RENDER_EVERY = 10
UPDATE_EVERY = 50
MONITOR_EVERY = 50
SAVE_EVERY = 100
TRAIN_EVERY = 100

MIN_EPSILON = 0.01
EPSILON_DECAY = 0.995
DISCOUNT = 0.95

MODEL_PATH = 'models'
VERSION = 'V1'


########################################################################################################################

def evaluate_performance(env: Env, agent: Agent, stack: FrameStack, episode_amount: int) -> float:
    """
    Evaluate the performance of the current model

    :param env: The game environment
    :param agent: The Agent
    :param episode_amount: The amount of episodes to test over
    :return: Average reward across n episodes
    """

    total_reward = 0.0

    for i in range(episode_amount):
        stack.initialize(shape)
        state = env.reset()
        done = False
        episode_reward = 0.0
        last_info = None

        while not done:
            stack.add_frame(state)

            # Normalize the pixels
            stack_state = stack.get_stacked_state()

            # Add 1 dimension as the batch size is 1
            get_response = np.expand_dims(stack_state, axis=0)
            action = agent.policy(get_response)
            new_state, reward, done, info = env.step(action)

            if last_info is None:
                episode_reward += reward
            else:
                episode_reward += calculate_reward(info, last_info, reward)

            state = new_state
            last_info = info

        total_reward += episode_reward

    return total_reward / episode_amount


########################################################################################################################


def collect_experience(env: Env, agent: Agent, buffer: ReplayBuffer, stack: FrameStack, episode: int,
                       epsilon: float) -> float:
    """
    A function to collect gameplay data

    :param env: the environment we are using to simulate the game
    :param agent: the agent being trained
    :param buffer: the replay buffer for storing states
    :param stack: the current frame stack used in training
    :param episode: the current episode number
    :return: None
    """

    stack.initialize(shape)
    state = env.reset()
    done = False
    last_info = None
    episode_total_reward = 0

    # Add the initial frame to the stack
    stack.add_frame(state)

    while not done:
        # Normalize the pixels
        stack_state = stack.get_stacked_state()

        # Exploration vs Exploitation
        if random.random() > epsilon:
            # Add 1 dimension as the batch size is 1
            get_response = np.expand_dims(stack_state, axis=0)
            action = agent.policy(get_response)
        else:
            action = random.randint(0, 11)

        new_state, reward, done, info = env.step(action)

        # Render every so often to monitor progress
        if episode % RENDER_EVERY == 0 and episode != 0:
            env.render()

        if last_info is not None:  # To ignore the 1st step of the episode
            reward = calculate_reward(info, last_info, reward)

        episode_total_reward += reward

        # Add the new state to memory
        stack.add_frame(new_state)
        new_stack_state = stack.get_stacked_state()
        buffer.store_experience(stack_state, new_stack_state, reward, action, done)

        # Updating info for next step
        last_info = info
        state = new_state

        # If the end of a level is reached or a life is lost, we clear the frame stack
        if info['flag_get'] or last_info['life'] > info['life']:
            stack.initialize(shape)

    return episode_total_reward


########################################################################################################################

def calculate_reward(info: dict, last_info: dict, reward: float) -> float:
    """
    Calculates the reward of a given state using the given reward along
    with a few additional parameters

    :param info: The information from this current state
    :param last_info: The information from the previous state
    :param reward: The reward from the gym's reward function -- Check notes
    :return: float
    """

    # Adding reward based on score so agent tries killing enemies and getting power ups
    old_score, new_score = last_info['score'], info['score']
    score_delta = new_score - old_score
    score_reward = score_delta / 10000  # So a 10 score change is only worth 0.001

    reward += score_reward

    return reward


########################################################################################################################
# Initialize the environment
env = gym_super_mario_bros.make('SuperMarioBros-v0', )
# env = AtariPreprocessing(env=env, frame_skip=4)
env = JoypadSpace(env, COMPLEX_MOVEMENT)

# Create some key variables
actions = len(COMPLEX_MOVEMENT)
shape = env.observation_space.shape
frame_stack_size = 4
input_shape = (frame_stack_size,) + shape

# Initialize the different components
agent = Agent(input_shape, actions, BATCH_SIZE, DISCOUNT)
buffer = ReplayBuffer(EXPERIENCE_STORAGE_SIZE, BATCH_SIZE)
stack = FrameStack(frame_stack_size)

epsilon = 1.0

loss_list = list()

# Training over n episodes
for episode in range(EPISODE_COUNT):
    collect_experience(env, agent, buffer, stack, episode, epsilon)
    experience_batch = buffer.sample_batch()

    if episode % TRAIN_EVERY == 0 and episode != 0:
        loss = agent.train(experience_batch)
        loss_list.append(loss)

    # if episode % MONITOR_EVERY == 0 and episode != 0:
    #     avg_reward = evaluate_performance(env, agent, stack, 10)
    #     print(f'Episode {episode}/{EPISODE_COUNT} -- Avg Reward is {avg_reward}, Loss is {loss}')
    # else:
    print(f'Episode {episode + 1}/{EPISODE_COUNT} -- Epsilon is {epsilon}')

    epsilon *= EPSILON_DECAY

    # Epsilon should never reach 0
    if epsilon < MIN_EPSILON:
        epsilon = MIN_EPSILON

    if episode % UPDATE_EVERY == 0 and episode != 0:
        agent.update_target_network()

    if episode % SAVE_EVERY == 0 and episode != 0:
        agent.save_weights(f'{MODEL_PATH}/{VERSION}', episode)

env.close()

plt.figure(figsize=(10, 6))
plt.plot([x for x in range(1, 1000)], loss_list, color='blue')
plt.title('Loss Over Episodes')
plt.xlabel('Episode')
plt.ylabel('Loss')
plt.grid(True)

plt.show()

# Implement 4 frame skips -- 4 frames are 1 input,
# so there is no need ot input 4 frames individually, input 4 frames as 1 chunk then the next 4 frames

# Implement grayscale conversion (Option)
# Change to Simple moveset (Option)
