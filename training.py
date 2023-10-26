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

###################################################################################################
EPISODE_COUNT = 10000

###################################################################################################

agent = Agent()
buffer = ReplayBuffer()
stack = FrameStack(6)

env = gym_super_mario_bros.make('SuperMarioBros-v0')
env = JoypadSpace(env, COMPLEX_MOVEMENT)

for episode in range(EPISODE_COUNT):
    experience_batch = buffer.sample_batch()
    loss = agent.train(experience_batch)


def collect_experience(env: Env, agent: Agent, buffer: ReplayBuffer, stack: FrameStack):
    """
    A function to collect gameplay data

    :param env: the environment we are using to simulate the game
    :param agent: the agent being trained
    :param buffer: the replay buffer for storing states
    :param stack: the current frame stack used in training
    :return: None
    """

    stack.initialize(env.observation_space.shape)
    state = env.reset()
    done = False
    last_info = None

    while not done:
        action = agent.policy(state)
        new_state, reward, done, info = env.step(action)

        if last_info is not None:
            reward = calculate_reward(info, last_info, reward)

        buffer.store_experience(state, new_state, reward, action, done)

        # Updating info for next step
        last_info = info
        state = new_state


def calculate_reward(info: dict, last_info: dict, reward: float):
    """
    Calculates the reward of a given state using the given reward along
    with a few additional parameters

    :param info: The information from this current state
    :param last_info: The information from the previous state
    :param reward: The reward from the gym's reward function -- Check notes
    :return: float
    """

    old_score, new_score = last_info['score'], info['score']
    score_delta = new_score - old_score
    score_reward = score_delta / 10000  # So a 10 score change is only worth 0.001

    reward += score_reward

    return reward


def frame_stack_to_state(frame_stack: FrameStack, state: np.array):
    """
    A function which adds the new frame and converts the frame stack into a np array

    :param frame_stack: The current frame stack
    :param state: The new state
    :return: np array
    """

    frame_stack.add_frame(state)
    state = np.stack(frame_stack.get_stack(), axis=-1)
    return state
