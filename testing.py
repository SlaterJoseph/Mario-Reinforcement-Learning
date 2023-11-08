import keras.models
import numpy as np
import tensorflow as tf

# Component Import
from Components.frame_stack import FrameStack

# Environment Import
import gym_super_mario_bros
from nes_py.wrappers import JoypadSpace
from gym_super_mario_bros.actions import COMPLEX_MOVEMENT

# Initialize the environment
env = gym_super_mario_bros.make('SuperMarioBros-v0', )
env = JoypadSpace(env, COMPLEX_MOVEMENT)
env.reset()
print('Initialized Environment')

# Create some key variables
actions = len(COMPLEX_MOVEMENT)
shape = env.observation_space.shape
frame_stack_size = 4
input_shape = (frame_stack_size,) + shape

# Paths of weights to test
q_path = "model/t-Network/V3/checkpoint"
q_model = keras.models.load_model(q_path)
stack = FrameStack(frame_stack_size)
print('Loaded Model')

state = env.reset()
done = False
episode_total_reward = 0

# Add the initial frame to the stack
stack.add_frame(state)

print('Starting Episode')
while not done:
    # Normalize the pixels
    stack_state = stack.get_stacked_state()
    get_response = np.expand_dims(stack_state, axis=0)
    action = q_model.predict(tf.convert_to_tensor(stack_state))

    i = 0
    while not done and i < 2:  # Implement frame skipping
        new_state, reward, done, info = env.step(action)
        i += 1

        # Adding important info from skipped frame
        stack.add_frame(new_state)
        state = new_state

    env.render()

