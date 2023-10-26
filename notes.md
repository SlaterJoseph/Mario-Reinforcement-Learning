## General Environment Info

Complex Movement - 12 total
- [['NOOP'] -- No Movement
- ['right']
- ['right', 'A']
- ['right', 'B']
- ['right', 'A', 'B'],
- ['A'],
- ['left'],
- ['left', 'A'],
- ['left', 'B'],
- ['left', 'A', 'B'],
- ['down']
- ['up']]

Action Space - Discrete -- 12

Shape - (240, 256, 3)

## Environment's Step Info 


| Key	     | Type	 | Description                                         |
|----------|-------|-----------------------------------------------------|
| coins	   | int	  | The number of collected coins                       |
| flag_get | bool  | True if Mario reached a flag or ax                  |
| life     | int	  | The number of lives left, i.e., {3, 2, 1}           |
| score	   | int	  | The cumulative in-game score                        |
| stage	   | int	  | The current stage, i.e., {1, ..., 4}                |
| status   | str	  | Mario's status, i.e., {'small', 'tall', 'fireball'} |
| time     | int	  | The time left on the clock                          |
| world	   | int	  | The current world, i.e., {1, ..., 8}                |
| x_pos	   | int	  | Mario's x position in the stage (from the left)     |
| y_pos	   | int	  | Mario's y position in the stage (from the bottom)   |


## Environment's Reward Function

The reward function assumes the objective of the game is to move as far right as possible (increase the agent's x value), as fast as possible, without dying. To model this game, three separate variables compose the reward:

v: the difference in agent x values between states
in this case this is instantaneous velocity for the given step
v = x1 - x0
x0 is the x position before the step
x1 is the x position after the step
moving right ⇔ v > 0
moving left ⇔ v < 0
not moving ⇔ v = 0
c: the difference in the game clock between frames
the penalty prevents the agent from standing still
c = c0 - c1
c0 is the clock reading before the step
c1 is the clock reading after the step
no clock tick ⇔ c = 0
clock tick ⇔ c < 0
d: a death penalty that penalizes the agent for dying in a state
this penalty encourages the agent to avoid death
alive ⇔ d = 0
dead ⇔ d = -15
r = v + c + d

The reward is clipped into the range (-15, 15).