import numpy as np

from collections import deque


class FrameStack:
    def __init__(self, frame_num: int = 4):
        """
        :param frame_num: The number of frames stored
        """

        self.frame_stack = deque(maxlen=frame_num)
        self.frame_num = frame_num

    def add_frame(self, frame) -> None:
        """
        Add a current frame to the framestack, pushing out the oldest one
        :param frame: The new frame to be added
        :return: None
        """
        self.frame_stack.append(frame)

    def initialize(self, shape: tuple[int]) -> None:
        """
        Reset the framestack to a completely empty state primed for new data
        :param shape: The shape of the environment
        :return: None
        """
        zero_frame = np.zeros(shape, dtype=np.uint8)

        for i in range(self.frame_num):
            self.frame_stack.append(zero_frame)

    def get_stack(self) -> deque:
        """
        A getter for the deque frame stack

        :return: deque
        """

        return self.frame_stack

    def get_stacked_state(self) -> np.array:
        """
        Turns the frame stack into 1 single frame/state

        :return: All frames in the stack compiled into 1
        """

        return np.stack(self.frame_stack, axis=0)
