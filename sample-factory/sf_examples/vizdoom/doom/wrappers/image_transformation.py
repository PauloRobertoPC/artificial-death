import cv2
import gymnasium as gym
from gymnasium.spaces import Sequence
import numpy as np

class ImageTransformationWrapper(gym.ObservationWrapper):
    def __init__(self, env, resized_shape:Sequence):
        super(ImageTransformationWrapper, self).__init__(env)
        self.resized_shape = resized_shape
        shape_with_channels = (1, resized_shape[0], resized_shape[1])
        self.observation_space = gym.spaces.Box(
            low=0, high=255, shape=shape_with_channels, dtype=np.uint8)

    def observation(self, obs):
        # turn into (h, w, c)
        if obs.shape[0] == 3:
             img = np.ascontiguousarray(np.moveaxis(obs, 0, -1))
        else:
             img = np.ascontiguousarray(obs)

        # to grayscale
        gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)

        # resize
        resized = cv2.resize(gray, (self.resized_shape[1], self.resized_shape[0]), interpolation=cv2.INTER_NEAREST)
        # turn into (1, h, w)
        resized = resized[None, :, :]
        return resized
