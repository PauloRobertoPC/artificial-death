import cv2
import gymnasium as gym
from gymnasium.spaces import Sequence
import numpy as np

class ImageTransformationWrapper(gym.ObservationWrapper):
    def __init__(self, env, resized_shape:Sequence):
        super(ImageTransformationWrapper, self).__init__(env)
        self.resized_shape = resized_shape
        old_shape = self.observation_space.shape
        shape_with_channels = (1, resized_shape[0], resized_shape[1])
        self.observation_space = gym.spaces.Box(
            low=0, high=255, shape=shape_with_channels, dtype=np.uint8)

    def observation(self, observation):
        return self.grayscale(observation)
        
    def grayscale(self, observation):

        # print(observation.shape)
        # print(observation.dtype)
        # (B, 3, H, W) → (B, H, W, 3)
        observation = np.moveaxis(observation, 1, -1)

        # Luminance formula (OpenCV-compatible BGR)
        gray = np.rint(
            0.114 * observation[..., 0] +
            0.587 * observation[..., 1] +
            0.299 * observation[..., 2]
        ).astype(np.uint8)

        # (B, H, W) → (B, 1, H, W)
        gray = gray[:, None, :, :]
        # gray = np.repeat(gray, 3, axis=1)
        return gray

def image_wrapper(resize_shape):
    return lambda env: ImageTransformationWrapper(env, resize_shape)