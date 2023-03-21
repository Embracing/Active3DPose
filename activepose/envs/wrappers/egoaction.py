import gym
import numpy as np

from .base import WrapperBase


class EgoAction(WrapperBase):
    def __init__(self, env):
        super().__init__(env)
        env.unwrapped.ego_action = True
