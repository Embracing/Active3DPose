from functools import wraps

import gym
import numpy as np

from .base import WrapperBase


class Aux_Rewards(WrapperBase):
    """
    Can be placed in any order
    """

    def __init__(self, env, **kwargs):
        super().__init__(env)

        if kwargs.get('centering', False):
            env.unwrapped.use_reward_centering = True

        if kwargs.get('distance', False):
            env.unwrapped.use_reward_distance = True

        if kwargs.get('obstruction', False):
            env.unwrapped.use_reward_obstruction = True

        if kwargs.get('iot', False):
            env.unwrapped.use_reward_iot = True

        if kwargs.get('anti_collision', False):
            env.unwrapped.use_reward_anti_collision = True

        # if cameras exceed the boundary
        if kwargs.get('camera_state', False):
            env.unwrapped.use_reward_camera_state = True

        # original indv
        if kwargs.get('covered_by_two', False):
            env.unwrapped.use_reward_covered_by_two = True
