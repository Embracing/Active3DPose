from functools import wraps

import gym
import numpy as np

from .base import WrapperBase


class AirWallOuter(WrapperBase):
    """
    Can be placed in any order
    """

    def __init__(self, env):
        super().__init__(env)
        env.unwrapped.use_airwall_outer = True


class AirWallInner(WrapperBase):
    """
    Can be placed in any order
    """

    def __init__(self, env, lower_bound, higher_bound):
        super().__init__(env)
        env.unwrapped.use_airwall_inner = True
        env.unwrapped.lower_bound_inner = lower_bound
        env.unwrapped.higher_bound_inner = higher_bound
