import time

import gym
import numpy as np

from .base import WrapperBase


class ScaleHuman(WrapperBase):
    def __init__(self, env, scale=2, index_list=None):
        super().__init__(env)

        if index_list is None:
            # all humans are scaled
            env.human_config['scale'] = [scale for _ in range(env.num_humans)]
        else:
            assert len(index_list) <= env.num_humans
            assert max(index_list) < env.num_humans

            for idx in index_list:
                env.human_config['scale'][idx] = scale
