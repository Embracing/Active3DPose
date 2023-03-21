import time

import numpy as np

from .base import WrapperBase
from .utils import bind


def reset_human_overwritten(self):
    """
    Reset Humans
    Overwritter
    """
    if not hasattr(self, 'controller'):
        # don not create again
        # if-clause assumes number of humans does not change.
        # Otherwise, remove this if-caluse
        self.controller = self.controller_class(
            self.env_interaction,
            self.human_config,
            self.env_config,
            action_setting='walk',
        )

        # waiting for humans to be ready
        print('Waiting for humans to be ready (1s)')
        time.sleep(1)

    self.controller.set_human_default_loc_rot_all()
    self.controller.reset_trajectory_samplers_all()


class ReachTargetDone(WrapperBase):
    """
    Placed in the final.
    """

    def __init__(self, env):
        super().__init__(env)

        bind(env.unwrapped, reset_human_overwritten)
        # bind(env.unwrapped, step)

    def step(self, action):
        observation, reward, done, info = self.env.step(action)

        for human in self.env.controller.human_list:
            if hasattr(human.point_sampler, 'is_finished') and human.point_sampler.is_finished():
                done = True
                break

        return observation, reward, done, info
