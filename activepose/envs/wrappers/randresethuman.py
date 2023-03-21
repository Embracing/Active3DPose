import time

from .base import WrapperBase
from .utils import bind


class RandResetHuman(WrapperBase):
    """
    Can be placed in any order
    1: Stationary scene.
    """

    def __init__(self, env, random_init_loc_rot=True, **kwargs):
        super().__init__(env)

        # bind(env.unwrapped, reset_human_overwritten)

        env.unwrapped.random_init_loc_rot = random_init_loc_rot
        # env.unwrapped.random_init_mesh = random_init_mesh
