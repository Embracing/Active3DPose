import time

from .base import WrapperBase
from .utils import bind


class ShuffleCamID(WrapperBase):
    """
    Can be placed in any order
    1: Stationary scene.
    """

    def __init__(
        self,
        env,
    ):
        super().__init__(env)

        # bind(env.unwrapped, reset_human_overwritten)

        env.unwrapped.shuffle_cam_id = True
        # env.unwrapped.random_init_mesh = random_init_mesh
