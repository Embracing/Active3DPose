from .base import WrapperBase
from .utils import bind


def outer(num):
    def movable_cameras_override(self):
        return num

    return movable_cameras_override


class MovableCam(WrapperBase):
    """
    Can be placed in any order
    1: Stationary scene.
    """

    def __init__(self, env, num_movable_cameras):
        super().__init__(env)

        assert num_movable_cameras <= env.num_cameras

        bind(env.unwrapped, outer(num_movable_cameras))

        if num_movable_cameras == 1:
            env.unwrapped.MULTI_AGENT = False
        else:
            env.unwrapped.MULTI_AGENT = True

        env.unwrapped.load_config(
            map_name=self.config.ENV.MAP_NAME,
            env_name=env.config.ENV.ENV_NAME,
            num_humans=env.config.ENV.NUM_OF_HUMANS,
            walk_speed_range=env.config.ENV.WALK_SPEED_RANGE,
            rot_speed_range=env.config.ENV.ROTATION_SPEED_RANGE,
        )
