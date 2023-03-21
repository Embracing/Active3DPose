import numpy as np

from .base import WrapperBase
from .utils import bind


def clip_control_dict_rot(self, control_dict):
    """
    Note: If current camera rot (e.g. random init) is out of the boundary,
        rot policy step will be very large to make the next
        rot angle come back to the valid area.
    """
    pitch_low = getattr(self, 'pitch_low', -360)
    pitch_high = getattr(self, 'pitch_high', 360)
    yaw_low = getattr(self, 'yaw_low', -360)
    yaw_high = getattr(self, 'yaw_high', 360)

    higher_bound = np.asarray((pitch_high, yaw_high))
    lower_bound = np.asarray((pitch_low, yaw_low))

    for cam_id, rot_policy in control_dict['rotate_policy'].items():
        cam_rot = self.env_interaction.cam_params[cam_id][3:5]
        delta_high = higher_bound - cam_rot
        delta_low = lower_bound - cam_rot
        control_dict['rotate_policy'][cam_id][0] = max(
            min(rot_policy[0], delta_high[0]), delta_low[0]
        )
        control_dict['rotate_policy'][cam_id][1] = max(
            min(rot_policy[1], delta_high[1]), delta_low[1]
        )


class RotLimit(WrapperBase):
    """
    Can be placed in any order
    """

    def __init__(
        self,
        env,
        pitch_low=-360,
        pitch_high=360,
        yaw_low=-360,
        yaw_high=360,
    ):
        super().__init__(env)

        bind(env.unwrapped, clip_control_dict_rot)

        env.unwrapped.pitch_low = pitch_low
        env.unwrapped.pitch_high = pitch_high
        env.unwrapped.yaw_low = yaw_low
        env.unwrapped.yaw_high = yaw_high
