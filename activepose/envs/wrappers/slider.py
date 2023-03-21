import numpy as np

from .base import WrapperBase
from .utils import bind


def make_control_policy_overwritten(self, action):
    control_dict = dict()
    if self.rot_dims > 0:
        control_dict['rotate_policy'] = {
            cam_id: self.rotation_angle_matrix[
                [0, 1], move_action[self.move_dims : self.move_dims + self.rot_dims]
            ]  # pitch yaw
            for cam_id, move_action in zip(
                self.env_config['camera_id_list'][: self.num_movable_cameras], action
            )
        }

    if self.move_dims <= 0:
        raise RuntimeError

    camera_move_along_axis_list = self.env_config['camera_move_along_axis_list'][
        : self.num_movable_cameras
    ]
    move_distance = self.move_distance_matrix[camera_move_along_axis_list, action[:, 0]]  # [C]
    move_policy_matrix = np.zeros((self.num_movable_cameras, 3), dtype=np.float32)  # [C, 3] for xyz
    move_policy_matrix[
        list(range(self.num_movable_cameras)), camera_move_along_axis_list
    ] = move_distance  # fill

    control_dict['move_policy'] = dict()

    for idx, (cam_id, move_action) in enumerate(
        zip(
            self.env_config['camera_id_list'][: self.num_movable_cameras],
            move_policy_matrix,
        )
    ):
        control_dict['move_policy'][cam_id] = move_action

    return control_dict


def outer(org_act_dims, org_move_dims):
    def set_action_dim_override(self):
        """
        action dims
        move dims
        rot dims
        """
        return org_act_dims - 2, org_move_dims - 2, org_act_dims - org_move_dims

    return set_action_dim_override


class Slider(WrapperBase):
    """
    Put this wrapper in any order?
    """

    def __init__(self, env):
        super().__init__(env)

        bind(env.unwrapped, make_control_policy_overwritten)
        bind(
            env.unwrapped, outer(env.action_dims, env.move_dims)
        )  # remove pitch and yaw in last 2 dims

        env.unwrapped.load_config(
            map_name=self.config.ENV.MAP_NAME,
            env_name=env.config.ENV.ENV_NAME,
            num_humans=env.config.ENV.NUM_OF_HUMANS,
            walk_speed_range=env.config.ENV.WALK_SPEED_RANGE,
            rot_speed_range=env.config.ENV.ROTATION_SPEED_RANGE,
        )
