import numpy as np

from .base import WrapperBase
from .utils import bind


def outer1(index_list, use_gt3d):
    def modify_control_dict_to_rulebased_rot(self, control_dict):
        assignment_matrix = np.zeros((self.num_cameras, self.max_num_humans))
        # assignment_matrix[:, :self.controller.num_humans] = 1
        if index_list is not None:
            # assert len(index_list) <= self.max_num_humans
            assignment_matrix[:, index_list] = 1
        else:
            assignment_matrix[:, : self.controller.num_humans] = 1

        yaw_policy, pitch_policy = self.cal_control_policy_from_assignment_3d(
            assignment_matrix, self.current_state_dict, use_gt3d
        )  # len == C

        # if num_rule_cameras is not specified,
        # all movable cameras are affected.
        if not hasattr(self, 'num_rule_cameras') or self.num_rule_cameras is None:
            self.num_rule_cameras = self.num_movable_cameras
        else:
            assert (
                self.num_rule_cameras >= self.num_movable_cameras
            ), f'rule based cameras {self.num_rule_cameras} should be >= movalble cameras {env.unwrapped.num_cameras}'

        # change rotate policy
        control_dict['rotate_policy'] = {
            cam_id: [pitch, yaw]  # pitch yaw
            for cam_id, pitch, yaw in zip(
                self.env_config['camera_id_list'][: self.num_rule_cameras],
                pitch_policy,
                yaw_policy,
            )
        }

    return modify_control_dict_to_rulebased_rot


def outer2(num):
    def set_action_dim_override(self):
        """
        action dims == num
        move dims == num
        rot dims == 0
        """
        return num, num, 0

    return set_action_dim_override


class RuleBasedRot(WrapperBase):
    """
    Can be placed in any order
    """

    def __init__(self, env, use_gt3d=False, num_rule_cameras=None, index_list=[0]):
        super().__init__(env)

        bind(env.unwrapped, outer1(index_list, use_gt3d))
        # Maybe env.unwrapped.action_space.shape[-1] ?
        bind(
            env.unwrapped, outer2(env.action_space.shape[-1] - 2)
        )  # remove pitch and yaw in last 2 dims

        env.unwrapped.rulebased_rot_use_gt3d = use_gt3d

        if num_rule_cameras is not None:
            assert (
                num_rule_cameras <= env.unwrapped.num_cameras
            ), f'rule based cameras {num_rule_cameras} should be <= total cameras {env.unwrapped.num_cameras}'
            env.unwrapped.num_rule_cameras = num_rule_cameras

        env.unwrapped.load_config(
            map_name=self.config.ENV.MAP_NAME,
            env_name=env.config.ENV.ENV_NAME,
            num_humans=env.config.ENV.NUM_OF_HUMANS,
            walk_speed_range=env.config.ENV.WALK_SPEED_RANGE,
            rot_speed_range=env.config.ENV.ROTATION_SPEED_RANGE,
        )
