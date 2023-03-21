import numpy as np

from .base import WrapperBase
from .utils import bind


def outer1(index_list):
    def calculate_rulebased_rot_imitation_action(self):
        assignment_matrix = np.zeros((self.num_cameras, self.max_num_humans))
        # assignment_matrix[:, :self.controller.num_humans] = 1
        if index_list is not None:
            # assert len(index_list) <= self.max_num_humans
            assignment_matrix[:, index_list] = 1
        else:
            assignment_matrix[:, : self.controller.num_humans] = 1

        # always use gt3d to calculate rulebased rotation policy
        yaw_policy, pitch_policy = self.cal_control_policy_from_assignment_3d(
            assignment_matrix,
            self.current_state_dict,
            use_gt3d=True,
        )  # len == C

        pitch_action = np.argmin(
            np.abs(pitch_policy[:, None] - self.rotation_angle_matrix[0]), axis=-1
        )  # [C]
        yaw_action = np.argmin(
            np.abs(yaw_policy[:, None] - self.rotation_angle_matrix[1]), axis=-1
        )  # [C]

        pitch_onehot = np.zeros(
            (len(pitch_action), len(self.rotation_angle_matrix[0])), dtype=np.int32
        )
        pitch_onehot[np.arange(len(pitch_action)), pitch_action] = 1

        yaw_onehot = np.zeros((len(yaw_action), len(self.rotation_angle_matrix[1])), dtype=np.int32)
        yaw_onehot[np.arange(len(yaw_action)), yaw_action] = 1

        return yaw_onehot, pitch_onehot

    return calculate_rulebased_rot_imitation_action


class ImitateRot(WrapperBase):
    """
    Can be placed in any order
    """

    def __init__(self, env, index_list=[0]):
        super().__init__(env)

        bind(env.unwrapped, outer1(index_list))
        env.unwrapped.cal_rulebased_rot_imit = True
