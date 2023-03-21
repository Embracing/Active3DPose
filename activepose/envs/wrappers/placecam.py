import random
from collections import OrderedDict

import numpy as np

from activepose.pose.multiviews import camera
from activepose.pose.utils2d.pose.skeleton_def import wholebody_kpinfo_dict

from .base import WrapperBase
from .utils import bind


def reset_cam_overwritten(self, init_human_num, init_objects):
    # self.env_config['camera_id_list'] = list(range(init_human_num + 1, init_human_num + 1 + self.num_cameras))

    if self.env_version == 12:
        self.env_config['camera_id_list'] = list(
            range(init_human_num + 1, init_human_num + 1 + self.num_cameras)
        )
    elif self.env_version == 13:
        self.env_config['camera_id_list'] = list(range(1, self.num_cameras + 1))
    else:
        raise NotImplementedError

    if getattr(self, 'shuffle_cam_id', False):
        random.shuffle(self.env_config['camera_id_list'])

    # make camera_param_dict
    high_bound = self.env_config['higher_bound_for_camera']  # 3
    low_bound = self.env_config['lower_bound_for_camera']  # 3

    if hasattr(self, 'init_loc_list') and len(self.init_loc_list) > 0:
        cam_loc_list = self.init_loc_list
    else:
        cam_loc_list = []
        if hasattr(self, 'place_on_slider') and self.place_on_slider:
            # random sample on slider
            # 'camera_move_along_axis_list': [0, 1, 0, 1]
            for idx in range(self.num_cameras):
                default_loc = self.env_config['camera_param_list'][idx][:3].copy()
                along_axis = self.env_config['camera_move_along_axis_list'][idx]
                default_loc[along_axis] = random.uniform(
                    low_bound[along_axis], high_bound[along_axis]
                )
                cam_loc_list.append(default_loc.tolist())
        else:
            for idx in range(self.num_cameras):
                cam_loc_list.append(
                    [random.uniform(low, high) for low, high in zip(low_bound, high_bound)]
                )

    cam_rot_list = self.cal_cam_init_direction(cam_loc_list)

    param_list = []
    for loc, rot in zip(cam_loc_list, cam_rot_list):
        param_list.append(np.array(loc + rot + [0, 90.0]))

    # 1st dim indicates free cameras
    if self.num_place_cameras is None:
        self.num_place_cameras = self.action_space.shape[0]
    self.env_config['camera_param_list'][: self.num_place_cameras] = param_list[
        : self.num_place_cameras
    ]

    self.env_config['camera_param_dict'] = OrderedDict(
        zip(self.env_config['camera_id_list'], self.env_config['camera_param_list'])
    )

    # do not recreate cameras
    camera_id_set_tobe_created = set(map(str, self.env_config['camera_id_list'])) - set(
        init_objects
    )
    if len(camera_id_set_tobe_created) > 0:
        camera_id_list_tobe_created = list(map(int, camera_id_set_tobe_created))
        self.env_interaction.create_cameras(camera_id_list_tobe_created)

    self.env_interaction.update_all_camera_parameters(self.env_config['camera_param_dict'])

    # reset camera models
    self.camera_model_list = []
    for cam_id in self.env_config['camera_id_list']:
        camera_param = self.env_config['camera_param_dict'][cam_id]
        CamModel = camera.CameraPose(
            *camera_param[:6], *self.env_interaction.resolution, camera_param[-1]
        )
        self.camera_model_list.append(CamModel)


def cal_cam_init_direction(self, cam_loc_list):
    """
    cam_loc_list: list of len C, each item is a list of [x,y,z]
    """
    if not hasattr(self, 'controller'):
        raise RuntimeError

    kp3d_list, _, _ = self.env_interaction.get_concurrent(
        human_list=self.controller.human_id_list, camera_list=[], viewmode_list=[]
    )

    used_unreal_joints_idx = list(wholebody_kpinfo_dict['mapping_dict_wholebody2unreal'].values())
    kp3d_arr = np.array(kp3d_list)  # [N, all_joints, 3]
    gt3d = kp3d_arr[:, used_unreal_joints_idx, :]  # [N, J, 3]

    # cam init direct to the center of all humans
    gt_human_location = gt3d.mean(axis=(0, 1))  # [3]

    expected_dir = gt_human_location[None, :] - np.asarray(cam_loc_list)  # [C, 3]
    dir_norm = np.linalg.norm(expected_dir, axis=-1, keepdims=True)  # [C, 1]

    np.divide(expected_dir, dir_norm, out=expected_dir, where=dir_norm != 0)  # [C, 3]

    pitch = np.arcsin(expected_dir[:, -1]) / np.pi * 180  # [C]
    yaw = np.arctan2(expected_dir[:, 1], expected_dir[:, 0]) / np.pi * 180  # [C]
    rot_policy = [[p, y] for p, y in zip(pitch, yaw)]  # a list of [pitch, yaw]

    return rot_policy


class PlaceCam(WrapperBase):
    """
    Can be placed in any order
    1: used to random place free cameras when reset() is called.
    2: used to designate locations for free cameras when reset() is called
    """

    def __init__(self, env, num_place_cameras=None, place_on_slider=False):
        super().__init__(env)

        if num_place_cameras is not None:
            assert (
                num_place_cameras <= env.num_cameras
            ), ' num of placed cameras {} must be less than existing cameras {}'.format(
                num_place_cameras, env.num_cameras
            )

        bind(env.unwrapped, reset_cam_overwritten)
        bind(env.unwrapped, cal_cam_init_direction)

        env.unwrapped.num_place_cameras = num_place_cameras
        env.unwrapped.place_on_slider = place_on_slider

    def reset(self, init_loc_list=list(), **kwargs):
        """
        [[x1,y1,z1], [x2, y2, z2], .. [xc, yc, zc]]
        can be smaller than C cameras
        Empty list --> random place cameras
        """
        self.env.unwrapped.init_loc_list = init_loc_list
        return self.env.reset(**kwargs)
