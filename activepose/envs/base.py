import copy
import itertools
import math
import os
import random
import sys
import time
from collections import OrderedDict

import cv2
import gym
import numpy as np
import numpy.ma as ma

# import skimage
import zarr
from gym import spaces

from activepose.env_config import (
    get_env_config,
    get_human_config,
    get_human_model_config,
    get_map_config,
)
from activepose.pose.multiviews import camera, triangulate
from activepose.pose.refine.one_euro import OneEuroFilter
from activepose.pose.utils2d.pose.core import initialize_streaming_pose_estimator
from activepose.pose.utils2d.pose.skeleton_def import get_skeleton, mapping_tuple
from activepose.pose.utils2d.pose.utils import get_bbox_from_kp2d_batch
from activepose.pose.utils2d.visualization import visualization2d_wrapper
from activepose.utils.file import VideoSaver, ZarrVariableBufferSaver

from .internal import Interaction, MultiHumanController_v13, RewardMultiPerson, RunUnreal


class MultiviewPose(gym.Env):
    MULTI_AGENT = True

    def __init__(self, config, **kwargs):
        super().__init__()
        self.closed = False

        if config.ENV.MAP_NAME not in config.ENV.ALL_MAPS:
            raise RuntimeError(f'Map {config.ENV.MAP_NAME} not supported! {config.ENV.ALL_MAPS}')

        self.binary = RunUnreal(config.ENV.BINARY_PATH)

        start_port = 10000
        if 'worker_index' in kwargs and 'num_envs' in kwargs and 'num_envs' in kwargs:
            start_port += kwargs['worker_index'] * kwargs['num_envs'] + kwargs['vector_index']
        else:
            pass

        print(
            f'Worker Index: {kwargs["worker_index"]}, Vector Index: {kwargs["vector_index"]}, Start Port: {start_port}'
        )

        if 'linux' in sys.platform:
            ip, port, unix_socket_path = self.binary.start(
                port=start_port,
                resolution=config.ENV.RESOLUTION,
                render_driver=config.ENV.RENDER_DRIVER,
                map_name=config.ENV.MAP_NAME,
            )
            self.env_interaction = Interaction(
                '', ip=ip, port=port, unix_socket_path=unix_socket_path
            )
        elif 'win' in sys.platform:
            ip, port, _ = self.binary.start(
                port=start_port,
                resolution=config.ENV.RESOLUTION,
                map_name=config.ENV.MAP_NAME,
            )
            self.env_interaction = Interaction('', ip=ip, port=port)
        else:
            raise NotImplementedError
        print(f'Open map: {config.ENV.MAP_NAME}')

        self.config = config

        pose_estimator = initialize_streaming_pose_estimator(self.config)
        model_dict = {'pose2d': pose_estimator}
        self.model_dict = model_dict

        if self.config.REC3D.USE_TEMPORAL_SMOOTHING:
            if self.config.REC3D.USE_ONE_EURO:
                self.one_euro_config = {
                    'freq': 120,  # Hz
                    'mincutoff': 1.0,
                    'beta': 1.0,
                    'dcutoff': 1.0,
                }
                self.one_euro = OneEuroFilter(**self.one_euro_config)

        self.camera_model_list = []
        self.resolution = self.env_interaction.resolution
        self.evolution_steps = self.config.ENV.EVOLUTION_STEPS
        assert self.evolution_steps >= 1

        self.env_version = self.config.ENV.VERSION
        if self.env_version == 12:
            raise RuntimeError
        elif self.env_version == 13:
            self.controller_class = MultiHumanController_v13
        else:
            raise NotImplementedError
        self.yaw_coef = self.config.ENV.YAW_PID_COEF
        self.pitch_coef = self.config.ENV.PITCH_PID_COEF
        self.fov_coef = self.config.ENV.FOV_PID_COEF
        self.yaw_coef_3d = self.config.ENV.YAW_PID_COEF_3D
        self.pitch_coef_3d = self.config.ENV.PITCH_PID_COEF_3D
        self.fov_coef_3d = self.config.ENV.FOV_PID_COEF_3D

        self.exp_body_size = (
            self.config.ENV.BODY_EXPECTED_RATIO * self.resolution[0] * self.resolution[1]
        )
        self.exp_hand_size = (
            self.config.ENV.HAND_EXPECTED_RATIO * self.resolution[0] * self.resolution[1]
        )
        # self.num_joints = len(wholebody_kpinfo_dict['mapping_dict_wholebody2unreal'])
        self.num_joints = len(mapping_tuple[0])

        self.reward_func = RewardMultiPerson(reward_3d_func=self.config.REWARD.REC3D_FUNC)

        self.max_num_humans = self.config.ENV.MAX_NUM_OF_HUMANS
        self.move_distance_matrix = np.array(
            self.config.ENV.MOVE_DISTANCE_MATRIX
        )  # 3 different move distance for x,y,z
        self.rotation_angle_matrix = np.array(self.config.ENV.ROTATION_ANGLE_MATRIX)

        self.info_trajectory = []
        self.done_trigger_count = 0
        self.num_steps = 0
        self.qt_render_inited = False
        self.use_airwall_outer = False
        self.use_airwall_inner = False

        self.load_config(
            map_name=self.config.ENV.MAP_NAME,
            env_name=self.config.ENV.ENV_NAME,
            num_humans=self.config.ENV.NUM_OF_HUMANS,
            walk_speed_range=self.config.ENV.WALK_SPEED_RANGE,
            rot_speed_range=self.config.ENV.ROTATION_SPEED_RANGE,
        )  # load num_cameras
        self.num_humans = self.config.ENV.NUM_OF_HUMANS

    def load_config(
        self,
        map_name='Blank',
        env_name=None,
        num_humans=3,
        walk_speed_range=[20, 30],
        rot_speed_range=[80, 100],
    ):
        """
        Warning
            Call this func may change the dim of action & obs space
            because num of cameras may change
        """
        self.load_env_and_set_action_overwritten(env_name)

        # for param <= 0, random select from [1-7, 10]
        _, self.human_config = copy.deepcopy(get_human_config(num_humans=num_humans))
        assert walk_speed_range is not None, 'walk_speed_range is None'
        assert rot_speed_range is not None, 'rot_speed_range is None'
        self.human_config['walk_speed_range'] = walk_speed_range
        self.human_config['rot_speed_range'] = rot_speed_range

        # Get map specific config
        self.map_config = copy.deepcopy(get_map_config(map_name))

        # Modify env_config according to the map config
        self.env_config['area'] += self.map_config['map_center'][:2]

        for item in self.env_config['camera_param_list']:
            item[:3] += self.map_config['map_center']

        self.env_config['floor_z_of_human'] = self.map_config['human_z']
        self.env_config['lower_bound_for_camera'] = (
            self.map_config['map_center'] + self.env_config['lower_bound_for_camera']
        ).tolist()

        self.env_config['higher_bound_for_camera'] = (
            self.map_config['map_center'] + self.env_config['higher_bound_for_camera']
        ).tolist()

        # Modify human_config
        for location_item in self.human_config['human_location_list']:
            location_item[:2] += self.map_config['map_center'][:2]
            location_item[-1] += self.map_config['human_z']

        # get NPC model config
        self.human_model_name, self.human_model_config = get_human_model_config(
            self.config.ENV.HUMAN_MODEL
        )

        # Modify reward func (mpjpe for missed joints) to account for map center offset
        self.reward_func.set_offset(self.map_config['map_center'])

        """
        F_human 20
        3d: (9) human_box3d:6, human_direction3d(xy):2, human_lost_joint_ratio:1,
        2d: (6) human_box2d:4, box_depth:1, box_IoU-target:1,
        Flag: (5)
        3d valid rec flag, 3d valid dir flag
        occlude-target, 2d valid box flag, 2d valid box depth flag
        """
        F_human_scale3d_len = 3
        F_human_lost_j_ratio_len = 1
        F_human_attr3d_len = F_human_scale3d_len + F_human_lost_j_ratio_len

        F_human_center3d_len = 3
        F_human_dir3d_len = 3

        F_human_world3d_len = (
            F_human_scale3d_len
            + F_human_lost_j_ratio_len
            + F_human_center3d_len
            + F_human_dir3d_len
        )
        F_human_local3d_len = F_human_center3d_len + F_human_dir3d_len
        F_human_3d_len = F_human_world3d_len + F_human_local3d_len

        F_human_box2d_len = 4
        F_human_depth2d_len = 1
        F_human_IoT_len = 1
        F_human_2d_len = F_human_box2d_len + F_human_depth2d_len + F_human_IoT_len

        F_human_flag_len = 6

        self.len_feature_human = F_human_3d_len + F_human_2d_len + F_human_flag_len  # 28

        low_area_bounds = (self.env_config['area'] - 20).min(axis=0)
        high_area_bounds = (self.env_config['area'] + 20).max(axis=0)

        low_human = np.zeros((self.len_feature_human,), dtype=np.float32)

        low_human[F_human_attr3d_len:F_human_3d_len] = [
            *low_area_bounds,
            0,
            -1,
            -1,
            -1,
        ] + [*(low_area_bounds - high_area_bounds), 0, -1, -1, -1]
        low_human[-F_human_flag_len:] = -1.0
        low_camera = np.array(
            [0] + self.env_config['lower_bound_for_camera'] + [-1, -1, -1] + [-1, -1],
            dtype=np.float32,
        )
        low_env = np.asarray([0] + [-1] * self.max_num_humans, dtype=np.float32)  # [8]
        low = np.concatenate(
            [
                low_env,  # [8]
                low_camera,  # [9]
                np.tile(low_human, self.max_num_humans).reshape((-1,)),
            ],
            axis=0,
        )  # [8+8+N_max*F_human]
        low = np.tile(low, (self.evolution_steps, self.num_cameras, 1)).astype(
            np.float32
        )  # [E, C, 8+8+N_max*F_human]

        high_human = np.zeros((self.len_feature_human,), dtype=np.float32)
        high_human[:F_human_attr3d_len] = [1, 1, 1, 1]
        high_human[F_human_attr3d_len:F_human_3d_len] = [
            *high_area_bounds,
            300,
            1,
            1,
            1,
        ] + [
            *(high_area_bounds - low_area_bounds),
            (high_area_bounds - low_area_bounds).max(),
            1,
            1,
            1,
        ]
        high_human[F_human_3d_len : F_human_3d_len + F_human_box2d_len] = [
            self.resolution[0],
            self.resolution[1],
            self.resolution[0],
            self.resolution[1],
        ]
        high_human[F_human_3d_len + F_human_box2d_len] = (high_area_bounds - low_area_bounds).max()
        high_human[F_human_3d_len + F_human_box2d_len + F_human_depth2d_len :] = 1
        high_human[F_human_3d_len + F_human_2d_len] = self.max_num_humans

        high_camera = np.array(
            [np.inf] + self.env_config['higher_bound_for_camera'] + [1, 1, 1] + [1, 1],
            dtype=np.float32,
        )
        high_env = np.asarray([np.inf] + [1] * self.max_num_humans, dtype=np.float32)  # [8]
        high = np.concatenate(
            [
                high_env,  # [8]
                high_camera,  # [9]
                np.tile(high_human, self.max_num_humans).reshape((-1,)),
            ],
            axis=0,
        )  # [8+9+N_max*F_human]
        high = np.tile(high, (self.evolution_steps, self.num_cameras, 1)).astype(
            np.float32
        )  # [E, C, 8+8+N_max*F_human]

        # [E, C, 8+8+N_max*F_human]
        self.observation_space = spaces.Box(
            low=low,
            high=high,
            dtype=np.float32,
        )

    def load_env_and_set_action_overwritten(self, env_name):
        """ """
        _, self.env_config = copy.deepcopy(get_env_config(env_name))  # None -> random choice
        self.num_cameras = self.env_config['num_cameras']
        self.num_movable_cameras = self.movable_cameras_override()
        self.action_dims, self.move_dims, self.rot_dims = self.set_action_dim_override()

        # [C, 5], each item in matrix shapes: xyz, pitch, yaw
        multipliers = self.move_dims * [self.move_distance_matrix.shape[1]] + self.rot_dims * [
            self.rotation_angle_matrix.shape[1]
        ]

        self.action_space = spaces.MultiDiscrete(
            multipliers * np.ones((self.num_movable_cameras, self.action_dims), dtype=np.int32)
        )

    def movable_cameras_override(self):
        return self.num_cameras

    def set_action_dim_override(self):
        """
        Default: x, y, z, pitch, yaw
        Return action dims, mov dims, rot dims
        """
        return 5, 3, 2

    def reset(self, **kwargs):
        """
        Reset environment:
        Reset camera params, human location & orientation.
        Reset other things.
        """

        # ========== reset env ===================
        # reset env cameras
        # we do not delete humans in the env.
        # So camera_id has to begin with init num of humans + 1,

        # self.env_interaction.destroy_existing_cameras()
        init_human_num = 0
        self.num_steps = 0

        init_objects = self.env_interaction.get_objects()
        for obj_name in init_objects:
            if 'human' in obj_name:
                init_human_num += 1

        self.reset_human_overwritten()

        self.reset_cam_overwritten(init_human_num, init_objects)

        # get_observation
        (
            observation_single,
            info_first_step,
        ) = self._get_observation_single_step_multi_person()
        observation = np.tile(observation_single, (self.evolution_steps, 1, 1))

        # reset info_trajectory
        self.info_trajectory = [info_first_step]

        self.done_trigger_count = 0

        return observation.astype(np.float32)

    def reset_cam_overwritten(self, init_human_num, init_objects):
        """
        Put this func after reset_human() because somtimes
        we need to know where humans are in advance.
        """
        # for v12 env, each human is associated with a tracking camera
        # for v13 env, the tracking camera is removed
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

        if getattr(self, 'EMA_action', False):
            self.EMA_t = None

        self.env_config['camera_param_dict'] = OrderedDict(
            zip(self.env_config['camera_id_list'], self.env_config['camera_param_list'])
        )

        # do not recreate cameras
        camera_id_set_tobe_created = set(map(str, self.env_config['camera_id_list'])) - set(
            init_objects
        )
        if len(camera_id_set_tobe_created) > 0:
            camera_id_list_tobe_created = list(map(int, camera_id_set_tobe_created))
            # self.env_interaction.create_cameras(camera_id_list_tobe_created)
            self.env_interaction.create_cameras_async(
                camera_id_list_tobe_created, self.env_config['camera_id_list']
            )

        self.env_interaction.update_all_camera_parameters_async(
            self.env_config['camera_param_dict']
        )

        # reset camera models
        self.camera_model_list = []
        for cam_id in self.env_config['camera_id_list']:
            # camera_param = self.env_config['camera_param_dict'][cam_id].tolist()
            camera_param = self.env_config['camera_param_dict'][cam_id]
            CamModel = camera.CameraPose(
                *camera_param[:6], *self.env_interaction.resolution, camera_param[-1]
            )
            self.camera_model_list.append(CamModel)

    def reset_human_overwritten(self):
        """
        Reset Humans
        Can be overwritter
        """
        self.controller = self.controller_class(
            self.env_interaction,
            self.human_config,
            self.env_config,
            action_setting=self.human_config['action'],
            random_init_locrot=getattr(self, 'random_init_loc_rot', False),
            changable_mesh=self.human_model_config['changable_mesh'],
        )

        # waiting for humans to be ready
        print('Waiting for env humans to be ready (1s)')
        time.sleep(1)

        # reset action
        # self.controller.reset_action_all()
        self.controller.reset_walk_all_async()

        # freeze walking
        if self.config.ENV.FREEZE_WALK_MODE == 'pause_game':
            self.env_interaction.pause_game_async()
        elif self.config.ENV.FREEZE_WALK_MODE == 'zero_speed':
            self.controller.freeze_walk_all()

    def step(self, action):
        """
        Action:
            [C, 5], 3: xyz, 5: xyz,pitch,yaw
            e.g. data[i, j] is in [0, 2], represents -10, 0, +10 move of camera_i, (xyz)_j,
        Take action as input, calculate rule-based control strategy for each camera,
        execute a fixed (e.g.20) steps of evolution,
        observation:
            [C, 6+N_max*j*6]
        reward: scalar
        done: True or False
        info:
            {"gt2d": [20, C, j, 2],
             "gt3d": [20, C, j, 3],
             "pred3d": [20, C, j, 3],
             "img": [20, h, w, 4],
             "gt_bblist": a list of [img1_obj, img2_obj, ... imgC_obj],
             "gt_rhand_bblist": a list of [img1_rhand, img2_rhand, ... imgC_rhand],
             "gt_lhand_bblist": a list of [img1_lhand, img2_lhand, ... imgC_lhand]}
        """
        assert len(self.info_trajectory) > 0, 'env must be reset before step'
        self.current_state_dict = self.info_trajectory[-1]  # take last info
        self.info_trajectory = []  # clear infos from last evoluation period.
        done = False
        self.num_steps += 1
        # assert len(action) == self.num_cameras

        observation_list = []

        if self.config.ENV.FREEZE_WALK_MODE == 'pause_game':
            self.env_interaction.resume_game_async()
        elif self.config.ENV.FREEZE_WALK_MODE == 'zero_speed':
            self.controller.restore_walk_all()

        for step in range(self.evolution_steps):
            # control with single step observation

            control_dict = self.make_control_policy_overwritten(action)
            at_airwall_outer = self.clip_control_dict_outer(control_dict)
            at_airwall_inner = self.clip_control_dict_inner(control_dict)
            at_airwall = at_airwall_outer or at_airwall_inner
            self.modify_control_dict_to_rulebased_rot(control_dict)
            self.clip_control_dict_rot(control_dict)  # clip pitch and yaw

            self.env_interaction.rotate_and_move_camera_async(control_dict)

            # get next observation
            (
                observation_single,
                info_single,
            ) = self._get_observation_single_step_multi_person()
            prev_info = self.current_state_dict.copy()
            prev_info.pop('prev_info', None)
            info_single['prev_info'] = prev_info
            info_single['at_airwall'] = at_airwall

            reward_dict = self.get_reward_overwritten(info_single)

            info_single['at_airwall'] = at_airwall
            info_single.update(reward_dict)
            info_single.update(control_dict)

            info_single['num_steps'] = self.num_steps
            self.info_trajectory.append(info_single)
            observation_list.append(observation_single)
            self.current_state_dict = self.info_trajectory[-1]

            # self.controller.step_action_all()
            self.controller.step_walk_all_async()

            # detect if this env is done
            if info_single['lost_joints_ratio'] >= self.config.REC3D.DONE_LOST_JOINTS_RATIO:
                self.done_trigger_count += 1
                if self.done_trigger_count >= self.config.ENV.DONE_LOST_LAST_STEPS:
                    done = True
            else:
                self.done_trigger_count = max(self.done_trigger_count - 1, 0)

        # return a scalar reward
        reward = 0
        for info_dict in self.info_trajectory:
            reward += info_dict['reward_3d']
        reward /= self.evolution_steps

        # observation
        observation = np.array(observation_list, dtype=np.float32)

        if self.config.ENV.FREEZE_WALK_MODE == 'pause_game':
            self.env_interaction.pause_game_async()
        elif self.config.ENV.FREEZE_WALK_MODE == 'zero_speed':
            self.controller.freeze_walk_all()

        return observation, reward, done, self.info_trajectory

    def clip_control_dict_outer(self, control_dict):
        """
        Return:
            if camera stucked at airwall
        """
        if not self.use_airwall_outer:
            return False

        # check if cameras exceed boundary
        lower_bound = np.asarray(self.env_config['lower_bound_for_camera'])
        higher_bound = np.asarray(self.env_config['higher_bound_for_camera'])

        stucked = False
        for cam_id, move_action in control_dict['move_policy'].items():
            cam_loc = self.env_interaction.cam_params[cam_id][:3]
            expected_cam_loc = cam_loc + move_action
            exceed_dim = np.logical_or(
                expected_cam_loc > higher_bound, expected_cam_loc < lower_bound
            )  # [3]
            if np.any(exceed_dim):
                control_dict['move_policy'][cam_id] = np.where(exceed_dim, 0, move_action)
                stucked = True
        return stucked

    def clip_control_dict_inner(self, control_dict):
        """
        Return:
            if camera stucked at airwall
        """
        if not hasattr(self, 'use_airwall_inner') or not self.use_airwall_inner:
            return False

        # check if cameras in boundary
        lower_bound = np.asarray(self.lower_bound_inner)
        higher_bound = np.asarray(self.higher_bound_inner)

        stucked = False
        for cam_id, move_action in control_dict['move_policy'].items():
            cam_loc = self.env_interaction.cam_params[cam_id][:3]
            expected_cam_loc = cam_loc + move_action
            expected_in_area_dim = np.logical_and(
                expected_cam_loc < higher_bound, expected_cam_loc > lower_bound
            )  # [3]
            if np.all(expected_in_area_dim):
                curr_in_area_dim = np.logical_and(
                    cam_loc < higher_bound, cam_loc > lower_bound
                )  # [3]
                prohibit_action_dim = expected_in_area_dim.astype(
                    np.int32
                ) - curr_in_area_dim.astype(np.int32)
                control_dict['move_policy'][cam_id] = np.where(prohibit_action_dim, 0, move_action)
                stucked = True
        return stucked

    def modify_control_dict_to_rulebased_rot(self, control_dict):
        """
        Override this func, all cal_control_policy_from_assignment_3d() here
        """
        return

    def clip_control_dict_rot(self, control_dict):
        """
        Override this func
        """
        return

    def make_control_policy_overwritten(self, action):
        transformation_vector_list = []
        for CamModel in self.camera_model_list:
            pitch, yaw = CamModel.pitch, CamModel.yaw
            # print('angle:', pitch, yaw)
            if getattr(self, 'ego_action', False):
                transformation_vector_list.append(
                    np.array(
                        (
                            np.cos(yaw / 180.0 * np.pi),
                            np.sin(yaw / 180.0 * np.pi),
                            1,
                        ),
                        dtype=np.float32,
                    )[
                        ..., None
                    ]  # [3, 1]
                )
            else:
                transformation_vector_list.append(np.ones((3, 1), dtype=np.float32))

        control_dict = dict()
        if self.rot_dims > 0:
            control_dict['rotate_policy'] = {
                cam_id: self.rotation_angle_matrix[
                    [0, 1], move_action[self.move_dims : self.move_dims + self.rot_dims]
                ]  # pitch yaw
                for cam_id, move_action in zip(
                    self.env_config['camera_id_list'][: self.num_movable_cameras],
                    action,
                )
            }
        if self.move_dims > 0:
            # ===== legacy code =====
            # """
            # Some miscalculations in the original code, but still works kinda well
            # """
            # control_dict['move_policy'] = {
            #     cam_id: (self.move_distance_matrix * transformation_vector_list[idx])[[0, 1, 2], move_action[:self.move_dims]]
            #     for idx, (cam_id, move_action) in enumerate(zip(self.env_config['camera_id_list'][:self.num_movable_cameras], action))
            # }
            # =======================

            control_dict['move_policy'] = {}
            for idx, (cam_id, move_action) in enumerate(
                zip(
                    self.env_config['camera_id_list'][: self.num_movable_cameras],
                    action,
                )
            ):
                expected_loc = self.move_distance_matrix[
                    [0, 1, 2], move_action[: self.move_dims]
                ]  # [3]
                CamModel = self.camera_model_list[idx]
                action_in_world = CamModel.cam_to_world(expected_loc[None, ...]).squeeze(
                    0
                ) - np.array([CamModel.x, CamModel.y, CamModel.z])
                control_dict['move_policy'][cam_id] = action_in_world

            if getattr(self, 'add_extra_force', False):
                hum_z_max = self.gt3d[..., -1].max()  # [N]
                hum_xy = self.gt3d[:, :, :2].mean(axis=1)  # [N, 2]
                camera_xyz = np.array(
                    [[CamModel.x, CamModel.y, CamModel.z] for CamModel in self.camera_model_list]
                )

                F = self.force_configs['F']
                human_safe_distance = self.force_configs['human_safe_distance']
                camera_safe_distance = self.force_configs['camera_safe_distance']

                for idx, cam_id in enumerate(
                    self.env_config['camera_id_list'][: self.num_movable_cameras]
                ):
                    CamModel = self.camera_model_list[idx]
                    xyz = np.array([CamModel.x, CamModel.y, CamModel.z])
                    xy, z = xyz[0:2], xyz[-1]

                    if z > hum_z_max * 1.5:
                        continue
                    human_diff = hum_xy - xy
                    human_norm = np.linalg.norm(human_diff, axis=1)  # [N]
                    human_mask = human_norm <= human_safe_distance  # cm

                    cam_diff = np.delete(camera_xyz, idx, axis=0) - xyz
                    cam_norm = np.linalg.norm(cam_diff, axis=1)  # [N]
                    cam_mask = cam_norm <= camera_safe_distance  # cm

                    coeff = F
                    if np.any(human_mask) or np.any(cam_mask):
                        force_list = []

                        if np.any(human_mask):
                            human_xy_force = -(coeff * human_diff / (human_norm[..., None] + 1e-8))[
                                human_mask
                            ].mean(axis=0)
                            human_force = np.append(human_xy_force, F)  # (1 ,3)
                            force_list.append(human_force)

                        if np.any(cam_mask):
                            cam_force = -(coeff * cam_diff / (cam_norm[..., None] + 1e-8))[
                                cam_mask
                            ].mean(axis=0)
                            force_list.append(cam_force)  # (N ,3)

                        v = control_dict['move_policy'][cam_id]

                        v += np.mean(force_list, axis=0)

                        inf_norm = np.linalg.norm(v, ord=np.inf)
                        if inf_norm > F:
                            v = v * F / inf_norm
                        control_dict['move_policy'][cam_id] = v
        else:
            raise RuntimeError

        if getattr(self, 'action_noise', False):
            noise_factor = np.random.uniform(low=self.action_noise[0], high=self.action_noise[1])
            for cam_id in control_dict['move_policy']:
                control_dict['move_policy'][cam_id] *= noise_factor

        if getattr(self, 'EMA_action', False):
            if not self.EMA_t:
                self.EMA_t = control_dict['move_policy']
            else:
                cur_policy = control_dict['move_policy']
                for cam_id in cur_policy:
                    cur_policy[cam_id] = self.EMA_t[cam_id] + self.k_EMA * (
                        cur_policy[cam_id] - self.EMA_t[cam_id]
                    )
                self.EMA_t = cur_policy.copy()
        return control_dict

    def get_reward_overwritten(self, info):
        """
        Get reward
        Can be overwritten by child env
        """
        reward_3d, reward_3d_dict = self.reward_func.reward_error3d(
            info['pred3d'], info['gt3d']
        )  # scalar
        reward_dict = reward_3d_dict

        if getattr(self, 'partial_triangulation_3d_dict', False):
            partial_reward_dict = {}
            partial_mpjpe_dict = {}
            partial_pck20_dict = {}
            for k, v in self.partial_triangulation_3d_dict.items():
                (
                    reward_3d_partial,
                    reward_3d_partial_extra,
                ) = self.reward_func.reward_error3d(
                    v, info['gt3d']
                )  # scalar
                partial_reward_dict[k] = reward_3d_partial
                partial_mpjpe_dict[k] = reward_3d_partial_extra['mpjpe_3d']
                partial_pck20_dict[k] = reward_3d_partial_extra['pck3d_20']

            # add an all-camera triangulation result
            default_key = tuple(range(self.num_cameras))
            partial_reward_dict[default_key] = reward_dict['reward_3d']
            partial_mpjpe_dict[default_key] = reward_dict['mpjpe_3d']
            partial_pck20_dict[default_key] = reward_dict['pck3d_20']
            reward_dict['shapley_reward_dict'] = partial_reward_dict
            reward_dict['shapley_mpjpe_dict'] = partial_mpjpe_dict
            reward_dict['shapley_pck20_dict'] = partial_pck20_dict

        dummy_indv_reward = np.zeros((self.num_cameras,), dtype=np.float32)

        if getattr(self, 'use_reward_covered_by_two', False):
            reward_vis, reward_vis_dict = self.reward_func.reward_visibility_indv(
                info['pred2d_scores'],
                # current_state_dict['pred2d_scores'],
                threshold=self.config.REWARD.VISIBILITY_THRESH,
            )
        else:
            reward_vis_dict = {'reward_vis': dummy_indv_reward}
        reward_dict.update(reward_vis_dict)

        # Punish camera when it exceeds boundary
        if getattr(self, 'use_reward_camera_state', False):
            reward_cam, reward_cam_dict = self.reward_func.reward_camera_state_indv(
                info['cam_param_list'],
                self.env_config['lower_bound_for_camera'],
                self.env_config['higher_bound_for_camera'],
            )
        else:
            reward_cam_dict = {'reward_camera_state': dummy_indv_reward}
        reward_dict.update(reward_cam_dict)

        if getattr(self, 'use_reward_centering', False):
            # TODO: tracking matrix accroding to high level network
            tracking_matrix = np.zeros((self.num_cameras, self.num_humans), dtype=bool)  # [C, N]
            tracking_matrix[:, 0] = True
            reward_centering, reward_centering_dict = self.reward_func.reward_centering(
                self.camera_model_list,
                info['gt3d'],
                tracking_matrix,
            )
        else:
            reward_centering_dict = {'reward_centering': dummy_indv_reward}
        reward_dict.update(reward_centering_dict)

        if getattr(self, 'use_reward_obstruction', False):
            tracking_matrix = np.zeros((self.num_cameras, self.num_humans), dtype=bool)  # [C, N]
            tracking_matrix[:, 0] = True
            (
                reward_obstruction,
                reward_obstruction_dict,
            ) = self.reward_func.reward_obstruction(
                self.camera_model_list, info['gt3d'], tracking_matrix
            )
        else:
            reward_obstruction_dict = {'reward_obstruction': dummy_indv_reward}
        reward_dict.update(reward_obstruction_dict)

        if getattr(self, 'use_reward_iot', False):
            reward_iot, reward_iot_dict = self.reward_func.reward_bb2d_iot_indv(
                info['gt_obs_dict']['obs_target_bb2d_IoT'],
                info['gt_obs_dict']['obs_target_occ_flag'],
            )
        else:
            reward_iot_dict = {'reward_iot': dummy_indv_reward}
        reward_dict.update(reward_iot_dict)

        if getattr(self, 'use_reward_distance', False):
            _, reward_distance_dict = self.reward_func.reward_distance(
                info['gt3d'],
                info['cam_param_list'],
            )
        else:
            reward_distance_dict = {'reward_distance': dummy_indv_reward}
        reward_dict.update(reward_distance_dict)

        if getattr(self, 'use_reward_anti_collision', False):
            _, reward_collision_dict = self.reward_func.reward_anti_collision(
                info['cam_param_list'], info['gt3d']
            )
        else:
            reward_collision_dict = {'reward_anti_collision': dummy_indv_reward}
        reward_dict.update(reward_collision_dict)

        return reward_dict

    def _get_observation_single_step_multi_person(self):
        """
        Get imgs from the environment and predict corresponding 2d keypoints.
        """
        _process_time_dict = {}

        start_time = time.time()
        kp3d_list, param_list, img_list = self.env_interaction.get_concurrent(
            human_list=self.controller.human_id_list,
            camera_list=self.env_config['camera_id_list'],
            viewmode_list=['lit', 'object_mask'],
        )
        elapsed_time = time.time() - start_time
        _process_time_dict['get_concurrent'] = elapsed_time

        # keep joints in use
        used_unreal_joints_idx = mapping_tuple[1][self.human_model_name]

        if isinstance(used_unreal_joints_idx, dict):
            # different filters
            kp3d_list = [item[used_unreal_joints_idx[len(item)]] for item in kp3d_list]
            gt3d = np.array(kp3d_list)  # [N, J, 3]
        else:
            # same filter
            kp3d_arr = np.array(kp3d_list)  # [N, all_joints, 3]
            gt3d = kp3d_arr[:, used_unreal_joints_idx, :]  # [N, J, 3]

        self.gt3d = gt3d

        # update camera models and get proj2d
        proj2d_wd = []
        for CamModel, camera_param in zip(self.camera_model_list, param_list):
            CamModel.update_camera_parameters_list(camera_param)
            proj2d_wd.append(CamModel.project_to_2d(gt3d.reshape((-1, 3)), return_depth=True))
        proj2d_wd = np.array(proj2d_wd)  # [C, N*J, 3]
        proj2d_wd = proj2d_wd.reshape(
            (self.num_cameras, self.controller.num_humans, -1, 3)
        )  # [C, N, J, 3], with depth to verify if this projection is valid

        proj2d = proj2d_wd[..., :2]  # [C, N, J, 2]
        proj_depth = proj2d_wd[..., -1]  # [C, N, J]

        # get keypoint visibility, exclude the out-of-view keypoints
        proj2d_vis = self.env_interaction.get_kp_visibility_from_mask(
            img_list[1],
            [f'human{id}' for id in self.controller.human_id_list],
            proj2d,  # [C, N, J, 2]
        )  # [C, N, j]

        # exclude keypoints at the back of camera
        valid_depth = []
        for CamModel, kp_depth in zip(self.camera_model_list, proj_depth):
            valid_depth.append(kp_depth > CamModel.f)  # [N, J]
        valid_depth = np.array(valid_depth, dtype=np.int32)  # [C, N, J]
        proj2d_vis = proj2d_vis * valid_depth  # [C, N, J]

        # exclude the eye joint, because there is a little misalignment.
        target2d_vis = proj2d_vis[:, 0, :-2]  # [C, J-2], visibility of the target
        vis_ratio_2d = np.sum(target2d_vis) / np.prod(target2d_vis.shape)

        vis_ratio_3d = np.sum(target2d_vis, axis=0) >= 2
        vis_ratio_3d = np.sum(vis_ratio_3d) / np.prod(vis_ratio_3d.shape)

        # get pred2d
        img_batch = np.array(img_list[0])  # [C, h, w, 4] RGBA, 0-255
        img_batch = img_batch[..., :-1][..., ::-1]  # [C, h, w, 3] BGR
        input_data_dict = {'imgs': img_batch}

        bb_from_mask_list = self.env_interaction.get_bb_batch(
            img_list[1],
            [f'human{id}' for id in self.controller.human_id_list],
            margin=0,
        )  # [img1_obj1, img2_obj1, ..., img1_obj2, img2_obj2]. [N*C], None for not viewed person
        input_data_dict['bb'] = bb_from_mask_list

        res_dict = self.model_dict['pose2d'](input_data_dict)  # crop zero-sized patch for None bb

        used_pred_joints_idx = mapping_tuple[0]
        pred2d = res_dict['pose2d']  # [N*C, j, 2]
        pred2d = pred2d[:, used_pred_joints_idx, :]  # [N*C, j, 2]

        pred2d_scores = res_dict['pose2d_scores']  # [N*C, j, 1]
        pred2d_scores = pred2d_scores[:, used_pred_joints_idx, :]  # [N*C, j, 1]

        if self.config.REC3D.TRIANGULATION_STRATEGY == 'confidence':
            scores = pred2d_scores.squeeze(-1).reshape(
                (self.controller.num_humans, self.num_cameras, -1)
            )  # [N, C, j]
            joints_vis = np.where(scores > self.config.REC3D.VISIBILITY_THRESH, 1, 0)  # [N, C, j]

            joints_vis = joints_vis.reshape((-1, self.num_joints))  # [N*C, j]

        elif self.config.REC3D.TRIANGULATION_STRATEGY == 'all':
            joints_vis = None

        pred2d = pred2d.reshape(
            (
                self.num_humans,
                self.num_cameras,
                self.num_joints,
                2,
            )
        )
        pred2d = pred2d.swapaxes(0, 1)  # [C, N, J, 2]

        if joints_vis is not None:
            joints_vis = joints_vis.reshape((self.num_humans, self.num_cameras, self.num_joints))
            joints_vis = joints_vis.swapaxes(0, 1)  # [C, N, J]

        start_time = time.time()
        if self.config.REC3D.USE_RANSAC and self.num_cameras > 2:
            pred3d = triangulate.fast_ransac(self.camera_model_list, pred2d, None)  # [N, j, 3]
        else:
            pred3d = triangulate.fast_triangulate(
                self.camera_model_list,
                pred2d,
                joints_vis,
            )  # [N, j, 3]

        if getattr(self, 'partial_triangulation', False):
            self.partial_triangulation_3d_dict = {}
            if self.num_cameras > 2:
                # calculate all combinations
                # 2, 3, 4, ..., n-1
                keys = []
                num_cam_idx = [i for i in range(self.num_cameras)]
                for group_size in range(2, self.num_cameras):
                    group_index = list(itertools.combinations(num_cam_idx, group_size))
                    keys.extend(group_index)

                for key in keys:
                    part_cam_model_list = [
                        CamModel
                        for cam_id, CamModel in enumerate(self.camera_model_list)
                        if cam_id in key
                    ]
                    part_pred2d = pred2d[key, ...]
                    part_joints_vis = joints_vis[key, ...]
                    new_res = triangulate.fast_triangulate(
                        part_cam_model_list,
                        part_pred2d,
                        part_joints_vis,
                    )  # [N, j, 3]
                    self.partial_triangulation_3d_dict[key] = new_res

        if self.config.REC3D.USE_TEMPORAL_SMOOTHING:
            if self.config.REC3D.COPY_LAST_PRED:
                lost_joints = ~np.any(pred3d, axis=-1)  # [N, j]
                if self.num_steps > 0:
                    # not the first reset observation
                    info_last = self.current_state_dict
                    pred3d_last = info_last['pred3d']  # [N_max, j, 3]
                    pred3d_last = pred3d_last[: len(pred3d)]  # [N, j, 3]
                    pred3d[lost_joints] = pred3d_last[lost_joints]
                else:
                    # the first step
                    pass
            if self.config.REC3D.USE_ONE_EURO:
                pred3d_filterd = self.one_euro(
                    pred3d, self.num_steps / self.one_euro_config['freq']
                )
                pred3d = pred3d_filterd  # [N, j, 3]

        # number of joints that have not been reconstructed
        rec_joints = np.any(pred3d, axis=-1)  # [N, j]
        rec_joints = rec_joints[[0], :]  # [1, j]
        lost_joints_ratio = 1 - rec_joints.sum() / np.prod(rec_joints.shape)

        # get observation and info dict

        # do some convertion
        fill_num = self.max_num_humans - self.controller.num_humans

        pred2d_info = np.concatenate(
            [pred2d, np.zeros((self.num_cameras, fill_num, pred2d.shape[2], 2))], axis=1
        )  # [C, N_max, j, 2]

        pred2d_scores = pred2d_scores.reshape((self.controller.num_humans, self.num_cameras, -1, 1))
        pred2d_scores = np.swapaxes(pred2d_scores, 0, 1)  # [C, N, j, 1]
        pred2d_scores_info = np.concatenate(
            [
                pred2d_scores,
                np.zeros((self.num_cameras, fill_num, pred2d_scores.shape[2], 1)),
            ],
            axis=1,
        )  # [C, N_max, j, 1]

        pred_bb_array = get_bbox_from_kp2d_batch(
            pred2d_info.reshape((self.num_cameras * self.max_num_humans, -1, 2)), 0.2
        )  # [C*N_max, 4]
        pred_bb_array = pred_bb_array.reshape((self.num_cameras, -1, 4))  # [C, N_max, 4]

        lost_humans_each_view = np.zeros(
            (self.num_cameras,), dtype=np.int32
        )  # how many humans lost in this view
        bb_from_mask_list_converted = [[] for _ in range(self.num_cameras)]  # [C]
        for idx, bb in enumerate(bb_from_mask_list):
            if bb is None:
                bb = np.zeros((4,), dtype=np.int32)
                lost_humans_each_view[idx % self.num_cameras] += 1
            bb_from_mask_list_converted[idx % self.num_cameras].append(bb)
        bb_from_mask_array = np.array(bb_from_mask_list_converted, dtype=np.int32)  # [C, N, 4]

        pred3d_info = np.concatenate(
            [pred3d, np.zeros((fill_num, pred3d.shape[1], 3))], axis=0
        )  # [N_max, j, 3]

        # ================= return value ===========================
        # always make gt observation, for supervision
        gt_observation, gt_obs_dict = self.make_observation(
            pred3d, gt3d, bb_from_mask_array, param_list, use_gt3d=True
        )

        cam_human_distance = []
        for CamModel in self.camera_model_list:
            xyz = np.array([CamModel.x, CamModel.y, CamModel.z])
            cam_human_distance.append(np.linalg.norm(xyz - gt3d, axis=-1).min(axis=-1))

        # return max_num_humans for pred items, num_humans for gt items
        info = {
            'lost_joints_ratio': lost_joints_ratio,  # scalar
            'lost_humans_each_view': lost_humans_each_view,  # [C]
            'gt3d': gt3d,  # [N, j, 3]
            'proj2d': proj2d,  # [C, N, j, 2], may exceed view, not in use now.
            'vis_ratio_2d': vis_ratio_2d,  # scalar, ratio of target's visible joints in all views
            'vis_ratio_3d': vis_ratio_3d,  # scalar, ratio of target's visible joints, at least visible in 2 views
            'pred_bb_array': pred_bb_array,  # [C, N_max, 4], add zero boxes to make up N_max
            'pred3d': pred3d_info,  # [N_max, j, 3] ndarrays, zero arrays if not reconstructed
            'pred2d': pred2d_info,  # [C, N_max, j, 2], ndarrays,
            'pred2d_scores': pred2d_scores_info,  # [C, N_max, j, 1], used to determine visibility
            'cam_param_list': param_list,  # [C] x,y,z,pitch,yaw,roll,fov
            'org_img_list': img_list[0],  # [C]
            'num_cameras': self.num_cameras,
            'num_humans': self.num_humans,
            'gt_obs_dict': gt_obs_dict,
            'cam_human_distance': np.asarray(cam_human_distance),
        }

        if getattr(self, 'use_gt_observation', False):
            if getattr(self, 'gt_obs_with_visibility', False):
                # mask gt3d with visibility
                pred3d = gt3d.copy()  # [N, j, 3]
                vis_mask = np.sum(proj2d_vis, axis=0) >= 2  # [N, j], covered by 2 cam
                pred3d = pred3d * vis_mask.astype(np.float32)[..., None]
                gt_observation_with_visibility, _ = self.make_observation(
                    pred3d,
                    gt3d,
                    bb_from_mask_array,
                    param_list,
                    use_gt3d=False,
                    gt_noise_scale=self.gt_noise_scale,
                )
                return gt_observation_with_visibility, info

            if getattr(self, 'gt_noise_scale', 0) != 0:
                # return gt observation with noise, and gt info
                gt_observation_with_noise, _ = self.make_observation(
                    pred3d,
                    gt3d,
                    bb_from_mask_array,
                    param_list,
                    use_gt3d=True,
                    gt_noise_scale=self.gt_noise_scale,
                )
                return gt_observation_with_noise, info
            else:
                # return gt observation, and gt info
                return gt_observation, info
        else:
            start_time = time.time()
            pred_observation, pred_obs_dict = self.make_observation(
                pred3d, gt3d, bb_from_mask_array, param_list, use_gt3d=False
            )
            info['pred_obs_dict'] = pred_obs_dict
            elapsed_time = time.time() - start_time
            _process_time_dict['get_obs_process'] = elapsed_time

            info['_process_time_dict'] = _process_time_dict
            return pred_observation, info

    def make_bbox_feature(self, bb_from_mask_array, depth):
        """
        bb_from_mask_array: [C, N, 4]
        depth: [C, N, 1]
        """
        img = np.zeros(
            (self.num_cameras, self.resolution[1], self.resolution[0]), dtype=np.float32
        )  # [C, h, w]

        return

    def make_observation(
        self,
        pred3d,
        gt3d,
        bb_from_mask_array,
        param_list,
        use_gt3d=True,
        gt_noise_scale=0,
    ):
        # ================== new observation ===================================
        # observation shape: [C, 2+6+N_max*F_human]
        # [C, N_max*F_human]
        # F_human 15: [human_box3d:6, human_direction3d(xy):2, human_box2d:4,
        #   box_depth:1, box_IoU-target:1, occlude-target:1]
        obs_max_human = np.zeros(
            (
                self.num_cameras,
                self.max_num_humans,
                self.len_feature_human,
            )
        )  # [C, N_max, F_human]

        # [N, j, 3], pred3d
        # [C, N, j, 2], pred2d
        # [C, N, 4], bb_from_mask_array (x1, y1, w, h), some items may be zeros boxes

        # ================ 3d (shared across cameras) ===========================
        # =================== world ================
        # TODO: how to identify outliers from these points
        pred_mask = ~np.tile(np.any(pred3d, axis=-1, keepdims=True), (1, 1, 3))  # [N, j ,3]
        gt_mask = np.zeros_like(gt3d, dtype=np.int32)
        if use_gt3d:
            pose3d = gt3d
            mask = gt_mask
        else:
            pose3d = pred3d
            mask = pred_mask

        ma_pose3d = ma.array(pose3d, mask=mask)  # [N, j, 3]
        ma_obs_human_center = ma.median(ma_pose3d, axis=1)  # [N, 3]
        obs_human_center = ma.filled(ma_obs_human_center, 0)  # [N, 3]

        if use_gt3d and gt_noise_scale != 0:
            obs_human_center[:, :2] += np.random.normal(
                0, scale=gt_noise_scale, size=obs_human_center[:, :2].shape
            )
        obs_human_bb3d_scale = np.array(self.human_config['scale'], dtype=np.float32)  # [N]
        obs_human_bb3d_scale = np.tile(obs_human_bb3d_scale[..., None], (1, 3))  # [N, 3]

        # note: skeleton specific calculation !!!
        # shoulder: 0, 1, hip: 6, 7
        lshoulder = pose3d[:, 0, :]  # [N, 3]
        rshoulder = pose3d[:, 1, :]  # [N, 3]
        lhip = pose3d[:, 6, :]  # [N, 3]
        rhip = pose3d[:, 7, :]  # [N, 3]
        shoulder_dir = (
            (lshoulder - rshoulder)
            * np.any(lshoulder, axis=-1, keepdims=True)
            * np.any(rshoulder, axis=-1, keepdims=True)
        )  # [N, 3]
        hip_dir = (
            (lhip - rhip)
            * np.any(lhip, axis=-1, keepdims=True)
            * np.any(rhip, axis=-1, keepdims=True)
        )  # [N, 3]
        horizontal_dir = hip_dir + shoulder_dir  # [N, 3]
        vertical_dir = np.array((0, 0, 1), dtype=np.float32).reshape((1, 3))  # [1, 3]

        # calculate direction
        obs_human_dir3d = np.cross(horizontal_dir, vertical_dir)  # [N, 3]
        obs_human_dir3d[:, -1] = 0  # [N, 3] zero vectors for invalid direction
        np.divide(
            obs_human_dir3d,
            np.linalg.norm(obs_human_dir3d, axis=1, keepdims=True),
            out=obs_human_dir3d,
            where=np.linalg.norm(obs_human_dir3d, axis=1, keepdims=True) != 0,
        )  # [N, 2], !! this actually modifies the first 2 dims of obs_human_dir3d

        # calculate lost joint ratio
        num_lost_joints = np.sum(pred_mask[..., 0], axis=-1, keepdims=True)  # [N, 1]
        num_lost_joints_ratio = num_lost_joints / self.num_joints  # [N, 1]

        # ================ 3d flag ==========================
        # 3d valid rec flag, 1 for valid, 0 for invalid
        obs_human_flag_center = ~np.any(ma_obs_human_center.mask, axis=1, keepdims=True)  # [N, 1]
        obs_human_flag_center = obs_human_flag_center.astype(np.float32)

        # 3d valid dir flag, 1 for valid, 0 for invalid
        obs_human_flag_dir3d = np.any(obs_human_dir3d, axis=-1, keepdims=True)  # [N, 1]
        obs_human_flag_dir3d = obs_human_flag_dir3d.astype(np.float32)  # [N, 1]

        # ============== local ==================
        obs_human_local = []
        obs_human_world = np.concatenate(
            [
                obs_human_center,  # [N, 3]
                obs_human_dir3d,  # [N, 3]
                np.zeros((1, 3), dtype=np.float32),  # [1, 3]
            ],
            axis=0,
        )  # [N+N+1, 3]
        for CamModel in self.camera_model_list:
            obs_human_local.append(CamModel.world_to_cam(obs_human_world))
        obs_human_local = np.array(obs_human_local)  # [C, N+N+1, 3]

        obs_human_center_local = obs_human_local[:, : self.num_humans, :]  # [C, N, 3]
        obs_human_dir3d_local = obs_human_local[
            :, self.num_humans : self.num_humans * 2, :
        ]  # [C, N, 3]
        origin_local = obs_human_local[:, [-1], :]  # [C, 1, 3]
        obs_human_dir3d_local = obs_human_dir3d_local - origin_local  # [C, N, 3]

        # keep invalid values
        obs_human_center_local = (
            obs_human_center_local * obs_human_flag_center[None, ...]
        )  # [C, N, 3]
        obs_human_dir3d_local = obs_human_dir3d_local * obs_human_flag_dir3d[None, ...]  # [C, N, 3]
        np.divide(
            obs_human_dir3d_local,
            np.linalg.norm(obs_human_dir3d_local, axis=-1, keepdims=True),
            out=obs_human_dir3d_local,
            where=np.linalg.norm(obs_human_dir3d_local, axis=-1, keepdims=True) != 0,
        )  # [C, N, 3]

        obs_human_world3d = np.concatenate(
            (
                obs_human_bb3d_scale,  # [N, 3]
                num_lost_joints_ratio,  # [N, 1]
                obs_human_center,  # [N, 3]
                obs_human_dir3d,  # [N, 3]
            ),
            axis=-1,
        )  # [N, 10]

        obs_human_world3d = np.tile(obs_human_world3d, (self.num_cameras, 1, 1))  # [C, N, 10]
        obs_human_3d = np.concatenate(
            [
                obs_human_world3d,  # [C, N, 10]
                obs_human_center_local,  # [C, N, 3]
                obs_human_dir3d_local,  # [C, N, 3]
            ],
            axis=-1,
        )  # [C, N, 16]

        # ==================== 2d =======================
        # calculate bb2d
        obs_human_bb2d = bb_from_mask_array.copy().astype(np.float32)  # [C, N, 4], xc,yc,w,h
        obs_human_bb2d[:, :, :2] += obs_human_bb2d[:, :, 2:] / 2

        obs_human_bb2d_depth = np.zeros(
            (self.num_cameras, self.num_humans, 1), dtype=np.float32
        )  # [C, N, 1]
        for idx, CamModel in enumerate(self.camera_model_list):
            obs_human_bb2d_depth[idx] = CamModel.project_to_2d(obs_human_center, return_depth=True)[
                :, [-1]
            ]  # [N, 1]
        # zero out non-predicted humans
        obs_human_bb2d_depth = (
            obs_human_bb2d_depth * obs_human_center.any(axis=-1, keepdims=True)[None, ...]
        )
        obs_human_bb2d_depth = obs_human_bb2d_depth * obs_human_bb2d.any(axis=-1, keepdims=True)

        # calculate IoU with target
        bb_temp = obs_human_bb2d.copy()  # [C, N, 4], x1,y1,x2,y2
        bb_temp[:, :, :2] -= bb_temp[:, :, 2:] / 2
        bb_temp[:, :, 2:] = bb_temp[:, :, :2] + bb_temp[:, :, 2:]

        left_up = np.minimum(bb_temp[:, [0], :2], bb_temp[:, :, :2])  # [C, N, 2]
        right_bottom = np.maximum(bb_temp[:, [0], 2:], bb_temp[:, :, 2:])  # [C, N, 2]
        union_len = right_bottom - left_up  # [C, N, 2]

        intersection_len = obs_human_bb2d[:, [0], 2:] + obs_human_bb2d[:, :, 2:] - union_len
        intersection_len = np.maximum(0, intersection_len)  # [C, N, 2]

        intersection_area = np.prod(intersection_len, axis=-1)  # [C, N]
        target_area = np.prod(obs_human_bb2d[:, [0], 2:], axis=-1)  # [C, 1]

        obs_human_bb2d_IoT = np.zeros_like(intersection_area, dtype=np.float32)  # [C, N]
        np.divide(
            intersection_area,
            target_area,
            out=obs_human_bb2d_IoT,
            where=target_area != 0,
        )

        obs_human_bb2d_IoT = obs_human_bb2d_IoT[..., None]  # [C, N, 1]
        obs_human_2d = np.concatenate(
            (
                obs_human_bb2d,  # [C, N, 4]
                obs_human_bb2d_depth,  # [C, N, 1]
                obs_human_bb2d_IoT,  # [C, N, 1]
            ),
            axis=-1,
        )  # [C, N, 6]

        # ============== 2d flag ==============
        # calculate occ-target flag, 1: occ, 0: not occ
        obs_human_flag_occ_target = obs_human_bb2d_IoT > 0  # [C, N, 1]
        obs_human_flag_occ_target = np.logical_and(
            obs_human_flag_occ_target,
            obs_human_bb2d_depth[:, [0], :] > obs_human_bb2d_depth,
        ).astype(
            np.float32
        )  # [C, N, 1]

        # 2d valid box flag, 1 for valid, 0 for invalid
        obs_human_flag_bb2d = np.any(obs_human_bb2d, axis=-1, keepdims=True)  # [C, N, 1]
        obs_human_flag_bb2d = obs_human_flag_bb2d.astype(np.float32)

        # 2d valid box depth flag, 1 for valid, 0 for invalid
        obs_human_flag_bb2d_depth = (
            obs_human_flag_bb2d * obs_human_flag_center[None, ...]
        )  # [C, N, 1]

        # 2d valid IoU flag
        # bb exceeds boundary or not detected, same as obs_human_flag_bb2d, not in use
        # obs_human_flag_bb2d_IoU = obs_human_flag_bb2d

        # ============== Integrate flags =======================
        obs_human_flag_2d = np.concatenate(
            (
                obs_human_flag_occ_target,  # [C, N, 1]
                obs_human_flag_bb2d,  # [C, N, 1]
                obs_human_flag_bb2d_depth,  # [C, N, 1]
            ),
            axis=-1,
        )  # [C, N, 3]
        obs_human_flag_3d = np.concatenate(
            (
                obs_human_flag_center,  # [N, 1]
                obs_human_flag_dir3d,  # [N, 1]
            ),
            axis=-1,
        )  # [N, 2]
        obs_human_flag_3d = np.tile(obs_human_flag_3d, (self.num_cameras, 1, 1))  # [C, N, 2]

        obs_human_id = (
            np.arange(self.num_humans, dtype=np.float32).reshape((1, -1, 1)) + 1
        )  # [1, N, 1]
        obs_human_id = np.tile(obs_human_id, (self.num_cameras, 1, 1))  # [C, N, 1]

        obs_human_flag = np.concatenate(
            (
                obs_human_id,  # [1]
                obs_human_flag_3d,  # [2]
                obs_human_flag_2d,  # [3]
            ),
            axis=-1,
        )  # [C, N, 6]

        # =============== Camera Param =====================
        param_array = np.array(param_list)  # [C, 7], x,y,z,pitch,yaw,roll,fov
        obs_camera_xyz = param_array[:, :3]  # [C, 3]
        obs_camera_py = param_array[:, 3:5]  # [C, 2]
        obs_camera_py = np.cos(obs_camera_py / 180.0 * np.pi)  # [C, 2]

        obs_camera_rot = []
        for CamModel in self.camera_model_list:
            obs_camera_rot.append(CamModel.cam_to_world(np.array([[0, 0, CamModel.f]])))
        obs_camera_rot = np.concatenate(
            obs_camera_rot, axis=0
        )  # [C, 3], unit optical axes in world coordinates
        obs_camera_rot = obs_camera_rot - obs_camera_xyz  # [C, 3]
        np.divide(
            obs_camera_rot,
            np.linalg.norm(obs_camera_rot, axis=1, keepdims=True),
            out=obs_camera_rot,
            where=np.linalg.norm(obs_camera_rot, axis=1, keepdims=True) != 0,
        )
        obs_camera_id = np.arange(self.num_cameras, dtype=np.float32)[..., None] + 1  # [C, 1]
        obs_camera = np.concatenate(
            [obs_camera_id, obs_camera_xyz, obs_camera_rot, obs_camera_py], axis=-1
        )  # [C, 9]

        # calculate relative representation for each cam
        obs_camera_world = np.concatenate(
            [
                obs_camera_xyz,  # [C, 3]
                obs_camera_rot + obs_camera_xyz,  # [C, 3]
            ],
            axis=0,
        )  # [2C, 3]

        obs_camera_local = []
        for CamModel in self.camera_model_list:
            obs_camera_local.append(CamModel.world_to_cam(obs_camera_world))
        obs_camera_local = np.array(obs_camera_local)  # [C, 2C, 3]

        obs_camera_xyz_local = obs_camera_local[:, : self.num_cameras, :]  # [C, C, 3]
        obs_camera_dir3d_local = obs_camera_local[:, self.num_cameras :, :]  # [C, C, 3]

        obs_camera_relative_matrix = np.concatenate(
            [
                obs_camera_xyz_local,  # [C, C, 3]
                obs_camera_dir3d_local,  # [C, C, 3]
            ],
            axis=-1,
        )  # [C, C, 6]

        obs_camera_relative_matrix[np.diag_indices(self.num_cameras)] = [
            0,
            0,
            0,
            0,
            0,
            1,
        ]

        # =============== env ===============================
        # num_cameras + num_humans(one-hot)
        obs_env = np.zeros((1 + self.max_num_humans), dtype=np.float32)  # [8]
        obs_env[0] = self.num_cameras
        obs_env[1 : 1 + self.num_humans] = 1
        obs_env = np.tile(obs_env, (self.num_cameras, 1))  # [C, 8]

        # =============== concatenate =======================
        obs_human = np.concatenate(
            (
                obs_human_3d,  # [C, N, 16]
                obs_human_2d,  # [C, N, 6]
                obs_human_flag,  # [C, N, 6]
            ),
            axis=-1,
        )  # [C, N, 16+6+6=28]

        obs_max_human[:, : self.num_humans, :] = obs_human

        observation = np.concatenate(
            [
                obs_env,  # [C, 8]
                obs_camera,  # [C, 9]
                obs_max_human.reshape((self.num_cameras, -1)),  # [C, N_max*F]
            ],
            axis=-1,
        )  # [C, 8+9+N_max*F]

        obs_dict = {
            # [C, C, 6], loc+dir, diagonal is [0, 0, 0, 0, 0, 1]
            'obs_camera_relative_matrix': obs_camera_relative_matrix,
            'observation': observation,  # [C, 8+9+N_max*F],
            'obs_target_bb2d_IoT': obs_human_bb2d_IoT,  # [C, N, 1]
            'obs_target_occ_flag': obs_human_flag_occ_target,  # [C, N, 1]
        }

        return observation, obs_dict

    def render(self, mode='cv2', timestap=None):
        if mode == 'cv2':
            for info in self.info_trajectory:
                org_img_list = info['org_img_list']  # [C], [h, w, 4] RGBA, 0-255
                viz_record = {
                    'bb': info['pred_body_bb_list'],
                    'lhand_bb': info['pred_lhand_bb_list'],
                    'rhand_bb': info['pred_rhand_bb_list'],
                    'pose2d': info['pred2d'],
                }
                viz_img_list = [org_img[..., :-1] for org_img in org_img_list]
                viz_imgs = visualization2d_wrapper(viz_img_list, viz_record)
                img_h = np.concatenate(viz_imgs[:2], axis=1)  # [h, C*w, 3]
                img_v = np.concatenate(viz_imgs[2:4], axis=1)  # [h, C*w, 3]
                img_show = np.vstack((img_h, img_v))  # [2*h, 2*w, 3]
                img_show = img_show[:, :, ::-1]
                cv2.imshow('2D views', img_show)
                cv2.waitKey(100)
        elif mode == 'offline-save':
            # run on server, save locally
            # support multi-person & single-person
            if len(self.info_trajectory) == 0:
                return

            if not self.qt_render_inited:
                # initialize
                self.camera_model_plot_list = copy.deepcopy(self.camera_model_list)
                # determine shape
                info = self.info_trajectory[0]
                shape_dict = {
                    'gt3d': info['gt3d'].shape,  # [N, j, 3]
                    'pred3d': None,  # None for variable len
                    'camera': (self.num_cameras, 5, 3),
                    # 'reward_3d': (4,),  # total, body, lhand, rhand
                }
                import socket

                hostname = socket.gethostname()
                common_dir = os.path.join(f'render_data/{hostname}', timestap)
                self.data_saver = ZarrVariableBufferSaver(
                    shape_dict,
                    save_path=os.path.join(common_dir, '3d.zarr'),
                    var_maxlen=self.max_num_humans,
                    capacity=100000,
                    map=self.config.ENV.MAP_NAME,
                )
                self.video_saver = VideoSaver(self.num_cameras, os.path.join(common_dir, '2d'))
                self.video_saver.init()
                self.qt_render_inited = True

            for idx, info in enumerate(self.info_trajectory):
                org_img_list = info['org_img_list']  # [C], [h, w, 4] RGBA, 0-255

                # scale bb

                viz_record = {
                    'bb': info['pred_bb_array'],  # [C, N_max, 4]
                    'pose2d': info['pred2d'],  # [C, N_max, j, 2], N_max for color consistency
                }
                viz_img_list = [org_img[..., :-1] for org_img in org_img_list]
                viz_imgs = visualization2d_wrapper(viz_img_list, viz_record)

                # save viz_images to videos
                self.video_saver.append(viz_imgs)

                # collect cameras
                corners_world_list = []
                for (
                    CamModel,
                    camera_param,
                ) in zip(self.camera_model_plot_list, info['cam_param_list']):
                    # CamModel = camera.CameraPose(*camera_param[:3], *camera_param[3:6], WIDTH, HEIGHT, camera_param[-1])
                    CamModel.update_camera_parameters_list(camera_param)

                    corners = np.array(
                        [
                            [
                                -self.resolution[0] / 2,
                                -self.resolution[1] / 2,
                                CamModel.f,
                            ],
                            [
                                self.resolution[0] / 2,
                                -self.resolution[1] / 2,
                                CamModel.f,
                            ],
                            [
                                -self.resolution[0] / 2,
                                self.resolution[1] / 2,
                                CamModel.f,
                            ],
                            [
                                self.resolution[0] / 2,
                                self.resolution[1] / 2,
                                CamModel.f,
                            ],
                            [0, 0, 0],
                        ]
                    )  # end point of the optical ax of the camera + optical center
                    corners[:-1, ...] /= 10.0
                    corners_world_list.append(CamModel.cam_to_world(corners))  # [5, 3]

                pred3d_zarr = zarr.empty(self.max_num_humans, dtype='array:f4')
                assert (
                    info['pred3d'].shape[0] <= self.max_num_humans
                ), f'prune pred.shape[0] to be <= {self.max_num_humans}'
                for idx, pred3d in enumerate(info['pred3d']):  # [N_max, j, 3]
                    if np.any(pred3d):  # not all zeros
                        pred3d_zarr[idx] = pred3d.astype(np.float32).reshape((-1,))

                save_dict = {
                    'gt3d': info['gt3d'],  # [N, J, 3]
                    'pred3d': pred3d_zarr,  # [N_max, J*3], flattened Zarr
                    'camera': np.array(corners_world_list),  # [C, 5, 3]
                }
                self.data_saver.append(save_dict)

    def close(self):
        if hasattr(self, 'env_interaction'):
            self.env_interaction.close()

        if hasattr(self, 'binary'):
            self.binary.close()

        if hasattr(self, 'data_saver'):
            self.data_saver.close()

        if hasattr(self, 'video_saver'):
            self.video_saver.close()

        if hasattr(self, 'win'):
            print('Waiting for GUI process to join...')
            self.win.join()
            print('GUI Exit!   Yes!')

        self.closed = True

    def __del__(self):
        """
        Automatic close resources
        """
        if self.closed:
            return

        self.close()

    def pid(self, coef, var, target):
        return coef * (target - var)

    def cal_control_policy_from_assignment_2d(self, action, state_dict):
        """
        Calculate control param for each camera
        All used bounding boxes may exceed the img boundary
        action: [C, N_max], C camera, N_max persons
        """

        action = np.asarray(action, dtype=bool)  # [C, N_max]
        bb_union_list = []  # [C] xmin, ymin, xmax, ymax
        for view_assignment, view_pred_bb in zip(action, state_dict['pred_bb_array']):
            # view_assignment [N_max], view_pred_bb [N_max, 4]
            if not np.any(view_assignment):
                # no assignment, then no rotation action for this view
                bb_union_list.append(np.array([0, 0, self.resolution[0], self.resolution[1]]))
                continue
            selected_bb = view_pred_bb[view_assignment, :]  # [S, 4]
            # filter out zero width/height boxes
            selected_bb = selected_bb[np.all(selected_bb[:, 2:], axis=-1), :]  # [S, 4]
            if len(selected_bb) > 0:
                selected_bb = np.concatenate(
                    (
                        selected_bb[:, :2],
                        selected_bb[:, [0]] + selected_bb[:, [2]],
                        selected_bb[:, [1]] + selected_bb[:, [3]],
                    ),
                    axis=-1,
                )  # [s, 4], xmin, ymin, xmax, ymax
                bb_union_list.append(
                    [
                        np.amin(selected_bb[:, 0]),
                        np.amin(selected_bb[:, 1]),
                        np.amax(selected_bb[:, 2]),
                        np.amax(selected_bb[:, 3]),
                    ]
                )
            else:
                bb_union_list.append(np.array([0, 0, self.resolution[0], self.resolution[1]]))

        # calculate yaw, pitch control policy
        # union box locate at the center of the image.
        img_center = (self.resolution[0] / 2, self.resolution[1] / 2)  # [w, h]

        yaw_policy = []  # [C]
        pitch_policy = []  # [C]

        for bb_union, CamModel in zip(bb_union_list, self.camera_model_list):
            target_center = (
                (bb_union[0] + bb_union[2]) / 2,
                (bb_union[1] + bb_union[3]) / 2,
            )
            delta_yaw = math.atan2(target_center[0] - img_center[0], CamModel.f) / np.pi * 180
            delta_pitch = math.atan2(target_center[1] - img_center[1], CamModel.f) / np.pi * 180
            yaw_policy.append(-self.pid(self.yaw_coef, delta_yaw, 0))
            pitch_policy.append(self.pid(self.pitch_coef, delta_pitch, 0))

        # filter policy

        assert len(yaw_policy) == self.num_cameras
        assert len(pitch_policy) == self.num_cameras
        yaw_policy = np.clip(yaw_policy, -30, 30)
        pitch_policy = np.clip(pitch_policy, -30, 30)

        return yaw_policy, pitch_policy

    def cal_control_policy_from_assignment_3d(self, action, state_dict, use_gt3d=False):
        """
        Call this func before cam model updated
        Calculate control param for each camera, according to current observation.
        All used bounding boxes may exceed the img boundary
        action: [C, N_max], C camera, N_max persons
        """
        action = np.asarray(action, dtype=bool)  # [C, N_max]

        if use_gt3d:
            pred3d = state_dict['gt3d']  # [N_max, j, 3] ndarrays, zero arrays if not reconstructed
            if len(pred3d) < self.max_num_humans:
                fill_num = self.max_num_humans - self.controller.num_humans
                pred3d = np.concatenate([pred3d, np.zeros((fill_num, pred3d.shape[1], 3))], axis=0)
        else:
            pred3d = state_dict[
                'pred3d'
            ]  # [N_max, j, 3] ndarrays, zero arrays if not reconstructed

        low_area_bounds = self.env_config['area'].min(axis=0)
        high_area_bounds = self.env_config['area'].max(axis=0)
        outlier_thresh = (high_area_bounds - low_area_bounds).max() / 2.0

        target_center = []  # [C], camera optical axis should direct to this point
        take_actions = np.zeros(
            (self.num_cameras,)
        )  # [C], indicate if this camera should take no action at this frame
        for cam_idx, view_assignment in enumerate(action):
            assigned_pred3d = pred3d[view_assignment, ...]  # [S, j, 3]
            if len(assigned_pred3d) == 0:
                # no person is selected as the target for this camera
                # keep still
                target_center.append(np.zeros((3,)))
                continue

            # filter out zero pred3d
            seletected_pred3d = assigned_pred3d[
                np.any(assigned_pred3d, axis=(1, 2)), ...
            ]  # [S, j, 3]
            if len(seletected_pred3d) == 0:
                # all selected people are not successfully reconstructed
                # keep still
                target_center.append(np.zeros((3,)))
                continue

            # filter out zero pred points and outlier points
            seletected_pred3d = seletected_pred3d[
                np.any(seletected_pred3d, axis=-1), ...
            ]  # [Se, 3]
            seletected_pred3d = seletected_pred3d[
                np.all(
                    np.abs(seletected_pred3d - self.map_config['map_center']) < outlier_thresh,
                    axis=-1,
                ),
                ...,
            ]  # [Se, 3]
            if len(seletected_pred3d) == 0:
                target_center.append(np.zeros((3,)))
                continue

            target_center.append(np.mean(seletected_pred3d, axis=0))  # [3]
            take_actions[cam_idx] = 1

        assert len(target_center) == self.num_cameras

        target_center_in_camframe = []
        for CamModel, target in zip(self.camera_model_list, target_center):
            target_center_in_camframe.append(CamModel.world_to_cam(target))

        target_center_in_camframe = np.concatenate(target_center_in_camframe, axis=0)  # [C, 3]
        expected_direction_in_camframe = target_center_in_camframe

        target_yaw = np.arctan2(
            expected_direction_in_camframe[:, 0], expected_direction_in_camframe[:, -1]
        )  # [C]
        target_pitch = -np.arctan2(
            expected_direction_in_camframe[:, 1],
            np.linalg.norm(expected_direction_in_camframe[:, [0, 2]], axis=-1),
        )  # [C]

        target_yaw = target_yaw / np.pi * 180
        target_pitch = target_pitch / np.pi * 180

        yaw_policy = self.pid(self.yaw_coef_3d, 0, target_yaw)
        pitch_policy = self.pid(self.pitch_coef_3d, 0, target_pitch)

        assert len(yaw_policy) == self.num_cameras
        assert len(pitch_policy) == self.num_cameras
        yaw_policy = np.clip(yaw_policy, -30, 30)
        pitch_policy = np.clip(pitch_policy, -30, 30)

        # mask out no action cameras
        yaw_policy = yaw_policy * take_actions
        pitch_policy = pitch_policy * take_actions

        return yaw_policy, pitch_policy
