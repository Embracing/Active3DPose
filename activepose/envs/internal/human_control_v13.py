import os
import random

import numpy as np
from scipy.spatial import distance as scipy_distance

from activepose.utils.file import read_anim_names

from .utils import PointSampler, TrajectorySampler


class Human:
    def __init__(
        self,
        env_interaction,
        id,
        point_sampler=None,
        action_setting='walk',
        z=82.15,
        walk_speed=125,
        rot_speed=200,
        dilation=1.0,
    ):
        """
        Assume humans are already created in the env
        env_interaction: which env this human is in. used to set anim
        area: human is bounded with area, walk in the valid area
        action_setting: 'walk', 'anim', 'mixed', 'fixed'
        init_z: smallest z for human on the floor.

        some variables can be customized
            self.action_setting: mixed
            self.walk_speed
            self.rot_speed

        Usage:
            reset_action() --> step_action() -->

        """
        self.env_interaction = env_interaction
        self.id = id

        self.action_setting = action_setting
        if self.action_setting != 'mixed':
            self.action_mode = action_setting
        self.walk_speed = walk_speed
        self.rot_speed = rot_speed
        self.z = z  # floor z for human
        self.dilation = dilation

        # assume vertices are centered around the orogin.
        # None, if wrapper by Multi-human Controller
        self.point_sampler = point_sampler

        if self.action_mode == 'fixed':
            self.env_interaction.set_dilation_async(f'human{self.id}', 0)
            # self.env_interaction.change_to_anim_mode('human{}'.format(self.id))


class MultiHumanController:
    def __init__(
        self,
        env_interaction,
        human_config,
        env_config,
        action_setting='walk',
        random_init_locrot=False,
        changable_mesh=True,
    ):
        """
        Contorl Human in batch mode.
        1. Create Human
        2. Initialize location, rotation, mesh, mask_color
        env_interaction:
            which env this human is in. used to set anim
        area:
             human is bounded with area, walk in the valid area
        action_setting:
            'walk', 'anim', 'mixed', 'fixed' for all humans
            'list' to provide different settings for individuals.
        init_z:
            smallest z for human on the floor.

        Some variables can be customized
            human.action_setting: mixed
            human.walk_speed
            human.rot_speed
        Usage:
            Call reset_action_all(), step_action_all(),
            freeze_walk_all(), restore_walk_all()
            to control human actions(walking, animating)
        """
        self.env_interaction = env_interaction
        self.human_config = human_config
        self.env_config = env_config

        # env_version specific variable, for env_v1.3
        self.floor_z_of_human = 82.15
        self.default_walk_speed = 100
        self.default_rotation_speed = 360

        # reset human
        self.human_id_list = self.human_config['human_id_list']
        self.num_humans = len(self.human_id_list)

        cmd_list = []

        # destroy extra humans, total: 10, human1 - human10
        # can run multiple times
        for id in range(1, 11):
            if id not in self.human_id_list:
                cmd_list.append(f'vbp human{id} destroy')

        if isinstance(action_setting, list):
            assert len(action_setting) == self.num_humans
            self.action_setting_list = action_setting
        else:
            self.action_setting_list = [action_setting] * self.num_humans

        cmd_list += [
            'vset /camera/0/location 0 0 1000'
        ]  # adjust camera 0 init location to prevent collision

        low_area_bounds = self.env_config['area'].min(axis=0)
        high_area_bounds = self.env_config['area'].max(axis=0)
        area_center = (low_area_bounds + high_area_bounds) / 2.0

        self.area_radius = (high_area_bounds - low_area_bounds).max() / 2.0
        self.dilation_list = [
            random.randrange(*self.human_config['walk_speed_range']) / self.default_walk_speed
            for _ in range(self.num_humans)
        ]

        self.random_point_sampler = PointSampler(
            shape=self.env_config['area_shape'],
            points=area_center[None, :] + (self.env_config['area'] - area_center[None, :]) * 0.9,
        )

        if random_init_locrot:
            init_loc_list = []
            while True:
                new_xy = self.random_point_sampler.sample()  # [2], xy
                for old_xy in init_loc_list:
                    if np.linalg.norm(new_xy - old_xy) < 60:
                        break
                else:
                    init_loc_list.append(new_xy)
                    if len(init_loc_list) == self.num_humans:
                        break
            init_loc_list = [x.tolist() + [self.floor_z_of_human] for x in init_loc_list]
            init_rot_list = [[0, random.randrange(0, 360), 0] for _ in range(self.num_humans)]
        else:
            init_loc_list = self.human_config['human_location_list']
            init_rot_list = self.human_config['human_rotation_list']

        for id, loc, rot, mesh_name, mask_color, scale, dilation in zip(
            self.human_id_list,
            init_loc_list,
            init_rot_list,
            random.choices(self.env_config['human_mesh_list'], k=self.num_humans),
            self.human_config['mask_color'],
            self.human_config['scale'],
            self.dilation_list,
        ):
            cmd_list.extend(
                [
                    'vset /object/human{id}/location {} {} {}'.format(*loc, id=id),
                    'vset /object/human{id}/rotation {} {} {}'.format(*rot, id=id),
                    'vset /object/human{id}/color {} {} {}'.format(*mask_color, id=id),
                    f'vset /object/human{id}/scale {scale} {scale} {scale}',
                    # 'vset /human/human{id}/mesh /Game/human_mesh/{mesh_name}'.format(mesh_name=mesh_name, id=id),
                    f'vbp human{id} set_dilation {dilation}',
                    f'vbp human{id} set_walk_radius {self.area_radius}',  # start walk by calling this command
                ]
            )

            # this param is map-specific
            if changable_mesh:
                cmd_list.append(
                    'vset /human/human{id}/mesh /Game/human_mesh/{mesh_name}'.format(
                        mesh_name=mesh_name, id=id
                    )
                )

        self.env_interaction.request(cmd_list)
        self.human_list = [
            Human(
                self.env_interaction,
                id=id,
                point_sampler=TrajectorySampler(points=self.env_config['target_trajectory'])
                if 'target_trajectory' in self.env_config
                else None,
                action_setting=action_setting,
                z=self.floor_z_of_human,
                walk_speed=dilation * self.default_walk_speed,
                rot_speed=dilation * self.default_rotation_speed,
                dilation=dilation,
            )
            for id, action_setting, dilation in zip(
                self.human_id_list, self.action_setting_list, self.dilation_list
            )
        ]

    def freeze_walk_all(self):
        walking_human = [human for human in self.human_list if human.action_mode == 'walk']
        if len(walking_human) == 0:
            return

        cmd_list = [f'vbp human{human.id} set_dilation 0' for human in walking_human]
        self.env_interaction.request(cmd_list)

    def restore_walk_all(self):
        walking_human = [human for human in self.human_list if human.action_mode == 'walk']
        if len(walking_human) == 0:
            return

        cmd_list = [f'vbp human{human.id} set_dilation {human.dilation}' for human in walking_human]
        self.env_interaction.request(cmd_list)

    def reset_action_all(self):
        return

    def step_action_all(self):
        return

    def set_walk_speed_batch(self, speed_list, human_list):
        """
        Work immediately
        """
        assert len(speed_list) == len(human_list)

        cmd_list = [
            f'vbp human{human.id} set_speed {speed}' for human, speed in zip(human_list, speed_list)
        ]
        self.env_interaction.request(cmd_list)

        for human, speed in zip(human_list, speed_list):
            human.walk_speed = speed

    def set_rot_speed_batch(self, speed_list, human_list):
        """
        Work immediately
        """
        assert len(speed_list) == len(human_list)

        cmd_list = [
            f'vbp human{human.id} set_rot_speed {speed}'
            for human, speed in zip(human_list, speed_list)
        ]
        self.env_interaction.request(cmd_list)

        for human, speed in zip(human_list, speed_list):
            human.rot_speed = speed

    def random_set_human_loc_rot_all(self):
        cmd_list = []

        location_list = []
        while True:
            new_xy = self.random_point_sampler.sample()  # [2], xy
            for old_xy in location_list:
                if np.linalg.norm(new_xy - old_xy) < 60:
                    break
            else:
                location_list.append(new_xy)
                if len(location_list) == self.num_humans:
                    break
        location_list = [x.tolist() + [self.floor_z_of_human] for x in location_list]
        rotation_list = [[0, random.randrange(0, 360), 0] for _ in range(self.num_humans)]

        for id, loc, rot in zip(self.human_id_list, location_list, rotation_list):
            cmd_list.extend(
                [
                    'vset /object/human{id}/location {} {} {}'.format(*loc, id=id),
                    'vset /object/human{id}/rotation {} {} {}'.format(*rot, id=id),
                ]
            )

        self.env_interaction.request(cmd_list)

    def set_human_default_loc_rot_all(self):
        cmd_list = []

        for id, loc, rot in zip(
            self.human_id_list,
            self.human_config['human_location_list'],
            self.human_config['human_rotation_list'],
        ):
            cmd_list.extend(
                [
                    'vset /object/human{id}/location {} {} {}'.format(*loc, id=id),
                    'vset /object/human{id}/rotation {} {} {}'.format(*rot, id=id),
                ]
            )

        self.env_interaction.request(cmd_list)

    def reset_trajectory_samplers_all(self):
        for human in self.human_list:
            if human.point_sampler is not None and isinstance(
                human.point_sampler, TrajectorySampler
            ):
                human.point_sampler.reset()

    # =============== func only for walk (no mixed, no anim) ========================
    def reset_walk_all_async(self):
        return

    def step_walk_all_async(self):
        return

    def set_dilation_batch_async(self, dilation_list, human_list):
        assert len(dilation_list) == len(human_list)

        cmd_list = [
            f'vbp human{human.id} set_dilation {dilation}'
            for human, dilation in zip(human_list, dilation_list)
        ]

        self.env_interaction.request_async(cmd_list)

    def random_set_human_loc_rot_all_async(self):
        cmd_list = []

        location_list = []
        while True:
            new_xy = self.random_point_sampler.sample()  # [2], xy
            for old_xy in location_list:
                if np.linalg.norm(new_xy - old_xy) < 60:
                    break
            else:
                location_list.append(new_xy)
                if len(location_list) == self.num_humans:
                    break
        location_list = [x.tolist() + [self.floor_z_of_human] for x in location_list]
        rotation_list = [[0, random.randrange(0, 360), 0] for _ in range(self.num_humans)]

        for id, loc, rot in zip(self.human_id_list, location_list, rotation_list):
            cmd_list.extend(
                [
                    'vset /object/human{id}/location {} {} {}'.format(*loc, id=id),
                    'vset /object/human{id}/rotation {} {} {}'.format(*rot, id=id),
                ]
            )

        self.env_interaction.request_async(cmd_list)
