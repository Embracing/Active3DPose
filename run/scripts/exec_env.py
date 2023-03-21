import argparse
import os
import random
import subprocess
import sys
import time
import traceback

import gym
import numpy as np
import torch
import torch.backends.cudnn as cudnn
from tqdm import tqdm

from activepose import envs
from activepose.config import config, update_config
from activepose.control.wrappers import *
from activepose.control.wrappers import (
    FlattenAction,
    FlattenObservation,
    NormalizeObservation,
    SingleAgentRewardLogger,
    SingleEvaluationStep,
    TruncateObservation,
)
from activepose.envs.wrappers import *
from activepose.pose.utils2d.pose.core import initialize_streaming_pose_estimator
from activepose.utils.file import create_logger


def parse_args():
    parser = argparse.ArgumentParser(description='debug_env')

    # general
    parser.add_argument(
        '--cfg',
        help='experiment configure file name',
        type=str,
        default=os.path.join('configs', 'w32_256x192_17j_coco.yaml'),
    )
    parser.add_argument('--num-humans', type=int, default=7)
    parser.add_argument('--max-num-humans', type=int)
    parser.add_argument('--map-name', type=str, default='Blank')
    parser.add_argument('--human-model', type=str, default='Default')
    parser.add_argument('--env-name', type=str)

    args = parser.parse_args()

    return args


if __name__ == '__main__':
    args = parse_args()

    # for reproducibility
    np.random.seed(1234)
    random.seed(1234)
    torch.manual_seed(1234)

    # cudnn related setting
    cudnn.benchmark = config.CUDNN.BENCHMARK
    cudnn.deterministic = config.CUDNN.DETERMINISTIC
    cudnn.enabled = config.CUDNN.ENABLED

    # make sure resource is closed
    try:
        # =========== pose 2d estimator ==============================
        logger, final_output_dir, timestap = create_logger(config, 'online')
        # logger.info(pprint.pformat(config))

        # ========== Create Env ======================================
        render_type = 'offline-save'
        env = gym.make(
            'base-v0',
            args=args,
            air_wall_outer=True,
            air_wall_inner={
                'use': False,
                'args': {
                    'lower_bound': [-200, -200, 0],
                    'higher_bound': [200, 200, 1000],
                },
            },
            gt_observation={
                'use': False,
                'args': {'gt_noise_scale': 0, 'affected_by_visibility': False},
            },
            place_cam={
                'use': False,
                'args': {'num_place_cameras': 2, 'place_on_slider': False},
            },
            fix_human={
                'use': False,
                'args': {'random_init_states': False, 'index_list': [0]},
            },
            movable_cam={'use': False, 'args': {'num_movable_cameras': 2}},
            rule_based_rot={
                'use': False,
                'args': {'use_gt3d': True, 'num_rule_cameras': 2, 'index_list': [0]},
            },
            slider={'use': False},
            reach_target_done={'use': False},
            scale_human={'use': False, 'args': {'scale': 2, 'index_list': [1, 2]}},
            ego_action={'use': False},
            rot_limit={
                'use': False,
                'args': {
                    'pitch_low': -85,
                    'pitch_high': 85,
                    'yaw_low': -360,
                    'yaw_high': 360,
                },
            },
            rand_reset_human={'use': False, 'args': {'random_init_loc_rot': True}},
            shuffle_cam_id={'use': False},
            partial_triangulation={'use': True},
            imitate_rot={'use': False, 'args': {'index_list': [0]}},
        )
        # call this func when you want to change default env, human config
        # env.load_config(env_name=None, num_humans=3)
        # env = Slider(env)
        env = SingleEvaluationStep(env)
        # env = NormalizeObservation(env)
        # env = FlattenAction(env)
        # env = FlattenObservation(env)
        env = TruncateObservation(env)
        # env = Aux_Rewards(env, centering=False, iot=True, distance=False)

        np.set_printoptions(precision=2, formatter={'float_kind': '{:f}'.format})

        obs = env.reset()
        env.render(mode=render_type, timestap=timestap)
        action = env.action_space.sample()
        action = np.ones_like(action)
        # action[0, 0] = 0
        # action[1, 1] = 0
        time_count = []
        mpjpe_list = []
        for idx in range(10):
            start = time.time()
            print(f'====== iter {idx} ======')
            observation, reward, done, info_trajectory = env.step(action)  # take a random action
            time_count.append(time.time() - start)
            # print(observation)
            # print(info_trajectory)
            print('reward:', reward)
            if not isinstance(info_trajectory, list):
                info_trajectory = [info_trajectory]
            for info in info_trajectory:
                if 'reward_3d' in info:
                    print('reward_3d: {}'.format(info['reward_3d']))
                if 'reward_vis' in info:
                    print('reward_vis:', info['reward_vis'])
                if 'reward_camera_state' in info:
                    print('reward_camera_state:', info['reward_camera_state'])
                if 'reward_centering' in info:
                    print('reward_centering:', info['reward_centering'])
                if 'reward_iot' in info:
                    print('reward_iot:', info['reward_iot'])
                if 'reward_distance' in info:
                    print('reward_distance:', info['reward_distance'])
                if 'mpjpe_3d' in info:
                    print(f"mpjpe_3d: {info['mpjpe_3d']}")
                    mpjpe_list.append(info['mpjpe_3d'])
                if 'vis_ratio_2d' in info:
                    print('vis_ratio_2d')
                    print(info['vis_ratio_2d'])
                if 'vis_ratio_3d' in info:
                    print('vis_ratio_3d')
                    print(info['vis_ratio_3d'])
                if 'proj2d_vis' in info:
                    print('proj2d_visibility')
                    print(info['proj2d_vis'])
                # if 'shapley_reward_dict' in info:
                #     print(f"shapley_reward_dict: {info['shapley_reward_dict']}")
                # if 'shapley_mpjpe_dict' in info:
                #     print(f"shapley_mpjpe_dict: {info['shapley_mpjpe_dict']}")
                # if 'shapley_pck20_dict' in info:
                #     print(f"shapley_pck20_dict: {info['shapley_pck20_dict']}")
                # if 'imit_pitch_action' in info:
                #     print(f"imit_pitch_action: {info['imit_pitch_action']}")
                # if 'imit_yaw_action' in info:
                #     print(f"imit_yaw_action: {info['imit_yaw_action']}")
            # if idx % 5 == 0 and idx != 0:
            #     env.reset()
            #     print('reset !!!!!')
            # print('=> Debug -- step() time: ', time.time() - start_time)
            env.render(mode=render_type, timestap=timestap)
        print('======= Mean MPJPE ========')
        print(np.mean(mpjpe_list))
        print('======= Success Rate @ 20 ========')
        print(np.mean(np.array(mpjpe_list) <= 20))

        print('action')
        print(action)
        print('===== FPS =====')
        print(np.mean(time_count))
    except Exception as e:
        traceback.print_exc()
        # print(e.message)
    finally:
        try:
            env
        except NameError:
            print('not find object: env')
        else:
            env.close()
            import socket

            hostname = socket.gethostname()
            render_path = f'render_data/{hostname}/{timestap}/3d.zarr'
            if os.path.exists(render_path):
                subprocess.run(
                    f'{sys.executable} -m run.scripts.copy2zip_zarr --zarr-path {render_path}',
                    shell=True,
                )
