# %%
import os
import subprocess
import sys
from datetime import datetime


UE4Binary_SLEEPTIME = 10

ROOT_DIR = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, ROOT_DIR)
os.chdir(ROOT_DIR)
os.environ['UE4Binary_SLEEPTIME'] = str(UE4Binary_SLEEPTIME)
print(f'ROOT DIR: {ROOT_DIR}')

import argparse
import pickle as pkl

import numpy as np

from activepose.config import binary_version_match, human_model_match, unreal_map_match

# %%
from activepose.control.envs import make_train_env


np.set_printoptions(precision=2)


def parse_args():
    from activepose.env_config import map_config_dict

    parser = argparse.ArgumentParser(description='eval')
    parser.add_argument('--num-humans', type=int, default=6, help='number of humans')
    parser.add_argument('--ckpt', type=str, required=True, help='path to rllib checkpoints')
    parser.add_argument('--ckpt-num', type=str, default=None, help='checkpoint number')
    parser.add_argument(
        '--env-name',
        type=str,
        help='name of the environment under activepose/env_config.py',
    )
    parser.add_argument(
        '--map-name',
        type=str,
        choices=list(map_config_dict.keys()),
        default='Blank',
        help='name of the map, pick from [Blank, SchoolGymDay, Building, Wilderness]',
    )
    parser.add_argument(
        '--render-steps',
        type=int,
        default=500,
        help='number of steps to run (and render if --no-render is not set)',
    )
    parser.add_argument('--num-episodes', type=int, default=1, help='number of episodes to run')
    parser.add_argument(
        '--no-render',
        default=False,
        action='store_true',
        help='do not render and store video',
    )
    parser.add_argument(
        '--use-gt', default=False, action='store_true', help='use ground truth 2D pose'
    )
    args = parser.parse_args()
    return args


# %%

cmd_args = parse_args()

if cmd_args.ckpt_num is None:
    # Default to take last
    cp_dir = sorted([f for f in os.listdir(cmd_args.ckpt) if 'checkpoint' in f], reverse=True)[0]
    cp_num = str(int(cp_dir.rsplit('_')[1]))
else:
    cp_num = str(cmd_args.ckpt_num)
    cp_dir = 'checkpoint' + '_' + cmd_args.ckpt_num.zfill(6)

cp_path = os.path.join(cmd_args.ckpt, cp_dir)
config_path = os.path.join(cmd_args.ckpt, 'params.pkl')
with open(config_path, 'rb') as f:
    cp_config = pkl.load(f)

checkpoint_path = os.path.join(cp_path, 'checkpoint-' + cp_num)
with open(checkpoint_path, mode='rb') as file:
    checkpoint = pkl.load(file)

import pprint

# %%
from ray.rllib.agents import ppo


if cmd_args.env_name is not None:
    cp_config['env_config']['args'].env_name = cmd_args.env_name

if cmd_args.map_name is not None:
    assert cmd_args.map_name in human_model_match, 'Map name not found!'
    assert cmd_args.map_name in binary_version_match, 'Map name not found!'
    cp_config['env_config']['args'].map_name = cmd_args.map_name
    cp_config['env_config']['args'].human_model = human_model_match[cmd_args.map_name]

if cmd_args.use_gt:
    print('Using Ground Truth!')
    cp_config['env_config']['gt_observation'] = {
        'use': True,
        'args': {'gt_noise_scale': 0.0, 'affected_by_visibility': False},
    }
else:
    print('Using Prediction!')
    cp_config['env_config']['gt_observation'] = {
        'use': False,
        'args': {'gt_noise_scale': 0.0},
    }

if cmd_args.num_humans is not None:
    cp_config['env_config']['args'].num_humans = cmd_args.num_humans

# cp_config['env_config']['ema_action'] = {
#         'use': True,
#         'args': {
#             'k_EMA': 0.5,
#             'action_noise': [0.8, 1.2],
#             'add_extra_force': False,
#             'force_configs': {
#                 'F': 30,
#                 'human_safe_distance' : 150.,
#                 'camera_safe_distance' : 90.,
#             }
#         }
#     }

cp_config['env_config']['UE4Binary_SLEEPTIME'] = str(UE4Binary_SLEEPTIME)
pprint.pprint(cp_config['env_config'])
env = make_train_env(cp_config.get('env_config', dict()), is_training=False)

algo = cp_config['env_config'].get('algo', 'PPO').upper()
if algo == 'PPO':
    policy_class = ppo.PPOTorchPolicy
elif cp_config['model']['custom_model'] is not None:
    from activepose.custom import CUSTOM_POLICIES_FROM_MODEL_NAMES, load_custom_models

    load_custom_models()
    policy_class = CUSTOM_POLICIES_FROM_MODEL_NAMES.get(
        cp_config['model']['custom_model'], ppo.PPOTorchPolicy
    )
else:
    raise NotImplementedError

policies = {}
policy_mapping_fn = lambda agent_id: 'default_policy'
if 'multiagent' in cp_config:
    for policy_id in cp_config['multiagent']['policies']:
        policies[policy_id] = policy_class(env.observation_space, env.action_space, cp_config)
    if 'policy_mapping_fn' in cp_config['multiagent']:
        policy_mapping_fn = cp_config['multiagent']['policy_mapping_fn']
else:
    policies['default_policy'] = policy_class(env.observation_space, env.action_space, cp_config)

from tqdm import tqdm

from activepose.config import binary_version_match, config, human_model_match

# %%
from activepose.utils.file import create_logger


worker = pkl.loads(checkpoint['worker'])
for policy_id, policy in policies.items():
    state = worker['state'][policy_id]
    state.pop('_optimizer_variables', None)

    try:
        if env.unwrapped.MULTI_AGENT:
            policy.set_state(state)
        else:
            if 'weights' in state:
                weights = state['weights']
            else:
                weights = state
            policy.set_weights(weights)
    except RuntimeError as e:
        print(e, file=sys.stderr)

# %%
result_dict = {}
mpjpe_3d_list = []
pck3d_150_list = []
num_cameras = env.unwrapped.num_cameras

print(cp_config['env_config']['ema_action'])
tag_name = datetime.now().strftime('%d%m%Y_%H%M')

if not cmd_args.no_render:
    logger, final_output_dir, timestap = create_logger(config, 'online')
    render_type = 'offline-save'

for ep in range(cmd_args.num_episodes):
    if env.unwrapped.MULTI_AGENT and not cp_config['env_config'].get('force_single_agent', False):
        observations = env.reset()
        agent_ids = list(observations.keys())
        policies_by_agent_id = {
            agent_id: policies[policy_mapping_fn(agent_id)] for agent_id in agent_ids
        }
        infos = {agent_id: {} for agent_id in agent_ids}
        states = {
            agent_id: policy.model.get_initial_state()
            for agent_id, policy in policies_by_agent_id.items()
        }
        actions = {}

        if not cmd_args.no_render:
            env.render(mode=render_type, timestap=timestap)
        for idx in tqdm(range(cmd_args.render_steps)):
            for agent_id, policy in policies_by_agent_id.items():
                results = policy.compute_single_action(
                    observations[agent_id],
                    state=states[agent_id],
                    info=infos[agent_id],
                    explore=False,
                )

                actions[agent_id], states[agent_id], *_ = results

            observations, _, dones, infos = env.step(actions)

            mpjpe_3d_list.append(10 * infos[agent_ids[0]]['mpjpe_3d'])
            pck3d_150_list.append(infos[agent_ids[0]]['pck3d_150'])
            mpjpe_mean = np.mean(mpjpe_3d_list)
            mpjpe_stddev = np.std(mpjpe_3d_list, ddof=1)
            mpjpe_percentiles = np.percentile(mpjpe_3d_list, [0, 25, 50, 75, 100])
            print(
                f'MPJPE = {mpjpe_3d_list[-1]:.2f}mm (mean={mpjpe_mean:.2f}mm, stddev={mpjpe_stddev:.2f}mm, percentiles={mpjpe_percentiles})'
            )
            print(
                f'PCK3D(150mm) = {pck3d_150_list[-1] * 100:.2f} (mean={np.mean(pck3d_150_list) * 100:.2f}%)'
            )

            if not cmd_args.no_render:
                env.render(mode=render_type, timestap=timestap)
            if dones['__all__']:
                break
    else:
        raise NotImplementedError
try:
    env.close()
except Exception:
    pass
if not cmd_args.no_render:
    import socket

    hostname = socket.gethostname()
    subprocess.run(
        f'{sys.executable} -m run.scripts.copy2zip_zarr --zarr-path render_data/{hostname}/{timestap}/3d.zarr',
        shell=True,
    )
