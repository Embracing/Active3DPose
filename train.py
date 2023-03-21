import argparse
import os
import socket
from datetime import datetime

import ray
import torch
from experiments.mappo_ctcr_wdl import make_ctcr_wdl_experiment
from ray import tune

from activepose import DEBUG, ROOT_DIR
from activepose.control.envs import build_train_env
from activepose.custom import load_custom_models
from activepose.utils.wandb_custom_callback import CustomWandbCallback


load_custom_models()

parser = argparse.ArgumentParser(description='')
parser.add_argument('--tags', nargs='+', default=[], help='wandb tags')
parser.add_argument('--project', type=str, help='wandb project')
parser.add_argument('--group', type=str, help='wandb group')
parser.add_argument('--restore', type=str, default=None, help='path to rllib checkpoints')
parser.add_argument(
    '--no-cluster',
    action='store_true',
    default=True,
    help='initiate a local ray instance instead of a remote ray cluster',
)
# === temp ===
parser.add_argument('--num-cams', type=int, default=2, help='number of cameras')
parser.add_argument(
    '--exp-mode',
    type=str,
    default='MAPPO+CTCR+WDL',
    help='experiment mode, pick one from [MAPPO+CTCR+WDL, MAPPO+CTCR, MAPPO+WDL, MAPPO]',
)
args = parser.parse_args()

LOCAL_DIR = os.path.join(ROOT_DIR, 'ray_results')

CLUSTER_MODE = True if not args.no_cluster else False

torch.cuda.device_count()

if not CLUSTER_MODE:
    NUM_NODE_CPUS = os.cpu_count()
    NUM_NODE_GPUS = torch.cuda.device_count()
    ray.init(num_cpus=NUM_NODE_CPUS, local_mode=DEBUG)
elif DEBUG:
    ray.init(local_mode=DEBUG)
else:
    ray.init(address='auto', dashboard_host='0.0.0.0', dashboard_port=8888)

cluster_resource = ray.cluster_resources()
NUM_NODE_CPUS = int(cluster_resource['CPU'])
NUM_NODE_GPUS = int(cluster_resource['GPU'])

print(f'CLUSTER MODE: {CLUSTER_MODE}')
print(f'DEBUG MODE: {DEBUG}')
print(f'NUM_NODE_CPUS: {NUM_NODE_CPUS}')
print(f'NUM_NODE_GPUS: {NUM_NODE_GPUS}')

exp = make_ctcr_wdl_experiment(
    n_cams=args.num_cams,
    EXP_MODE=args.exp_mode,
)

tmp_env_config = exp.spec['config']['env_config']
if args.group is None:
    wandb_group = f"{exp.spec['config']['env_config']['args'].env_name}"
else:
    wandb_group = args.group

exp.spec[
    'trial_dirname_creator'
] = lambda trial: f"{trial.trainable_name}_{trial.trial_id}_{datetime.now().strftime('%b%d')}"
exp.spec['config']['env_config']['UE4Binary_SLEEPTIME'] = '60'

callbacks = [
    CustomWandbCallback(
        project=args.project,
        group=wandb_group,
        api_key_file='wandb_api_key',
        tags=args.tags,
        force=True,
        log_config=False,
    )
]  # set log_config to False so you don't get that bunch of useless stats on wandb

if DEBUG:
    exp.spec['config']['env_config']['UE4Binary_SLEEPTIME'] = '20'
    exp.spec['config']['num_workers'] = 1
    exp.spec['config']['num_envs_per_worker'] = 1
    exp.spec['config']['train_batch_size'] = (
        1
        * exp.spec['config']['num_workers']
        * exp.spec['config']['num_envs_per_worker']
        * exp.spec['config']['rollout_fragment_length']
    )
    exp.spec['config']['sgd_minibatch_size'] = exp.spec['config']['train_batch_size'] // 1
    callbacks = []

tune.register_env('active-pose-parallel', build_train_env)

# %%
# For Wandb
from activepose.config import config as unreal_config
from activepose.config import update_config


update_config(unreal_config, exp.spec['config']['env_config']['args'])
for i, c in enumerate(callbacks):
    if isinstance(c, CustomWandbCallback):
        setattr(callbacks[i], 'unreal_config', unreal_config)

# %%
if args.restore:
    exp.spec['restore'] = args.restore

print(
    f"train_batch_size = {exp.spec['config']['train_batch_size']}, sgd_minibatch_size = {exp.spec['config']['sgd_minibatch_size']}"
)
print(exp.spec['config'])

tune.run_experiments(experiments=exp, callbacks=callbacks, verbose=3)
