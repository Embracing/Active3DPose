import shutil
import time
from threading import Thread

from ray import tune

from activepose.envs import *


NUM_ENVS_PER_WORKER = 4
from .wrappers import *


def remove_unreal_logs(binary_path):
    binary_dir = os.path.dirname(os.path.abspath(binary_path))
    log_path = os.path.abspath(os.path.join(binary_dir, '..', '..', 'Saved'))
    while True:
        if os.path.exists(log_path):
            shutil.rmtree(log_path, ignore_errors=True)
        time.sleep(30 * 60)


def build_train_env(env_config):
    env = make_train_env(env_config, is_training=True)
    return env


def make_train_env(env_config, is_training):
    worker_index = getattr(env_config, 'worker_index', 0)
    vector_index = getattr(env_config, 'vector_index', 0)

    os.environ['UE4Binary_SLEEPTIME'] = env_config.get('UE4Binary_SLEEPTIME', '120')
    os.environ['DISPLAY'] = ':'

    if is_training:
        env_config['args'].worker_index = worker_index
        env_config['args'].vector_index = vector_index

        if env_config.get('mixed_training', True):
            env_config['args'].num_humans = 1 + worker_index % 6

    env = make_env(**env_config)  # update config in make_env

    daemon = Thread(
        name='remove_unreal_logs',
        target=remove_unreal_logs,
        args=(env.binary.bin_path,),
        daemon=True,
    )
    daemon.start()

    env = SingleEvaluationStep(env)

    env = NormalizeObservation(env)

    if env_config.get('aux_rewards', False) and env_config['aux_rewards'].get('use', False):
        env = Aux_Rewards(env, **env_config['aux_rewards'].get('args', dict()))

    if env.unwrapped.MULTI_AGENT and not env_config.get('force_single_agent', False):
        # Multi-Agents Mode

        env = SplitActionSpace(env)
        if env_config.get('convert_multi_discrete_to_discrete', False):
            env = DiscreteAction(env)

        env = JointObservationTuneReward(
            env,
            teammate_stats_dim=env_config.get('teammate_stats_dim', 9),
            reward_dict=env_config.get('reward_dict', None)
            if env_config.get('aux_rewards', False)
            else None,
        )

        if env_config.get('shapley_reward', False):
            assert env_config.get(
                'partial_triangulation', False
            ), 'Shapley Reward only works with partial triangulation'
            env = ShapleyValueReward(env)

        if env_config.get('done_when_colliding', False) and env_config['done_when_colliding'].get(
            'use', False
        ):
            env = DoneWhenColliding(env, **env_config['done_when_colliding'].get('args', dict()))

        if env_config.get('remove_info', False):
            env = FilterRedundantInfo(env)

        if env_config.get('running_normalized_reward', False):
            env = RunningNormalizedReward(env, momentum=0.1)

        if env_config.get('flatten_multi_discrete_action', False):
            env = FlattenMultiDiscrete(env)

        env = RllibMultiAgentAPI(env)

    else:
        # Single-Agent Mode

        env = FlattenAction(env)

        # only DQN use this wrapper now
        if env_config.get('convert_multi_discrete_to_discrete', False):
            env = DiscreteAction(env)

        env = FlattenObservation(env)

        if is_training:
            env = SingleAgentRewardLogger(env=env, reward_dict=env_config.get('reward_dict', None))

        if env_config.get('remove_info', False):
            env = FilterRedundantInfo(env)

        if env_config.get('running_normalized_reward', False):
            env = RunningNormalizedReward(env, momentum=0.1)

    return env
