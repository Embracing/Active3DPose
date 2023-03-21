import os
import platform

import numpy as np
from ray.rllib.agents.callbacks import DefaultCallbacks as RllibCallbackBase
from ray.tune.callback import Callback as TuneCallbackBase


class CustomMetricCallback(RllibCallbackBase):
    DEFAULT_CUSTOM_METRICS = [
        'team_reward',
        'mpjpe_3d',
        # 'individual_reward',
        # 'total_reward',
        # 'vis_reward',
        # 'reconstruction_reward',
        #
        # 'distance_reward',
        # 'centering_reward',
        # 'iot_reward',
        # 'obstruction_reward',
        # 'min_partial_mpjpe_2c',
        # 'min_partial_mpjpe_3c',
        # 'min_partial_mpjpe_4c',
        #
        # 'min_partial_pck20_2c',
        # 'min_partial_pck20_3c',
        # 'min_partial_pck20_4c',
        #
        # 'mpjpe_0_best_rate', 'mpjpe_0_best_diff',
        # 'mpjpe_1_best_rate', 'mpjpe_1_best_diff',
        # 'mpjpe_2_best_rate', 'mpjpe_2_best_diff',
        #
        # 'mpjpe_01_best_rate', 'mpjpe_01_best_diff',
        # 'mpjpe_12_best_rate', 'mpjpe_12_best_diff',
        # 'mpjpe_20_best_rate', 'mpjpe_20_best_diff',
        # 'mpjpe_012_best_rate', 'mpjpe_012_best_diff',
        'lost_joints_ratio',
    ]
    DEFAULT_CUSTOM_METRICS += [f'pck3d_{m}' for m in range(5, 155, 5)]
    DEFAULT_CUSTOM_METRICS += [f'ex_pck3d_{m}' for m in range(5, 155, 5)]
    DEFAULT_CUSTOM_METRICS += ['avg_pck3d', 'avg_ex_pck3d']

    def __init__(self, custom_metrics=None):
        super().__init__()

        self.custom_metrics = custom_metrics or self.DEFAULT_CUSTOM_METRICS

    def on_episode_start(self, *, worker, base_env, policies, episode, env_index, **kwargs):
        for key in self.custom_metrics:
            episode.user_data[key] = []
        episode.user_data['num_cameras'] = None
        episode.user_data['num_humans'] = None

    def on_episode_step(self, *, worker, base_env, episode, env_index, **kwargs):
        agent_infos = list(map(episode.last_info_for, episode.get_agents()))

        for key in self.custom_metrics:
            values = []
            for info in agent_infos:
                try:
                    values.append(info[key])
                except KeyError:
                    pass
                if episode.user_data['num_cameras'] is None:
                    episode.user_data['num_cameras'] = info['num_cameras']
                if episode.user_data['num_humans'] is None:
                    episode.user_data['num_humans'] = info['num_humans']

            if len(values) > 0:
                episode.user_data[key].append(np.mean(values))

    def on_episode_end(self, *, worker, base_env, policies, episode, env_index, **kwargs):
        suffixes = ['']
        num_cameras = episode.user_data['num_cameras']
        num_humans = episode.user_data['num_humans']
        if num_cameras is not None and num_cameras is not None:
            suffixes.append(f'_{num_cameras}c{num_humans}h')

        for suffix in suffixes:
            for key in self.custom_metrics:
                episode.custom_metrics[f'{key}{suffix}'] = float(np.mean(episode.user_data[key]))
                if ('reward' in key or 'mpjpe' in key) and not key.startswith('episode'):
                    episode.custom_metrics[f'episode_{key}{suffix}'] = float(
                        np.sum(episode.user_data[key])
                    )
                    episode.custom_metrics[f'last_{key}{suffix}'] = float(
                        np.mean(episode.user_data[key][-10:])
                    )
                if key == 'mpjpe_3d':
                    episode.custom_metrics[f'{key}_stddev{suffix}'] = float(
                        np.std(episode.user_data[key], ddof=1)
                    )
                    success_rate_interval = list(range(0, 21)) + list(range(30, 110, 10))
                    for thresh in success_rate_interval:
                        success_rate = float(
                            np.mean(np.asarray(episode.user_data['mpjpe_3d']) <= thresh)
                        )
                        episode.custom_metrics[
                            f'episode_mpjpe_3d_success_rate_{thresh}{suffix}'
                        ] = success_rate


class SymlinkCheckpointCallback(TuneCallbackBase):
    def on_checkpoint(self, iteration, trials, trial, checkpoint, **info):
        source = checkpoint.value
        for target_dir in (trial.logdir, trial.local_dir):
            target = os.path.join(target_dir, 'latest-checkpoint')
            print(f'Symlink "{source}" to "{target}".')
            self.symlink(source, target)

    @staticmethod
    def symlink(source, target):
        temp_target = f'{target}.temp'

        os_symlink = getattr(os, 'symlink', None)
        if callable(os_symlink):
            os_symlink(source, temp_target)
        elif platform.system() == 'Windows':
            import ctypes

            csl = ctypes.windll.kernel32.CreateSymbolicLinkW
            csl.argtypes = (ctypes.c_wchar_p, ctypes.c_wchar_p, ctypes.c_uint32)
            csl.restype = ctypes.c_ubyte
            flags = 1 if os.path.isdir(source) else 0
            if csl(temp_target, source, flags) == 0:
                raise ctypes.WinError(f'Cannot create symlink "{source}" to "{target}".')
        else:
            raise OSError(f'Cannot create symlink "{source}" to "{target}".')

        os.replace(temp_target, target)
