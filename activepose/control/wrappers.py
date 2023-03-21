import glob
import os
import pickle as pkl
from collections import OrderedDict, defaultdict

import gym
import numpy as np
from gym import spaces

from activepose.config import config as CONFIG
from activepose.control.utils import ExponentialMovingStats


try:
    from ray import rllib
except ModuleNotFoundError:
    print('=> Warning: rllib is not installed!')
    RLlibHomogeneousMultiAgentEnv = None
    RllibMultiAgentAPI = None
else:

    class RLlibHomogeneousMultiAgentEnv(rllib.MultiAgentEnv):
        def observation_space_sample(self, agent_ids=None):
            if agent_ids is None:
                agent_ids = self.get_agent_ids()

            observation = {agent_id: self.observation_space.sample() for agent_id in agent_ids}
            return observation

        def action_space_sample(self, agent_ids=None):
            if agent_ids is None:
                agent_ids = self.get_agent_ids()

            actions = {agent_id: self.action_space.sample() for agent_id in agent_ids}
            return actions

        def action_space_contains(self, x):
            if not isinstance(x, dict):
                return False

            return all(map(self.action_space.contains, x.values()))

        def observation_space_contains(self, x):
            if not isinstance(x, dict):
                return False

            return all(map(self.observation_space.contains, x.values()))

    class RllibMultiAgentAPI(gym.Wrapper, RLlibHomogeneousMultiAgentEnv):
        def __init__(self, env, id_format='camera_{}'.format):
            super().__init__(env)

            self.id_format = id_format
            self.action_space = env.action_space[0]
            self.observation_space = env.observation_space[0]

            self.agent_ids = list(map(self.id_format, range(self.num_cameras)))
            self._agent_ids = set(self.agent_ids)
            setattr(self.unwrapped, '_agent_ids', self._agent_ids)
            setattr(self.unwrapped, 'get_agent_ids', lambda: self._agent_ids)

        def reset(self, **kwargs):
            return self.seq2dict(self.env.reset(**kwargs))

        def step(self, action):
            action = list(map(action.get, self.agent_ids))
            action = np.asarray(action)
            observations, rewards, dones, infos = tuple(map(self.seq2dict, self.env.step(action)))
            dones['__all__'] = all(dones.values())
            return observations, rewards, dones, infos

        def seq2dict(self, seq):
            return OrderedDict([(self.id_format(i), item) for i, item in enumerate(seq)])


class SingleEvaluationStep(gym.ObservationWrapper):
    def __init__(self, env):
        super().__init__(env)

        self.observation_space = spaces.Box(
            low=env.observation_space.low[-1].copy(),
            high=env.observation_space.high[-1].copy(),
            dtype=env.observation_space.dtype,
        )

    def step(self, action):
        observation, reward, done, info = self.env.step(action)
        return self.observation(observation), reward, done, info[-1]

    def observation(self, observation):
        return observation[-1]


class NormalizeObservation(gym.ObservationWrapper):
    def __init__(self, env):
        super().__init__(env)

        self.low = env.observation_space.low.copy()
        self.high = env.observation_space.high.copy()
        self.bounded_below = env.observation_space.bounded_below
        self.bounded_above = env.observation_space.bounded_above
        self.bounded_both = np.logical_and(self.bounded_below, self.bounded_above)
        self.mask = np.logical_and(self.bounded_both, self.high > self.low)
        self.half_diff = (self.high - self.low) / 2.0

        self.observation_space = spaces.Box(
            low=self.observation(self.low),
            high=self.observation(self.high),
            dtype=env.observation_space.dtype,
        )

    def observation(self, observation):
        normalized = observation.copy()
        normalized[..., self.bounded_below] = (
            normalized[..., self.bounded_below] - self.low[self.bounded_below]
        )
        normalized[..., self.mask] = normalized[..., self.mask] / self.half_diff[self.mask] - 1.0
        return normalized


class TruncateObservation(gym.ObservationWrapper):
    def __init__(self, env, num_observed_humans=None):
        super().__init__(env)

        if num_observed_humans is None:
            self.num_tracked_humans = self.num_humans  # default to all humans
        else:
            self.num_tracked_humans = int(num_observed_humans)
            # assert self.num_tracked_humans <= self.num_humans

        self.obs_dims = (
            CONFIG.OBS_DIM['ENV']
            + CONFIG.OBS_DIM['CAMERA']
            + self.num_tracked_humans * (CONFIG.OBS_DIM['HUMAN'])
        )

        low = env.observation_space.low[..., : self.obs_dims]
        high = env.observation_space.high[..., : self.obs_dims]
        self.observation_space = spaces.Box(low=low, high=high, dtype=env.observation_space.dtype)

    def observation(self, observation):
        return observation[..., : self.obs_dims]


class FlattenAction(gym.ActionWrapper):
    def __init__(self, env):
        super().__init__(env)
        self.action_space = spaces.MultiDiscrete(env.action_space.nvec.flatten())

    def action(self, action):
        return action.reshape(self.env.action_space.nvec.shape)

    def reverse_action(self, action):
        return action.flatten()


class OverrideNoAction(gym.ActionWrapper):
    def __init__(self, env):
        super().__init__(env)

    def action(self, action):
        return np.ones_like(action)

    def reverse_action(self, action):
        return np.ones_like(action)


class DiscreteAction(gym.ActionWrapper):
    def __init__(self, env):
        super().__init__(env)
        # if isinstance(env.action_space, spaces.Tuple):
        if self.MULTI_AGENT and isinstance(
            env.action_space, spaces.Tuple
        ):  # QMIX has spaces.Tuples as its action space
            self.action_space = spaces.Tuple(
                [spaces.Discrete(x.nvec.prod()) for x in env.action_space]
            )
            self.action = self.multi_action
        else:
            self.action_space = spaces.Discrete(env.action_space.nvec.prod())

    def action(self, action):
        action = int(action)
        multi_action = []
        for n in self.env.action_space.nvec.ravel():
            multi_action.append(action % n)
            action //= n
        multi_action = np.array(multi_action, dtype=self.env.action_space.dtype)

        return multi_action.reshape(self.env.action_space.nvec.shape)

    def multi_action(self, actions):
        actions = actions.astype('int64')
        expanded_actions = []

        for i, action in enumerate(actions):
            multi_action = []
            for n in self.env.action_space[-1].nvec.ravel():  # assume homogeneous agents
                multi_action.append(action % n)
                action //= n
            multi_action = np.array(multi_action, dtype=self.env.action_space[-1].dtype).reshape(
                self.env.action_space[-1].nvec.shape
            )
            expanded_actions.append(multi_action)

        return np.array(expanded_actions)


class FlattenObservation(gym.ObservationWrapper):
    def __init__(self, env):
        super().__init__(env)

        low = env.observation_space.low.flatten()
        high = env.observation_space.high.flatten()
        low.fill(-np.inf)
        high.fill(np.inf)
        self.observation_space = spaces.Box(low=low, high=high, dtype=env.observation_space.dtype)

    def observation(self, observation):
        return observation.flatten()


def get_coeff_dict(reward_dict: tuple) -> dict:
    if not reward_dict:
        return {
            'team_reward': 1.0,
            'vis_reward': 0.0,
            'individual_reward': 0.0,
            'distance_reward': 0.0,
            'centering_reward': 0.0,
            'obstruction_reward': 0.0,
            'iot_reward': 0.0,
            'anti_collision_reward': 0.0,
        }
    else:
        coeff_dict_name, coeff_dict = reward_dict
        return coeff_dict


class SingleAgentRewardLogger(gym.Wrapper):
    def __init__(self, env, reward_dict):
        super().__init__(env)
        coeff_dict = get_coeff_dict(reward_dict)
        self.coeff_team = coeff_dict.get('team_reward', 0.0)
        self.coeff_vis = coeff_dict.get('vis_reward', 0.0)
        self.coeff_distance = coeff_dict.get('distance_reward', 0.0)
        self.coeff_centering = coeff_dict.get('centering_reward', 0.0)
        self.coeff_obstruction = coeff_dict.get('obstruction_reward', 0.0)
        self.coeff_iot = coeff_dict.get('iot_reward', 0.0)
        self.coeff_anti_collision = coeff_dict.get('anti_collision_reward', 0.0)

    def step(self, action):
        def get_item(value):
            return np.asarray(value).ravel().mean()

        observation, reward, done, info = self.env.step(action)
        team_reward = get_item(info['reward_3d'])

        individual_reward = (
            self.coeff_vis * get_item(info['reward_vis'])
            + self.coeff_distance * get_item(info['reward_distance'])
            + self.coeff_centering * get_item(info['reward_centering'])
            + self.coeff_obstruction * get_item(info['reward_obstruction'])
            + self.coeff_anti_collision * get_item(info['reward_anti_collision'])
            + self.coeff_iot * get_item(info['reward_iot'])
        )

        total_reward = self.coeff_team * team_reward + individual_reward

        vis_reward = get_item(info['reward_vis'])
        reconstruction_reward = get_item(info['reward_3d'])
        centering_reward = get_item(info['reward_centering'])
        obstruction_reward = get_item(info['reward_obstruction'])
        iot_reward = get_item(info['reward_iot'])
        distance_reward = get_item(info['reward_distance'])
        anti_collision_reward = get_item(info['reward_anti_collision'])
        info.update(
            team_reward=team_reward,
            individual_reward=individual_reward,
            distance_reward=distance_reward,
            total_reward=total_reward,
            vis_reward=vis_reward,
            reconstruction_reward=reconstruction_reward,
            obstruction_reward=obstruction_reward,
            anti_collision_reward=anti_collision_reward,
            iot_reward=iot_reward,
            centering_reward=centering_reward,
        )

        return observation, total_reward, done, info


class FilterRedundantInfo(gym.Wrapper):
    def step(self, action):
        observation, reward, done, info = self.env.step(action)

        def func(key):
            return (
                'reward' in key
                or 'shapley' in key
                or 'mpjpe' in key
                or 'pck3d' in key
                or 'expert' in key
                or key == 'prev_info'
                or key == 'index'
                or key == 'num_cameras'
                or key == 'num_humans'
                or key == 'cam_param_list'
                or key == 'inv_depth_list'
                or key == 'joint_action'
                or key == 'gt3d'
                or key == 'observation'
                or key == 'lost_joints_ratio'
                or key == 'gt_obs_dict'
                or key == 'pred_obs_dict'
                or key == 'imit_pitch_action'
                or key == 'imit_yaw_action'
                or key == 'cam_human_distance'
                or key == 'move_policy'
                or key == '_process_time_dict'
                or key == 'vis_ratio_2d'
                or key == 'vis_ratio_3d'
            )

        if isinstance(info, dict):
            info = {key: info[key] for key in info if func(key)}
        elif isinstance(info, list):
            info = [{key: i[key] for key in i if func(key)} for i in info]

        return observation, reward, done, info


class SplitActionSpace(gym.ActionWrapper):
    def __init__(self, env):
        super().__init__(env)

        self.num_cameras = env.num_cameras
        space = spaces.MultiDiscrete(env.action_space.nvec[0])
        self.action_space = spaces.Tuple([space] * self.num_cameras)

    def action(self, action):
        return np.asarray(action).reshape(self.env.action_space.nvec.shape)

    def reverse_action(self, action):
        return tuple(action)


class JointObservationTuneReward(gym.ObservationWrapper):
    def __init__(self, env, reward_dict, teammate_stats_dim):
        assert env.MULTI_AGENT, 'Cannot use JointObservation without being multi-agent'
        super().__init__(env)

        self.num_cameras = env.num_cameras

        matrix = []
        for i in range(self.num_cameras):
            matrix.append(np.roll(np.arange(self.num_cameras), -i))
        self.matrix = np.asarray(matrix, dtype=np.int64)

        low = env.observation_space.low.copy()
        high = env.observation_space.high.copy()
        low.fill(-np.inf)
        high.fill(+np.inf)

        if teammate_stats_dim is not None:
            self.other_dim = teammate_stats_dim
            self.camera_index_slice = slice(
                CONFIG.OBS_SLICES['CAMERA'].start,
                CONFIG.OBS_SLICES['CAMERA'].start + self.other_dim,
            )

            new_low = np.concatenate(
                [low[0]] + [low[0][self.camera_index_slice]] * (self.num_cameras - 1)
            )
            # actually dont need to do very specific indexing, like [8 : 9 + self.other_dim]
            # as all of the lows and highs are set to be infinities here...
            new_high = np.concatenate(
                [high[0]] + [high[0][self.camera_index_slice]] * (self.num_cameras - 1)
            )
        else:
            self.other_dim = low[0].size
            new_low = low.flatten()
            new_high = high.flatten()

        space = spaces.Box(low=new_low, high=new_high, dtype=env.observation_space.dtype)

        self.observation_space = spaces.Tuple((space,) * self.num_cameras)

        # Reward function coefficients
        coeff_dict = get_coeff_dict(reward_dict)

        self.coeff_team = coeff_dict.get('team_reward', 0.0)
        self.coeff_vis = coeff_dict.get('vis_reward', 0.0)
        self.coeff_distance = coeff_dict.get('distance_reward', 0.0)
        self.coeff_centering = coeff_dict.get('centering_reward', 0.0)
        self.coeff_obstruction = coeff_dict.get('obstruction_reward', 0.0)
        self.coeff_iot = coeff_dict.get('iot_reward', 0.0)
        self.coeff_anti_collision = coeff_dict.get('anti_collision_reward', 0.0)

        self.prev_action = None

    def reset(self):
        observation = self.env.reset()
        self.prev_action = np.ones_like(self.action_space.sample())
        return self.observation(observation)

    def step(self, action):
        observation, reward, done, info = self.env.step(action)
        observations = self.observation(observation)

        team_reward = info['reward_3d']
        reconstruction_reward = info['reward_3d']
        vis_rewards = np.asarray(info['reward_vis'], dtype=np.float64)
        distance_rewards = np.asarray(info['reward_distance'], dtype=np.float64)
        centering_rewards = np.asarray(info['reward_centering'], dtype=np.float64)
        obstruction_rewards = np.asarray(info['reward_obstruction'], dtype=np.float64)
        iot_rewards = np.asarray(info['reward_iot'], dtype=np.float64)
        anti_collision_rewards = np.asarray(info['reward_anti_collision'], dtype=np.float64)

        rewards = (
            self.coeff_team * team_reward
            + self.coeff_vis * vis_rewards
            + self.coeff_distance * distance_rewards
            + self.coeff_centering * centering_rewards
            + self.coeff_obstruction * obstruction_rewards
            + self.coeff_anti_collision * anti_collision_rewards
            + self.coeff_iot * iot_rewards
        ).tolist()
        # print(f"rewards = {team_reward}+{vis_rewards}+{distance_rewards}+{centering_rewards}")
        dones = [done] * self.num_cameras
        infos = [
            {
                **info,
                'index': c,
                'joint_action': np.array(action),
                'team_reward': team_reward,
                'vis_reward': vis_rewards[c],
                'total_reward': rewards[c],
                'distance_reward': distance_rewards[c],
                'centering_reward': centering_rewards[c],
                'iot_reward': iot_rewards[c],
                'anti_collision_reward': anti_collision_rewards[c],
                'reconstruction_reward': reconstruction_reward,
                'obstruction_reward': obstruction_rewards[c],
                'mpjpe_3d': info['mpjpe_3d'],
                'state_reward': info['reward_camera_state'][c],
            }
            for c in range(self.num_cameras)
        ]
        for c, info in enumerate(infos):
            if 'prev_info' in info:
                info['prev_info'] = info['prev_info'].copy()
                info['prev_info'].update(
                    {
                        'index': c,
                        'joint_action': self.prev_action,
                    }
                )

        self.prev_action = np.array(action)

        return observations, rewards, dones, infos

    def observation(self, observation):
        observations = []
        for c in range(self.num_cameras):
            own_obs = observation[self.matrix[c]][0]
            # self.camera_index_slice usually is [8:8 + other_dim]
            other_obs = np.asarray(
                [obs[self.camera_index_slice] for obs in observation[self.matrix[c]][1:]]
            )
            observations.append(np.concatenate((own_obs, other_obs.flatten())))

        return np.asarray(observations)


class ShapleyValueReward(gym.Wrapper):
    def __init__(self, env):
        super().__init__(env)

        self.episode_steps = 0
        self.counts = {'0': 0, '1': 0, '2': 0, '01': 0, '12': 0, '20': 0, '012': 0}

        self.shapley_coeffs = None

    def reset(self, **kwargs):
        self.episode_steps = 0
        self.counts = {'0': 0, '1': 0, '2': 0, '01': 0, '12': 0, '20': 0, '012': 0}

        self.shapley_coeffs = None

        return self.env.reset(**kwargs)

    def step(self, action):
        observation, reward, done, info = super().step(action)

        values = info[0]['shapley_reward_dict']

        if self.shapley_coeffs is None:
            n = self.num_cameras
            norm = np.math.factorial(n)
            self.shapley_coeffs = [dict() for _ in range(n)]
            for key in values:
                key_set = set(key)
                c = len(key_set) - 1
                coeff = np.math.factorial(c) * np.math.factorial(n - c - 1) / norm
                for i in key_set:
                    self.shapley_coeffs[i][key] = coeff
                if c >= 2:
                    for i in key_set:
                        self.shapley_coeffs[i][tuple(sorted(key_set.difference([i])))] = -coeff

        for i in range(self.num_cameras):
            r = 0.0
            for key, coeff in self.shapley_coeffs[i].items():
                r += coeff * values[key]
            reward[i] = r * self.num_cameras

        shapley_mpjpe = info[0]['shapley_mpjpe_dict']
        partial_mpjpe = defaultdict(list)
        for k, v in shapley_mpjpe.items():
            partial_mpjpe[len(k)].append(v)
        partial_mpjpe = {f'min_partial_mpjpe_{k}c': np.min(v) for k, v in partial_mpjpe.items()}

        shapley_pck20 = info[0]['shapley_pck20_dict']
        partial_pck20 = defaultdict(list)
        for k, v in shapley_pck20.items():
            partial_pck20[len(k)].append(v)
        partial_pck20 = {f'max_partial_pck20_{k}c': np.max(v) for k, v in partial_pck20.items()}

        mpjpe = {}
        mpjpe['01'] = shapley_mpjpe[(0, 1)]
        mpjpe['12'] = shapley_mpjpe[(1, 2)]
        mpjpe['20'] = shapley_mpjpe[(0, 2)]
        mpjpe['012'] = shapley_mpjpe[(0, 1, 2)]
        best_mpjpe = min(mpjpe.values())
        mpjpe['0'] = min(mpjpe['01'], mpjpe['20'])
        mpjpe['1'] = min(mpjpe['01'], mpjpe['12'])
        mpjpe['2'] = min(mpjpe['20'], mpjpe['12'])
        self.counts['0'] += int(best_mpjpe == mpjpe['0'])
        self.counts['1'] += int(best_mpjpe == mpjpe['1'])
        self.counts['2'] += int(best_mpjpe == mpjpe['2'])
        self.counts['01'] += int(best_mpjpe == mpjpe['01'])
        self.counts['12'] += int(best_mpjpe == mpjpe['12'])
        self.counts['20'] += int(best_mpjpe == mpjpe['20'])
        self.counts['012'] += int(best_mpjpe == mpjpe['012'])
        self.episode_steps += 1

        partial_mpjpe.update(
            {
                f'mpjpe_{key}_best_rate': step / self.episode_steps
                for key, step in self.counts.items()
            }
        )
        partial_mpjpe.update(
            {f'mpjpe_{key}_best_diff': best_mpjpe - value for key, value in mpjpe.items()}
        )

        for i in info:
            i.update(partial_mpjpe)
            i.update(partial_pck20)

        return observation, reward, done, info


class DoneWhenColliding(gym.Wrapper):
    def __init__(self, env, threshold=50.0, collision_tolerance=1):
        super().__init__(env)

        self.threshold = threshold
        self.collision_tolerance = collision_tolerance
        self.count = 0

    def reset(self, **kwargs):
        self.count = 0
        return self.env.reset(**kwargs)

    def step(self, action):
        observations, rewards, dones, infos = self.env.step(action)

        cam_param_list, gt3d = infos[0]['cam_param_list'], infos[0]['gt3d']

        collision = False
        for i, cam in enumerate(cam_param_list):
            target_xys = np.asarray([target.mean(axis=0)[:2] for target in gt3d])
            cam_xy, cam_z = np.asarray(cam[:2]), cam[3]
            if (
                np.linalg.norm(cam_xy[np.newaxis, :] - target_xys, axis=-1) < self.threshold
            ).any() and cam_z < 200:
                collision = True
                break

        if collision:
            self.count += 1
            rewards = (np.array(rewards) - 10).tolist()
            if self.count >= self.collision_tolerance:
                rewards = (np.array(rewards) - 10).tolist()
                dones = [True] * len(dones)
        else:
            self.count = max(self.count - 2, 0)

        return observations, rewards, dones, infos


class RunningNormalizedReward(gym.Wrapper):
    def __init__(self, env, momentum=0.1):
        super().__init__(env)

        assert 0.0 <= momentum < 1.0
        self.momentum = momentum

        self.running_stats = None

    def step(self, action):
        observation, reward, done, info = self.env.step(action)

        if isinstance(reward, (list, tuple, np.ndarray)):
            if self.running_stats is None:
                self.running_stats = [ExponentialMovingStats(self.momentum) for _ in reward]

            new_reward = []
            for i, r, rs in zip(info, reward, self.running_stats):
                rs.push(r)
                new_r = (r - rs.mean()) / max(rs.standard_deviation(), 1e-8)
                new_reward.append(new_r)

                i['reward_running_mean'] = rs.mean()
                i['reward_running_stddev'] = rs.standard_deviation()
                i['reward_normalized'] = new_r

            reward = new_reward
        elif isinstance(reward, dict):
            if self.running_stats is None:
                self.running_stats = {
                    agent_id: ExponentialMovingStats(self.momentum) for agent_id in reward
                }

            new_reward = {}
            for agent_id in reward:
                r = reward[agent_id]
                rs = self.running_stats[agent_id]
                rs.push(r)
                new_r = (r - rs.mean()) / max(rs.standard_deviation(), 1e-8)
                new_reward[agent_id] = new_r

                info[agent_id]['reward_running_mean'] = rs.mean()
                info[agent_id]['reward_running_stddev'] = rs.standard_deviation()
                info[agent_id]['reward_normalized'] = new_r

            reward = new_reward
        else:
            if self.running_stats is None:
                self.running_stats = ExponentialMovingStats(self.momentum)

            rs = self.running_stats

            rs.push(reward)
            reward = (reward - rs.mean()) / (max(rs.standard_deviation(), 1e-8))

            info['reward_running_mean'] = rs.mean()
            info['reward_running_stddev'] = rs.standard_deviation()
            info['reward_normalized'] = reward

        return observation, reward, done, info


class MultiDiscrete2DiscreteActionMapper:
    def __init__(self, original_space):
        assert isinstance(original_space, spaces.MultiDiscrete)
        self.nvec = original_space.nvec
        self.original_space = original_space
        self.original_mask_space = spaces.MultiBinary(np.sum(self.nvec))

        self.n = np.prod(self.nvec)
        self.space = spaces.Discrete(self.n)
        self.mask_space = spaces.MultiBinary(self.n)

        self.strides = np.asarray(
            list(reversed(np.cumprod(list(reversed(self.nvec.ravel())))))[1:] + [1],
            dtype=self.space.dtype,
        )

        self._mask_mapping = None

    @property
    def mask_table(self):
        if self._mask_mapping is None:
            self._mask_mapping = np.zeros((self.n, np.sum(self.nvec)), dtype=np.bool8)
            all_multi_discrete_actions = self.multi_discrete_action_batched(
                list(range(self.n)), strict=False
            )
            offsets = np.cumsum([0, *self.nvec.ravel()[:-1]], dtype=np.int64)
            indices = all_multi_discrete_actions.reshape(self.n, -1) + offsets[np.newaxis, :]
            for n, index in enumerate(indices):
                self._mask_mapping[n, index] = True

        return self._mask_mapping

    def multi_discrete_action_batched(self, discrete_action_batch, strict=True):
        discrete_action_batch = np.asarray(discrete_action_batch, dtype=self.space.dtype)

        assert discrete_action_batch.ndim == 1
        if strict:
            for discrete_action in discrete_action_batch:
                assert self.space.contains(discrete_action), (
                    f'Discrete action {discrete_action} outside given '
                    f'discrete action space {self.space}.'
                )

        multi_discrete_action_batch = []
        for s in self.strides:
            multi_discrete_action_batch.append(discrete_action_batch // s)
            discrete_action_batch = discrete_action_batch % s

        multi_discrete_action_batch = np.stack(multi_discrete_action_batch, axis=-1)
        return multi_discrete_action_batch.reshape(-1, *self.original_space.shape).astype(
            self.original_space.dtype
        )

    def multi_discrete_action(self, discrete_action):
        return self.multi_discrete_action_batched([discrete_action])[0]

    def discrete_action_batched(self, multi_discrete_action_batch, strict=True):
        multi_discrete_action_batch = np.asarray(
            multi_discrete_action_batch, dtype=self.original_space.dtype
        )

        assert multi_discrete_action_batch.shape[1:] == self.nvec.shape
        if strict:
            for multi_discrete_action in multi_discrete_action_batch:
                assert self.original_space.contains(multi_discrete_action), (
                    f'Multi-discrete action {multi_discrete_action} outside given '
                    f'multi-discrete action space {self.original_space}.'
                )

        batch_size = multi_discrete_action_batch.shape[0]
        multi_discrete_action_batch = multi_discrete_action_batch.reshape(batch_size, -1)

        discrete_action_batch = (self.strides[np.newaxis, :] * multi_discrete_action_batch).sum(
            axis=-1
        )
        return discrete_action_batch.astype(self.space.dtype).ravel()

    def discrete_action(self, multi_discrete_action):
        return self.discrete_action_batched([multi_discrete_action])[0]

    def discrete_action_mask(self, multi_discrete_action_mask):
        multi_discrete_action_mask = np.asarray(multi_discrete_action_mask, dtype=np.bool8)

        assert self.original_mask_space.contains(multi_discrete_action_mask), (
            f'Multi-discrete action mask {multi_discrete_action_mask} outside given '
            f'Multi-discrete action mask space {self.original_mask_space}.'
        )

        return (multi_discrete_action_mask >= self.mask_table).all(axis=-1)


class FlattenMultiDiscrete(gym.ActionWrapper):
    def __init__(self, env):
        super().__init__(env)

        if isinstance(env.action_space, spaces.Tuple):
            self.tuple_space = True
            self.action_mapper = tuple(
                MultiDiscrete2DiscreteActionMapper(original_space=space)
                for space in env.action_space
            )
            self.action_space = spaces.Tuple(tuple(mapper.space for mapper in self.action_mapper))
        else:
            self.tuple_space = False
            assert isinstance(env.action_space, spaces.MultiDiscrete)
            self.action_mapper = MultiDiscrete2DiscreteActionMapper(original_space=env.action_space)
            self.action_space = self.action_mapper.space

    def action(self, action):
        if self.tuple_space:
            return tuple(
                self.action_mapper[i].multi_discrete_action(action[i]) for i in range(len(action))
            )
        return self.action_mapper.multi_discrete_action(action)

    def reverse_action(self, action):
        if self.tuple_space:
            return tuple(
                self.action_mapper[i].discrete_action(action[i]) for i in range(len(action))
            )
        return self.action_mapper.discrete_action(action)
