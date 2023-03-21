import gym
import numpy as np

from .base import WrapperBase


# class GroundTruthObservation(WrapperBase):
#     def reset(self, **kwargs):
#         observation = self.env.reset(**kwargs)
#         info_list = self.env.info_trajectory * len(observation)
#         return self.observation(observation, info_list)

#     def step(self, action):
#         observation, reward, done, info_list = self.env.step(action)

#         return self.observation(observation, info_list), reward, done, info_list

#     def observation(self, observation, info_list):
#         """
#         info : list of dict
#         observation : [evolution_steps, C, 6+N_max*j*6]

#         proj2d may exceed img boundary if the target person is not in the FOV of the camera.
#         """

#         for idx, info in enumerate(info_list):

#             obs_max_human = np.zeros((
#                 self.env.num_cameras,
#                 self.env.max_num_humans,
#                 self.env.num_joints,
#                 6
#             ))  # [C, N_max, j, 6]

#             pred2d_scores = info['pred2d_scores']  # [C, N_max, j, 1], used to determine visibility
#             pred2d_scores = pred2d_scores[:, :self.env.controller.num_humans, ...]  # [C, N, j, 1]

#             obs_gt3d = np.tile(info['gt3d'], (self.env.num_cameras, 1, 1, 1))  # [C, N, j, 3]
#             obs_human = np.concatenate([obs_gt3d, info['proj2d'], pred2d_scores], axis=-1)  # [C, N, j, 6]
#             obs_max_human[:, :self.env.controller.num_humans, ...] = obs_human
#             obs_max_human = obs_max_human.reshape((self.env.num_cameras, -1))  # [C, N_max*j*6]

#             observation[idx, :, 6:] = obs_max_human
#         return observation


class GroundTruthObservation(WrapperBase):
    def __init__(self, env, gt_noise_scale=0, affected_by_visibility=False):
        super().__init__(env)
        env.unwrapped.use_gt_observation = True
        env.unwrapped.gt_noise_scale = gt_noise_scale
        env.unwrapped.gt_obs_with_visibility = affected_by_visibility
