import warnings
from typing import Dict, List, Tuple

import numpy as np
import torch
import torch.nn as nn
from gym.spaces import Discrete, MultiDiscrete
from ray.rllib.models.modelv2 import ModelV2
from ray.rllib.models.preprocessors import get_preprocessor
from ray.rllib.models.torch.recurrent_net import RecurrentNetwork as TorchRNN
from ray.rllib.policy.rnn_sequencing import add_time_dimension
from ray.rllib.policy.sample_batch import SampleBatch
from ray.rllib.policy.view_requirement import ViewRequirement
from ray.rllib.utils.annotations import override
from ray.rllib.utils.torch_utils import one_hot, sequence_mask
from ray.rllib.utils.typing import ModelConfigDict, TensorType

from ..model import MDN, MLP


class Embedding(nn.Module):
    def __init__(
        self,
        num_cameras,
        max_num_humans,
        lstm_size,
        fcnet_hiddens=None,
        masking_target=True,
    ):
        super().__init__()

        if fcnet_hiddens is None:
            fcnet_hiddens = [128, 128, 128]
        self.num_cameras = C = num_cameras
        self.max_num_humans = Nh = max_num_humans
        self.fcnet_hiddens = fcnet_hiddens
        self.lstm_size = lstm_size
        self.embed_dim = fcnet_hiddens[-1]
        self.feature_dim = self.embed_dim + self.lstm_size
        self.masking_target = masking_target

        self.mlp = MLP([9 + Nh * 18 + (C - 1) * 9, *self.fcnet_hiddens])
        self.lstm = nn.LSTM(self.embed_dim, self.lstm_size, batch_first=True)

    def forward(self, observation, state_in):
        K = self.max_visible_num_humans
        Ks = self.prediction_steps

        embed, sorted_obs, mask, human_mask, withinview_mask = self.forward_embedding(observation)
        hidden, [h, c] = self.lstm(
            embed, [torch.unsqueeze(state_in[0], 0), torch.unsqueeze(state_in[1], 0)]
        )
        features = torch.cat([embed, hidden], dim=-1)
        state_out = [torch.squeeze(h, 0), torch.squeeze(c, 0)]

        return features, mask, human_mask, withinview_mask, state_out

    def forward_embedding(self, observation):
        B, T, F = observation.shape
        C = self.num_cameras
        Nh = self.max_num_humans
        K = self.max_visible_num_humans

        human_obs = observation[..., 1 + Nh + 9 : 1 + Nh + 9 + Nh * 28]  # shape = (B, T, F)
        human_obs = human_obs.reshape(B, T, Nh, -1)  # shape = (B, T, Nh, f)

        human_reduced_obs = human_obs[..., 4:22]  # shape = (B, T, Nh, f)
        target_obs = human_reduced_obs[..., 0:1, :]  # (B, T, 1, f)
        pedastrain_reduced_obs = human_reduced_obs[..., 1:, :]  # shape = (B, T, Nh - 1, f)

        human_mask = observation[..., 1 : 1 + Nh] != -1  # (B, T, Nh)
        withinview_mask = human_obs[..., -2] != -1  # (B, T, Nh)
        # withinview_mask[..., :] = True

        mask = torch.logical_and(human_mask, withinview_mask)

        if self.masking_target:
            target_obs[~mask[..., 0:1]] = 0.0
        pedastrain_reduced_obs[~mask[..., 1:]] = 0.0
        combined_obs = torch.cat([target_obs, pedastrain_reduced_obs], dim=-2)
        obs_embed = torch.cat(  # shape = (B, T, F)
            [
                observation[..., 1 + Nh : 1 + Nh + 9],
                combined_obs.reshape(B, T, -1),  # shape = (B, T, K * f)
                observation[..., 1 + Nh + 9 + Nh * 28 :],
            ],
            dim=-1,
        )

        embed = self.mlp(obs_embed)

        return embed, combined_obs, mask, human_mask, withinview_mask

    def get_initial_state(self):
        #  View API is supported across all of RLlib.
        # Place hidden states on same device as model.
        h = [
            self.mlp.linear_layers[-1].weight.new(1, self.lstm_size).zero_().squeeze(0),
            self.mlp.linear_layers[-1].weight.new(1, self.lstm_size).zero_().squeeze(0),
        ]
        return h


class MAPPOPartialModel(TorchRNN, nn.Module):
    def __init__(self, obs_space, action_space, num_outputs, model_config, name, **kwargs):
        nn.Module.__init__(self)
        super().__init__(obs_space, action_space, num_outputs, model_config, name)

        self.num_cameras = kwargs['num_cameras']
        self.max_num_humans = kwargs.get('max_num_humans', 7)
        self.num_actions = len(action_space)
        self.obs_size = get_preprocessor(obs_space)(obs_space).size

        self.fcnet_hiddens = kwargs.get('fcnet_hiddens', [128, 128, 128])
        self.actnet_hiddens = kwargs.get('actnet_hiddens', [128])
        self.vfnet_hiddens = kwargs.get('vfnet_hiddens', [128])
        self.lstm_size = kwargs.get('cell_size', 128)
        self.mdn_hiddens = kwargs.get('mdn_hiddens', [128, 128])
        self.mdn_num_gaussians = kwargs.get('mdn_num_gaussians', 16)

        self.prediction_steps = kwargs.get('prediction_steps', 1)
        self.masking_target = kwargs.get('masking_target', True)
        self.observation_sorting = kwargs.get('observation_sorting', False)
        self.merge_back = kwargs.get('merge_back', True)
        self.coordinate_scale = kwargs.get('coordinate_scale', 500.0)

        if not self.observation_sorting:
            self.max_visible_num_humans = self.max_num_humans
        else:
            warnings.warn('observation_sorting is enabled!')
            self.max_visible_num_humans = kwargs.get('max_visible_num_humans', 5)

        self.prediction_loss_coeff = kwargs.get('prediction_loss_coeff', 1.0)
        self.pred_coeff_dict = kwargs.get(
            'pred_coeff_dict',
            {
                'coeff_cam_pred': 1.0,
                'coeff_other_cam_pred': 1.0,
                'coeff_reward_pred': 1.0,
                'coeff_human_pred': 1.0,
                'coeff_obstructor_pred': 0.1,
            },
        )

        self.coeff_cam_pred = self.pred_coeff_dict['coeff_cam_pred']
        self.coeff_human_pred = self.pred_coeff_dict['coeff_human_pred']
        self.coeff_obstructor_pred = self.pred_coeff_dict['coeff_obstructor_pred']
        self.coeff_other_cam_pred = self.pred_coeff_dict['coeff_other_cam_pred']
        self.coeff_reward_pred = self.pred_coeff_dict['coeff_reward_pred']
        # self.coeff_cur_depth_pred = self.pred_coeff_dict.get('coeff_cur_depth_pred', 0.0)
        # self.coeff_next_depth_pred = self.pred_coeff_dict.get('coeff_next_depth_pred', 0.0)

        self.use_prev_action = model_config['lstm_use_prev_action']
        self.use_prev_reward = model_config['lstm_use_prev_reward']

        if isinstance(action_space, Discrete):
            self.action_dim = action_space.n
        elif isinstance(action_space, MultiDiscrete):
            self.action_dim = np.sum(action_space.nvec)
        elif action_space.shape is not None:
            self.action_dim = int(np.product(action_space.shape))
        else:
            self.action_dim = int(len(action_space))

        # Set self.num_outputs to the number of output nodes desired by the
        # caller of this constructor.
        self.num_outputs = num_outputs
        if self.num_outputs is None:
            self.num_outputs = self.action_dim

        num_inputs = self.obs_size
        # Add prev-action/reward nodes to input to LSTM.
        if self.use_prev_action:
            num_inputs += self.action_dim
        if self.use_prev_reward:
            num_inputs += 1

        # Build the Module from fc + LSTM + 2xfc (action + value outs).
        self.embedding = Embedding(
            num_cameras=self.num_cameras,
            max_num_humans=self.max_num_humans,
            max_visible_num_humans=self.max_visible_num_humans,
            prediction_steps=self.prediction_steps,
            mdn_hiddens=self.mdn_hiddens,
            mdn_num_gaussians=self.mdn_num_gaussians,
            fcnet_hiddens=self.fcnet_hiddens,
            lstm_size=self.lstm_size,
            masking_target=self.masking_target,
            observation_sorting=self.observation_sorting,
            merge_back=self.merge_back,
        )

        self.feature_dim = self.embedding.feature_dim
        self.global_feature_dim = (
            self.feature_dim
            + 8 * self.num_cameras
            + (self.num_cameras - 1) * self.num_actions
            + self.max_num_humans * 3
        )  # + self.num_cameras * 12 * 16

        self.action_branch = MLP([self.feature_dim, *self.actnet_hiddens, num_outputs])
        self.value_branch = MLP([self.global_feature_dim, *self.vfnet_hiddens, 1])
        self.camera_prediction_branch = MDN(
            self.feature_dim + self.action_dim,
            # self.obs_size,
            8,
            self.mdn_hiddens,
            self.mdn_num_gaussians,
        )
        self.other_camera_linear = nn.Linear(
            self.feature_dim + self.action_dim, 16 * (self.num_cameras - 1)
        )
        self.other_camera_prediction_branch = MDN(
            16,
            # self.obs_size,
            8,
            self.mdn_hiddens,
            self.mdn_num_gaussians,
        )
        self.reward_prediction_branch = MDN(
            self.feature_dim + self.action_dim,
            1,
            self.mdn_hiddens,
            self.mdn_num_gaussians,
        )
        # self.cur_depth_prediction_branch = MLP([self.feature_dim, *self.mdn_hiddens, 12 * 16])
        # self.next_depth_prediction_branch = MLP([self.feature_dim + self.action_dim, *self.mdn_hiddens, 12 * 16])
        # Holds the current "base" output (before logits layer).
        self._features = None
        self._hidden = None

        # Add prev-a/r to this model's view, if required.
        if model_config['lstm_use_prev_action']:
            self.view_requirements[SampleBatch.PREV_ACTIONS] = ViewRequirement(
                SampleBatch.ACTIONS, space=self.action_space, shift=-1
            )
        if model_config['lstm_use_prev_reward']:
            self.view_requirements[SampleBatch.PREV_REWARDS] = ViewRequirement(
                SampleBatch.REWARDS, shift=-1
            )

        self.view_requirements[SampleBatch.INFOS] = ViewRequirement()

    @override(ModelV2)
    def get_initial_state(self):
        # TODO: (sven): Get rid of `get_initial_state` once Trajectory
        #  View API is supported across all of RLlib.
        # Place hidden states on same device as model.
        return self.embedding.get_initial_state()

    @override(ModelV2)
    def value_function(self):
        assert self._features is not None, 'must call forward() first'

        Nh = self.max_num_humans
        C = self.num_cameras

        try:
            camera_poses = self._raw_gt_observation[..., 1 + Nh + 1 : 1 + Nh + 9]
            camera_poses = camera_poses.reshape((*self._features.shape[:2], -1))
        except AttributeError:
            camera_poses = torch.zeros(size=(*self._features.shape[:2], C * 8))
        camera_poses = camera_poses.to(self._features.device, dtype=self._features.dtype)

        try:
            joint_action = self._joint_action
            teammate_actions = joint_action[..., 1:, :]
            teammate_actions = teammate_actions.reshape((*self._features.shape[:2], -1))
        except AttributeError:
            teammate_actions = torch.ones(
                size=(*self._features.shape[:2], (C - 1) * self.num_actions)
            )
        teammate_actions = teammate_actions.to(self._features.device, dtype=self._features.dtype)

        try:
            gt3d = self._gt3d
            gt3d = gt3d.reshape((*self._features.shape[:2], Nh, 14, 3))
        except AttributeError:
            gt3d = torch.zeros(size=(*self._features.shape[:2], Nh, 14, 3))
        gtcenter = gt3d.mean(dim=-2)
        gtcenter = gtcenter.view(*self._features.shape[:2], -1)
        gtcenter = gtcenter.to(self._features.device, dtype=self._features.dtype)

        # try:
        #     inv_depth = self._inv_depth
        #     inv_depth = inv_depth.reshape((*self._features.shape[:2], -1))
        # except AttributeError:
        #     inv_depth = torch.zeros(size=(*self._features.shape[:2], C * 12 * 16))
        # inv_depth = inv_depth.to(self._features.device, dtype=self._features.dtype)

        features = torch.cat([self._features, camera_poses, teammate_actions, gtcenter], dim=-1)

        return torch.reshape(self.value_branch(features), [-1])

    @override(TorchRNN)
    def forward(
        self,
        input_dict: dict[str, TensorType],
        state: list[TensorType],
        seq_lens: TensorType,
    ) -> tuple[TensorType, list[TensorType]]:
        assert seq_lens is not None
        # Push obs through "unwrapped" net's `forward()` first.
        # wrapped_out, _ = self._wrapped_forward(input_dict, [], None)
        wrapped_out = input_dict['obs_flat']

        # Concat. prev-action/reward if required.
        prev_a_r = []
        if self.model_config['lstm_use_prev_action']:
            if isinstance(self.action_space, (Discrete, MultiDiscrete)):
                prev_a = one_hot(input_dict[SampleBatch.PREV_ACTIONS].float(), self.action_space)
            else:
                prev_a = input_dict[SampleBatch.PREV_ACTIONS].float()
            prev_a_r.append(torch.reshape(prev_a, [-1, self.action_dim]))
        if self.model_config['lstm_use_prev_reward']:
            prev_a_r.append(torch.reshape(input_dict[SampleBatch.PREV_REWARDS].float(), [-1, 1]))

        if prev_a_r:
            wrapped_out = torch.cat([wrapped_out] + prev_a_r, dim=1)

        # Then through our LSTM.
        input_dict['obs_flat'] = wrapped_out

        infos = input_dict[SampleBatch.INFOS]
        if (
            isinstance(infos, np.ndarray)
            and infos.dtype == np.dtype('O')
            and any(isinstance(i, dict) for i in infos)
        ):

            def get_camera_params(info):
                if self.training:
                    try:
                        info = info['prev_info']
                    except BaseException:
                        pass
                try:
                    return info['cam_param_list']
                except BaseException:
                    print(info)
                    with open('debug.txt', mode='a') as file:
                        file.write(f'all: {all([isinstance(i, dict) for i in infos])}\n')
                        file.write(f'training: {input_dict.is_training}\n')
                        file.write(f'len: {len(infos)}\n')
                        file.write(f'seq_lens: {seq_lens}\n')
                        file.write(f'infos: {infos}\n')
                        file.write(f'info: {info}\n\n')
                    raise

            cam_params = self.extract_from_info(
                input_dict, func=get_camera_params, seq_lens=seq_lens
            )
            self._cam_params = cam_params

            def get_joint_action(info):
                if self.training:
                    info = info['prev_info']

                joint_action = info['joint_action']
                index = info['index']
                joint_action = np.roll(joint_action, axis=0, shift=-index)
                return joint_action

            joint_action = self.extract_from_info(
                input_dict, func=get_joint_action, seq_lens=seq_lens
            )
            self._joint_action = joint_action

            def get_raw_gt_observation(info):
                if self.training:
                    info = info['prev_info']

                observation = info['gt_obs_dict']['observation']
                index = info['index']
                observation = np.roll(observation, axis=0, shift=-index)
                return observation

            raw_gt_observation = self.extract_from_info(
                input_dict, func=get_raw_gt_observation, seq_lens=seq_lens
            )
            self._raw_gt_observation = raw_gt_observation

            def get_padded_gt3d(info):
                if self.training:
                    info = info['prev_info']

                gt3d = info['gt3d']
                gt3d = np.pad(
                    gt3d,
                    pad_width=(
                        (0, self.max_num_humans - gt3d.shape[0]),
                        (0, 0),
                        (0, 0),
                    ),
                    mode='constant',
                )
                assert gt3d.shape == (self.max_num_humans, 14, 3)
                return gt3d

            gt3d = self.extract_from_info(input_dict, func=get_padded_gt3d, seq_lens=seq_lens)
            self._gt3d = gt3d

            # def get_inv_depth(info):
            #     if self.training:
            #         info = info["prev_info"]

            #     inv_depth_list = info["inv_depth_list"]
            #     index = info["index"]
            #     inv_depth_list = np.roll(inv_depth_list, axis=0, shift=-index)
            #     return inv_depth_list

            # inv_depth = self.extract_from_info(input_dict,
            #                                    func=get_inv_depth,
            #                                    seq_lens=seq_lens)
            # self._inv_depth = inv_depth

        return super().forward(input_dict, state, seq_lens)

    @override(TorchRNN)
    def forward_rnn(self, inputs, state, seq_lens):
        """Feeds `inputs` (B x T x ..) through the Gru Unit.

        Returns the resulting outputs as a sequence (B x T x ...).
        Values are stored in self._cur_value in simple (B) shape (where B
        contains both the B and T dims!).

        Returns:
            NN Outputs (B x T x ...) as sequence.
            The state batches as a List of two items (c- and h-states).
        """
        (
            self._features,
            self.mask,
            self.human_mask,
            self.withinview_mask,
            self.sort_indices,
            self.target_dist,
            self.obstructor_dists,
            state_out,
        ) = self.embedding(inputs, state)
        action_out = self.action_branch(self._features).clamp(min=-1e8, max=+1e8)
        return action_out, state_out

    @override(ModelV2)
    def custom_loss(self, policy_loss, loss_inputs):
        seq_lens = loss_inputs[SampleBatch.SEQ_LENS].detach().cpu().numpy()
        max_seq_len = self._features.shape[1]
        mask = sequence_mask(
            loss_inputs[SampleBatch.SEQ_LENS],
            max_seq_len,
            time_major=self.is_time_major(),
        )
        # mask = torch.reshape(mask, [-1])
        num_valid = torch.sum(mask)

        B, T = self._features.shape[:2]
        C = self.num_cameras
        Nh = self.max_num_humans
        K = self.max_visible_num_humans
        Ks = self.prediction_steps

        if Ks > 1:
            multi_step_mask = torch.cat(
                [mask[:, : -(Ks - 1)], torch.zeros_like(mask[:, : (Ks - 1)])], dim=1
            )
        else:
            multi_step_mask = mask

        def reduce_mean_valid(t, masks=()):
            if len(masks) == 0:
                return torch.sum(t[mask]) / num_valid
            else:
                masks = [mask, *masks]
                for i, m in enumerate(masks):
                    while m.ndim < t.ndim:
                        m = m.unsqueeze(dim=-1)
                    m = torch.broadcast_to(m, t.size())
                    masks[i] = m

                m = masks[0]
                for mm in masks[1:]:
                    m = torch.logical_and(m, mm)

                if m.any().item():
                    return torch.sum(t[m]) / torch.sum(m)
                else:
                    return t.new(1).zero_()[0]

        next_observation = loss_inputs[SampleBatch.NEXT_OBS]
        if isinstance(self.action_space, (Discrete, MultiDiscrete)):
            action = one_hot(loss_inputs[SampleBatch.ACTIONS].float(), self.action_space)
        else:
            action = loss_inputs[SampleBatch.ACTIONS].float()

        next_observation = next_observation.view((B, T, *next_observation.shape[1:]))
        next_gt_observation = self.extract_from_info(
            loss_inputs,
            func=lambda info: info['gt_obs_dict']['observation'][info['index']],
            seq_lens=seq_lens,
        ).float()
        action = action.view((B, T, -1))
        next_observation = next_observation.to(policy_loss[0].device)
        next_gt_observation = next_gt_observation.to(policy_loss[0].device)

        action = action.to(policy_loss[0].device)

        next_camera_pose = next_observation[..., 1 + Nh + 1 : 1 + Nh + 9]
        next_human_pose = next_gt_observation[..., 1 + Nh + 9 + 4 : 1 + Nh + 9 + 6]
        next_obstructor_pose = torch.stack(
            [
                next_gt_observation[..., 1 + Nh + 9 + h * 28 + 4 : 1 + Nh + 9 + h * 28 + 6]
                for h in range(1, Nh)
            ],
            dim=-2,
        )
        next_other_camera_pose = torch.stack(
            [
                next_observation[
                    ...,
                    1 + Nh + 9 + Nh * 28 + 9 * c + 1 : 1 + Nh + 9 + Nh * 28 + 9 * c + 9,
                ]
                for c in range(C - 1)
            ],
            dim=-2,
        )

        next_human_pose = next_human_pose / self.coordinate_scale
        next_obstructor_pose = next_obstructor_pose / self.coordinate_scale

        next_human_pose_padded = torch.cat(
            [next_human_pose, torch.zeros_like(next_human_pose[:, : Ks - 1])], dim=1
        )

        next_obstructor_pose_padded = torch.cat(
            [next_obstructor_pose, torch.zeros_like(next_obstructor_pose[:, : Ks - 1])],
            dim=1,
        )

        if self.observation_sorting:
            indices = self.sort_indices.unsqueeze(dim=-1).repeat(
                1, 1, 1, next_obstructor_pose.shape[-1]
            )
            next_obstructor_pose = torch.gather(next_obstructor_pose, dim=-2, index=indices)
            next_obstructor_pose[..., K - 1 :, :] = 0.0

            sorted_mask = torch.gather(self.mask[..., 1:], dim=-1, index=self.sort_indices)
            sorted_mask[..., K - 1 :] = False
        else:
            sorted_mask = self.mask[..., 1:]

        obstructor_log_probs = []
        for p, dists in enumerate(self.obstructor_dists):
            obstructor_log_prob = 0
            for offset, dist in enumerate(dists):
                obstructor_log_prob = (
                    obstructor_log_prob
                    + dist.log_prob(next_obstructor_pose_padded[:, offset : offset + T, p])[0]
                )
            obstructor_log_probs.append(obstructor_log_prob)
        obstructor_log_prob = torch.stack(obstructor_log_probs, dim=-1)

        rewards = self.extract_from_info(
            loss_inputs, func=lambda info: (info['team_reward'],)
        ).float()
        rewards = rewards.to(policy_loss[0].device)

        features = torch.cat([self._features, action], dim=-1)
        camera_log_prob, camera_mdn_out = self.camera_prediction_branch.log_prob(
            features, next_camera_pose, epsilon=1e-6
        )

        human_log_prob = 0
        for offset, dist in enumerate(self.target_dist):
            human_log_prob = (
                human_log_prob + dist.log_prob(next_human_pose_padded[:, offset : offset + T])[0]
            )

        other_camera_features = self.other_camera_linear(features)
        other_camera_features = other_camera_features.reshape((B, T, C - 1, -1))
        (
            other_camera_log_prob,
            other_camera_mdn_out,
        ) = self.other_camera_prediction_branch.log_prob(
            other_camera_features, next_other_camera_pose, epsilon=1e-6
        )

        reward_log_prob, reward_mdn_out = self.reward_prediction_branch.log_prob(
            features, rewards, epsilon=1e-6
        )

        camera_prediction_loss = -reduce_mean_valid(camera_log_prob)
        human_prediction_loss = -reduce_mean_valid(human_log_prob, masks=(multi_step_mask,))
        obstructor_prediction_loss = -reduce_mean_valid(
            obstructor_log_prob, masks=(multi_step_mask, sorted_mask[..., : K - 1])
        )
        other_camera_prediction_loss = -reduce_mean_valid(other_camera_log_prob)
        reward_prediction_loss = -reduce_mean_valid(reward_log_prob)

        # if self.coeff_cur_depth_pred > 0.0:
        #     cur_gt_inv_depth = self.extract_from_info(
        #         loss_inputs,
        #         func=lambda info: info['prev_info']["inv_depth_list"][info['index']].ravel(),
        #         seq_lens=seq_lens
        #     ).float().to(policy_loss[0].device)

        #     pred_cur_inv_depth = self.cur_depth_prediction_branch(self._features)
        #     cur_depth_prediction_loss = reduce_mean_valid(F.mse_loss(pred_cur_inv_depth, cur_gt_inv_depth,
        #                                                              reduction='none').mean(dim=-1))
        # else:
        #     cur_depth_prediction_loss = torch.zeros_like(policy_loss[0])

        # if self.coeff_next_depth_pred > 0.0:
        #     next_gt_inv_depth = self.extract_from_info(
        #         loss_inputs,
        #         func=lambda info: info["inv_depth_list"][info['index']].ravel(),
        #         seq_lens=seq_lens
        #     ).float().to(policy_loss[0].device)

        #     pred_next_inv_depth = self.next_depth_prediction_branch(features)
        #     next_depth_prediction_loss = reduce_mean_valid(F.mse_loss(pred_next_inv_depth, next_gt_inv_depth,
        #                                                               reduction='none').mean(dim=-1))
        # else:
        #     next_depth_prediction_loss = torch.zeros_like(policy_loss[0])

        prediction_loss = (
            self.coeff_cam_pred * camera_prediction_loss
            + self.coeff_human_pred * human_prediction_loss
            + self.coeff_obstructor_pred * obstructor_prediction_loss
            + self.coeff_other_cam_pred * other_camera_prediction_loss
            + self.coeff_reward_pred * reward_prediction_loss
        )  # \
        # + self.coeff_cur_depth_pred * cur_depth_prediction_loss \
        # + self.coeff_next_depth_pred * next_depth_prediction_loss

        additional_loss = self.prediction_loss_coeff * prediction_loss

        self.camera_prediction_loss_metric = camera_prediction_loss.item()
        self.human_prediction_loss_metric = human_prediction_loss.item()
        self.obstructor_prediction_loss_metric = obstructor_prediction_loss.item()
        self.other_camera_prediction_loss_metric = other_camera_prediction_loss.item()
        self.reward_prediction_loss_metric = reward_prediction_loss.item()
        # self.cur_depth_prediction_loss_metric = cur_depth_prediction_loss.item()
        # self.next_depth_prediction_loss_metric = next_depth_prediction_loss.item()
        self.prediction_loss_metric = prediction_loss.item()
        self.additional_loss_metric = additional_loss.item()
        self.policy_loss_metric = np.mean([loss.item() for loss in policy_loss])

        return [loss_ + additional_loss for loss_ in policy_loss]

    def metrics(self):
        return {
            'policy_loss': self.policy_loss_metric,
            'camera_prediction_loss': self.camera_prediction_loss_metric,
            'human_prediction_loss': self.human_prediction_loss_metric,
            'obstructor_prediction_loss': self.obstructor_prediction_loss_metric,
            'other_camera_prediction_loss': self.other_camera_prediction_loss_metric,
            'reward_prediction_loss': self.reward_prediction_loss_metric,
            # "cur_depth_prediction_loss": self.cur_depth_prediction_loss_metric,
            # "next_depth_prediction_loss": self.next_depth_prediction_loss_metric,
            'prediction_loss': self.prediction_loss_metric,
            'additional_loss': self.additional_loss_metric,
            'prediction_loss_coeff': self.prediction_loss_coeff,
        }

    def extract_from_info(
        self,
        input_dict,
        func,
        seq_lens=None,
        to_tensor=torch.tensor,
        time_dimension=True,
    ):
        if seq_lens is None:
            seq_lens = input_dict[SampleBatch.SEQ_LENS].detach().cpu().numpy()
        max_seq_len = input_dict[SampleBatch.OBS].shape[0] // seq_lens.shape[0]

        valid_data = []
        data_flattened = []
        offset = 0
        have_tensor = False
        have_invalid = False
        for seq_len in seq_lens:
            for info in input_dict[SampleBatch.INFOS][offset : offset + seq_len]:
                if isinstance(info, dict):
                    data = func(info)
                    valid_data.append(data)
                else:
                    data = None
                    have_invalid = True
                data_flattened.append(data)
                have_tensor = have_tensor or isinstance(data, torch.Tensor)
            data_flattened += [data_flattened[-1]] * (max_seq_len - seq_len)
            offset += seq_len

        if have_invalid:
            mean = np.mean(list(map(np.array, valid_data)), axis=0)
            data_flattened = [(data if data is not None else mean) for data in data_flattened]
        if not have_tensor:
            data_flattened = np.array(data_flattened)
        data_flattened = to_tensor(data_flattened)

        if time_dimension:
            data_flattened = add_time_dimension(
                data_flattened,
                max_seq_len=max_seq_len,
                framework='torch',
                time_major=self.is_time_major(),
            )

        return data_flattened
