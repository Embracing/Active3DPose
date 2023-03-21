import numpy as np
import numpy.ma as ma


# class Reward:
#     '''
#     define different types of reward functions
#     '''

#     def __init__(self):
#         pass

#     def reward_error3d(self, pred3d, gt3d, hand_coef=2):
#         """
#         pred3d: [j, 3]
#         """
#         error = np.linalg.norm(pred3d - gt3d, axis=1)  # [j]
#         body_error = error[wholebody_kpinfo_dict['body_idx_after_mapping']]
#         lhand_error = error[wholebody_kpinfo_dict['lhand_idx_after_mapping']]
#         rhand_error = error[wholebody_kpinfo_dict['rhand_idx_after_mapping']]

#         reward_body = -body_error.mean()
#         reward_lhand = -lhand_error.mean()
#         reward_rhand = -rhand_error.mean()
#         reward = reward_body + hand_coef * (reward_lhand + reward_rhand)

#         reward_dict = {
#             'reward_body': reward_body,
#             'reward_lhand': reward_lhand,
#             'reward_rhand': reward_rhand
#         }
#         # reward = -(body_error.mean() + hand_coef * (lhand_error.mean() + rhand_error.mean()))
#         return reward, reward_dict

#     def reward_score(self, pred2d_score, action, vis_thresh=0.5):
#         """
#         Reward  visible selectioin.
#         pred2d_score: [C, j, 1]
#         vis_thresh: TODO: maybe smaller or larger?
#         Return: [C, P] (P=3 for body, lhand, rhand)
#         """
#         pred2d_score = pred2d_score.squeeze(-1)  # [C, j]
#         # pred2d_vis_score = pred2d_score - vis_thresh  # [C, j]

#         # TODO: maybe more aggresive
#         body_score = pred2d_score[:, wholebody_kpinfo_dict['body_idx_after_mapping']]  # [C, jb]
#         lhand_score = pred2d_score[:, wholebody_kpinfo_dict['lhand_idx_after_mapping']]  # [C, jl]
#         rhand_score = pred2d_score[:, wholebody_kpinfo_dict['rhand_idx_after_mapping']]  # [C, j2]

#         body_score = body_score.mean(axis=1)  # [C]
#         lhand_score = lhand_score.mean(axis=1)  # [C]
#         rhand_score = rhand_score.mean(axis=1)  # [C]
#         body_vis_score = body_score - vis_thresh  # [C]
#         lhand_vis_score = lhand_score - vis_thresh  # [C]
#         rhand_vis_score = rhand_score - vis_thresh  # [C]

#         vis_score_matrix = np.stack([body_vis_score, lhand_vis_score, rhand_vis_score], axis=1)  # [C, 3]
#         reward_vis_matrix = vis_score_matrix * action  # [C, 3]
#         # reward_vis_vector = reward_vis_matrix.sum(axis=1)  # [C]

#         # calculate reward for punishing the situation that one part is selected by less than 2 views.
#         score_matrix = np.stack([body_score, lhand_score, rhand_score], axis=1)  # [C, 3]
#         count = action.sum(axis=0)  # [3]
#         reward_selection = -np.array(count < 2, dtype=np.int32)  # [3]
#         reward_selection_matrix = score_matrix * (1 - action) * reward_selection[None, :]  # [C, 3]
#         # reward_selection_vector = reward_selection_matrix.sum(axis=1)  # [C]

#         return reward_vis_matrix, reward_selection_matrix

#     def reward_multiview(self, action, coef=3):
#         """
#         action: [C, P]
#         Negative reward if a part is selected by less than 2 views.
#         Return: scalar

#         TODO: how to deliver a reward to each camera, not a scalar.
#         """
#         count = action.sum(axis=0)  # [P]
#         reward = -coef * (count < 2)  # [P]
#         reward = reward.sum()
#         return reward


class RewardMultiPerson:
    """
    define different types of reward functions
    """

    def __init__(self, reward_3d_func='l2', env_config=None):
        self.reward_3d_func = reward_3d_func
        self.env_config = env_config
        # self.map_config = map_config

    def set_offset(self, offset):
        """
        Set Ground Center Offset

        offset: ndarray, [3]
        """
        self.offset = offset

    def reward_error3d(self, pred3d, gt3d):
        """
        gt3d: [N, j, 3]
        pred3d: [N_max, j, 3], or [N, j, 3] also supported
        According to num of cameras and num of humans
        2 camera, for 1 human, under 5 is positive
        2 camera, for 7 humans in line, under 15 is positive
        """

        assert len(pred3d) >= len(gt3d)
        pred3d = pred3d[: len(gt3d), ...]  # [N, j, 3]

        mpjpe = np.linalg.norm(pred3d - gt3d, axis=-1)  # [N, j]

        # add offset for zero pred 3d joints
        target_pred3d = pred3d[0].copy()  # [j, 3]
        target_gt3d = gt3d[0]  # [j, 3]
        zero_mask = ~np.any(target_pred3d, axis=-1)  # [j]
        target_pred3d[zero_mask] += self.offset  # [j, 3]
        target_mpjpe = np.linalg.norm(target_pred3d - target_gt3d, axis=-1)  # [j]

        if self.reward_3d_func == 'l2':
            error = mpjpe.copy()  # [N, j]
            error = np.clip(error, None, 100)  # max joints error == 100
            # make reward half positive and half negative
            # TODO: Adaptively tighten this bound?
            if len(gt3d) == 1:
                critical = 5
            elif len(gt3d) == 7:
                critical = 15
            else:
                critical = 10
            critical = 0
            error = error - critical  # [N, j]
        elif self.reward_3d_func == 'gemen':
            gemen_factor = 0.2
            gemen_loss = np.power(mpjpe * gemen_factor, 2)  # [N, j]
            error = 2 * gemen_loss / (gemen_loss + 4)  # [N, j], 0 ~ 2
            # TODO: add critical here
            error = error - 1  # [N, j], -1 ~ 1
        else:
            raise NotImplementedError

        error = error[0]  # only person-0
        error = np.mean(error)

        reward = -error

        ex_mpjpe_h0 = mpjpe[0].copy()  # [j]
        zero_mask = ~np.any(pred3d[0], axis=-1)  # [j]

        # assign a big number to exclude PCK3D caculation
        ex_mpjpe_h0[zero_mask] = 101  # [j]

        reward_dict = {
            'reward_3d': reward,  # for person #0
            'mpjpe_3d': target_mpjpe.mean(),  # for person #0
            'reward_peak': (mpjpe[0].mean() > 25).astype(np.float32) * -1,
        }

        pck_intervals = range(5, 155, 5)
        # pck3d
        reward_dict.update({f'pck3d_{m}': (mpjpe[0] * 10 <= m).mean() for m in pck_intervals})
        reward_dict.update(
            {'avg_pck3d': np.mean([reward_dict[f'pck3d_{m}'] for m in pck_intervals])}
        )

        # exclude pck3d
        reward_dict.update({f'ex_pck3d_{m}': (ex_mpjpe_h0 * 10 <= m).mean() for m in pck_intervals})
        reward_dict.update(
            {'avg_ex_pck3d': np.mean([reward_dict[f'ex_pck3d_{m}'] for m in pck_intervals])}
        )

        return reward, reward_dict

    def reward_error3d_soft(self, pred3d, gt3d, conf):
        """
        gt3d: [N, j, 3]
        pred3d: [N_max, j, 3]
        conf: []
        """
        return 0, dict()

    # def reward_visibility_indv(self, curr_pred2d_scores, prev_pred2d_scores, threshold=0.5):
    #     """
    #     curr_pred2d_scores: [C, N_max, j, 1]
    #     prev_pred2d_scores: [C, N_max, j, 1]
    #     """
    #     N_max, num_joints = curr_pred2d_scores.shape[1:3]
    #     score_2 = 100.
    #     score_3 = 10.
    #     curr_visibility = curr_pred2d_scores.reshape(len(curr_pred2d_scores), -1)  # [C, N_max*j]
    #     curr_visibility  = np.array(curr_visibility > threshold, dtype=np.int32)  # [C, N_max*j]
    #     curr_total_vis = np.sum(curr_visibility, axis=0)  # [N_max*j]

    #     prev_visibility = prev_pred2d_scores.reshape(len(prev_pred2d_scores), -1)  # [C, N_max*j]
    #     prev_visibility  = np.array(prev_visibility > threshold, dtype=np.int32)  # [C, N_max*j]
    #     prev_total_vis = np.sum(prev_visibility, axis=0)  # [N_max*j]

    #     delta = curr_visibility - prev_visibility  # [C, N_max*j] 1, 0, -1
    #     delta_total = curr_total_vis - prev_total_vis  # [N_max*j]

    #     # if total increase
    #     # find items that total reach 2,
    #     reward_positive = 0
    #     sel_2 = np.logical_and.reduce((curr_total_vis >= 2, prev_total_vis < 2))  # [N_max*j]
    #     reward_positive = np.sum(delta[:, sel_2] > 0, axis=1) * score_2  # [C]
    #     sel_3 = np.logical_and.reduce((delta_total > 0, curr_total_vis >= 2, prev_total_vis >= 2))  # [N_max*j]
    #     reward_positive += np.sum(delta[:, sel_3] > 0, axis=1) * score_3  # [C]

    #     reward_negative = 0
    #     sel_2 = np.logical_and.reduce((prev_total_vis >= 2, curr_total_vis < 2))  # [N_max*j]
    #     reward_negative = np.sum(delta[:, sel_2] < 0, axis=1) * score_2  # [C]
    #     sel_3 = np.logical_and.reduce((delta_total < 0, prev_total_vis >= 2, curr_total_vis >= 2))   # [N_max*j]
    #     reward_negative += np.sum(delta[:, sel_3] > 0, axis=1) * score_3  # [C]

    #     reward = (reward_positive + reward_negative) / (N_max * num_joints)
    #     reward_dict = {
    #         'reward_vis': reward  # [C]
    #     }

    #     return reward, reward_dict

    def reward_visibility_indv(self, pred2d_scores, threshold=0.5):
        """
        pred2d_scores: [C, N_max, j, 1]
        """
        joints_score = 2
        # TODO: change back
        pred2d_scores = pred2d_scores[:, [0], ...]
        pred2d_scores = pred2d_scores.squeeze(-1)  # [C, N_max, j]
        visibility = pred2d_scores > threshold  # [C, N_max, j]
        joints_covered_num = visibility.sum(axis=0, keepdims=True)  # [1, N_max, j]
        can_be_rec_joints = joints_covered_num >= 2  # [1, N_max, j]

        assigned_matrix = np.logical_and(can_be_rec_joints, visibility)  # [C, N_max, j]

        # only cameras with conf > threshold participate in reward assignments
        assigned_weights = pred2d_scores * assigned_matrix.astype(np.int32)  # [C, N_max, j]
        np.divide(
            assigned_weights,
            assigned_weights.sum(axis=0, keepdims=True),
            out=assigned_weights,
            where=assigned_weights.sum(axis=0, keepdims=True) != 0,
        )  # [C, N_max, j]

        #     assigned_scores = np.divide(
        #         assigned_matrix.astype(np.int32) * joints_score,
        #         joints_covered_num,
        #         where = can_be_rec_joints,
        #     )  # [C, N_max, j]
        assigned_scores = assigned_matrix.astype(np.int32) * joints_score * assigned_weights
        assigned_scores_per_camera = assigned_scores.mean(axis=(1, 2))  # [C]

        # assigned_scores_per_camera = assigned_scores.sum(axis=(1, 2))  # [C]
        # assigned_scores_per_camera = assigned_scores_per_camera / (self.controller.num_humans * self.num_joints)

        # Emprical value
        # critical_error = 0.65

        reward = assigned_scores_per_camera
        reward_dict = {'reward_vis': reward}

        return reward, reward_dict

    def reward_bb2d_iot_indv(self, bb2d_IoT, occ_flag):
        """
        TODO: make sure input is gt observation
        bb2d_IoT: [C, N, 1]
        occ_flag: [C, N, 1], 1:occ, 0: not occ
        """
        other_bb = bb2d_IoT[:, 1:, 0]  # [C, N-1]
        other_occ_flag = occ_flag[:, 1:, 0]  # [C, N-1]
        other_occ_flag = other_occ_flag.astype(np.float32)  # [C, N-1]

        other_occ_iot = other_bb * other_occ_flag  # [C, N-1]
        # TODO: more robust method, robust to num_humans?
        reward = -np.sum(other_occ_iot, axis=-1)  # [C]

        reward_dict = {'reward_iot': reward}
        return reward, reward_dict

    def reward_camera_state_indv(self, cam_param_list, lower_boundary, higher_boundary):
        """
        Punish camera when it exceeds boundary
        cam_param_list: a list of [C], x,y,z,pitch,yaw,roll,fov
        lower_bound: list of 3,
        higher_bound: list of 3,
        """
        cam_state = np.asarray(cam_param_list)  # [C, 7]
        cam_location = cam_state[:, :3]  # [C, 3]
        lower_boundary = np.asarray(lower_boundary)  # [3]
        higher_boundary = np.asarray(higher_boundary)  # [3]
        exceeds_boundary = np.logical_or(
            cam_location < lower_boundary[None, :],
            cam_location > higher_boundary[None, :],
        )  # [C, 3]

        exceeds_value_lower = np.maximum(lower_boundary[None, :] - cam_location, 0)
        exceeds_value_higher = np.maximum(cam_location - higher_boundary[None, :], 0)
        exceeds_value = np.maximum(exceeds_value_lower, exceeds_value_higher)  # [C, 3], >=0
        # reward = np.sum(-np.exp(np.power(exceeds_value, 0.1)) + 1, axis=-1)  # [C]
        reward = np.sum(-exceeds_value, axis=-1)

        reward_dict = {'reward_camera_state': reward}  # [C]

        return reward, reward_dict

    def reward_centering(self, camera_model_list, gt3d, tracking_matrix):
        """
        Individual reward
        This func takes the angle between the cam optical and the target as the penalty.

        Params:
            camera_model_list: a list of [C] cam models
            gt3d: [N, j, 3]
            tracking_matrix: [C, N], C camera, N persons, assign targets to cams.

        Return:
            [C] array, between [-1 - cos(bar_angle), 0]
        """

        reward_list = []
        for view_assignment, CamModel in zip(tracking_matrix, camera_model_list):
            assigned_gt3d = gt3d[view_assignment, ...]  # [S, j, 3]
            if len(assigned_gt3d) == 0:
                reward_list.append(0)  # no targets assigned to this camera
            else:
                target_center = assigned_gt3d.mean(axis=(0, 1))  # [3]
                target_center_in_camframe = CamModel.world_to_cam(target_center).squeeze(0)  # [3]
                norm = np.linalg.norm(target_center_in_camframe)
                if norm != 0:
                    normed_direction = target_center_in_camframe / norm
                else:
                    reward_list.append(0)
                    continue
                cos_angle = np.dot(
                    np.array((0, 0, 1), dtype=np.float32),
                    normed_direction,
                )
                # tolearance: < 30 degree == 0, > 30 penalty
                # TODO: tolerance bar varies according to the distance to target
                reward = min(0, cos_angle - np.cos(30 / 180 * np.pi))
                reward_list.append(reward)

        reward = np.array(reward_list)  # [C]
        reward_dict = {'reward_centering': reward}

        return reward, reward_dict

    def reward_obstruction(self, camera_model_list, gt3d, tracking_matrix, alpha=400, sigma=300):
        """
        Individual reward
        This func takes the angle between the cam optical and the target as the penalty.

        Params:
            camera_model_list: a list of [C] cam models
            gt3d: [N, j, 3]
            tracking_matrix: [C, N], C camera, N persons, assign targets to cams.

        Return:
            [C] array, between [-1 - cos(bar_angle), 0]
        """

        reward_list = []
        for view_assignment, CamModel in zip(tracking_matrix, camera_model_list):
            assigned_gt3d = gt3d[view_assignment, ...]  # [S, j, 3]
            if len(assigned_gt3d) == 0:
                reward_list.append(0)  # no targets assigned to this camera
            else:
                cx, cy, cz = CamModel.x, CamModel.y, CamModel.z
                tx, ty, tz = target_center = assigned_gt3d.mean(axis=(0, 1))  # [3]
                rewards = []
                for h, flag in enumerate(view_assignment):
                    if flag:
                        continue
                    ox, oy, oz = obstructor = gt3d[h, ...].mean(
                        axis=0
                    )  # [S, j, 3] -> [j, 3] -> mean -> [3]
                    to = np.array([ox - tx, oy - ty])
                    oc = np.array([cx - ox, cy - oy])
                    if np.linalg.norm(to) == 0:
                        continue
                    dot = np.dot(to, oc)
                    if dot >= 0:
                        d2 = np.square(np.linalg.norm(oc)) - np.square(dot / np.linalg.norm(to))
                        d = np.sqrt(np.maximum(d2, 0))
                    else:
                        d = np.linalg.norm(oc)
                    # rewards.append(np.tanh(d / alpha))
                    rewards.append(1.0 - 1.0 / (np.square(d / alpha) + 1.0))
                if len(rewards) > 0:
                    reward = np.min(rewards, axis=0)
                else:
                    reward = 0

                reward_list.append(reward)

        reward = np.array(reward_list)  # [C]
        reward_dict = {'reward_obstruction': reward}

        return reward, reward_dict

    def reward_distance(self, gt3d, cam_param_list):
        """
        Individual reward
        Keep cameras at a distance from target

        gt3d: [N, J, 3]
        cam_param_list: a list of [C], x,y,z,pitch,yaw,roll,fov
        """

        target = gt3d[0].mean(axis=0)  # [3]
        num_cameras = len(cam_param_list)

        # TODO: single-target
        # trapezoid shape reward
        reward = np.zeros((num_cameras,), dtype=np.float32)
        hh, ll = 300, 150

        C2T_euclideans = np.linalg.norm(
            [param[:3] - target for param in cam_param_list], axis=1
        )  # [C]

        for c in range(num_cameras):
            if C2T_euclideans[c] <= ll:
                reward[c] += max(1 - 5 * (1 - C2T_euclideans[c] / ll), -1)
            elif C2T_euclideans[c] >= hh:
                reward[c] += max(1 - 5 * ((C2T_euclideans[c] - hh) / hh), -1)
            else:
                reward[c] += 1.0

        reward_dict = {'reward_distance': reward}

        return reward, reward_dict

    def reward_anti_collision(self, cam_param_list, gt3d):
        """
        Individual reward
        Collision avoidances among cameras agents
        Currently no judging between whose fault is it, the intruder and the victim both receive penalties.

        cam_param_list: a list of [C], x,y,z,pitch,yaw,roll,fov
        """

        coeff_AC, coeff_h = 5.0, 0.5

        # Anti-collision
        num_cameras = len(cam_param_list)
        reward = np.zeros((num_cameras,), dtype=np.float32)
        COLLISION_THRESHOLD = 80

        for i, cam in enumerate(cam_param_list):
            num_collisions = (
                np.linalg.norm([cam[:3] - target.mean(axis=0) for target in gt3d], axis=1)
                < COLLISION_THRESHOLD
            ).sum()
            reward[i] = -1.0 * num_collisions * coeff_AC

        # Height reward
        z_LL, z_HH = (
            self.env_config['lower_bound_for_camera'][-1],
            self.env_config['higher_bound_for_camera'][-1],
        )
        for i, cam in enumerate(cam_param_list):
            reward[i] += coeff_h * np.exp(-5 * (cam[2] - z_LL) / (z_HH - z_LL))

        reward_dict = {'reward_anti_collision': reward}

        return reward, reward_dict
