import itertools
import random
import time

import numpy as np
from pymvg.camera_model import CameraModel
from pymvg.multi_camera_system import MultiCameraSystem


# from multiviews.cameras import unfold_camera_param


# def build_multi_camera_system(cameras, no_distortion=True):
#     """
#     Build a multi-camera system with pymvg package for triangulation

#     Args:
#         cameras: list of camera parameters
#     Returns:
#         cams_system: a multi-cameras system
#     """
#     pymvg_cameras = []
#     for (name, camera) in cameras:
#         R, T, f, c, k, p = unfold_camera_param(camera, avg_f=False)
#         camera_matrix = np.array(
#             [[f[0], 0, c[0]], [0, f[1], c[1]], [0, 0, 1]], dtype=float)
#         proj_matrix = np.zeros((3, 4))
#         proj_matrix[:3, :3] = camera_matrix
#         distortion = np.array([k[0], k[1], p[0], p[1], k[2]])
#         distortion.shape = (5,)
#         T = -np.matmul(R, T)
#         M = camera_matrix.dot(np.concatenate((R, T), axis=1))
#         camera = CameraModel.load_camera_from_M(
#             M, name=name, distortion_coefficients=None if no_distortion else distortion)
#         if not no_distortion:
#             camera.distortion = distortion  # bug with pymvg
#         pymvg_cameras.append(camera)
#     return MultiCameraSystem(pymvg_cameras)


def build_multi_camera_system(camera_list, no_distortion=True):
    """
    Build a multi-camera system with pymvg package for triangulation

    Args:
        camera_list: list of (name, camera_obj)
    Returns:
        cams_system: a multi-cameras system
    """
    distortion = None

    pymvg_cameras = []
    for name, camera in camera_list:
        M = camera.get_intrinsic() @ camera.get_extrinsic(homo=False)
        camera_model = CameraModel.load_camera_from_M(
            M, name=name, distortion_coefficients=None if no_distortion else distortion
        )
        if not no_distortion:
            camera_model.distortion = distortion  # bug with pymvg
        pymvg_cameras.append(camera_model)
    return MultiCameraSystem(pymvg_cameras)


def triangulate_one_point(camera_system, points_2d_set):
    """
    Triangulate 3d point in world coordinates with multi-view 2d points

    Args:
        camera_system: pymvg camera system
        points_2d_set: list of structure (camera_name, point2d)
    Returns:
        points_3d: 3x1 point in world coordinates
    """
    points_3d = camera_system.find3d(points_2d_set, undistort=False)
    return points_3d


def triangulate_poses(camera_objs, poses2d, joints_vis=None, no_distortion=True, nviews=4):
    """
    Triangulate 3d points in world coordinates of multi-view 2d poses
    by interatively calling $triangulate_one_point$

    Args:
        camera_objs: [N*C] a list of camera parameters, each corresponding to
                       one prediction in poses2d
        poses2d: [N*C, j, 2], [human1_view1, human1_view2,..., human2_view1, human2_view2,...]
        joints_vis: [N*C, j], only visible joints participate in triangulation
    Returns:
        poses3d: ndarray of shape N x j x 3
    """
    njoints = poses2d.shape[1]
    ninstances = len(camera_objs) // nviews
    if joints_vis is not None:
        assert np.all(joints_vis.shape == poses2d.shape[:2])
    else:
        joints_vis = np.ones((poses2d.shape[0], poses2d.shape[1]))

    poses3d = []
    for i in range(ninstances):
        camera_list = []
        for j in range(nviews):
            camera_name = f'camera_{j}'
            camera_list.append((camera_name, camera_objs[i * nviews + j]))
        # start = time.time()
        camera_system = build_multi_camera_system(camera_list, no_distortion)
        # print(f'build time: {time.time() - start}')

        pose3d = np.zeros((njoints, 3))
        for k in range(njoints):
            points_2d_set = []

            for j in range(nviews):
                if joints_vis[i * nviews + j, k]:
                    camera_name = f'camera_{j}'
                    points_2d = poses2d[i * nviews + j, k, :]
                    points_2d_set.append((camera_name, points_2d))
            if len(points_2d_set) < 2:
                continue
            pose3d[k, :] = triangulate_one_point(camera_system, points_2d_set).T
        poses3d.append(pose3d)
    return np.array(poses3d)


def ransac(poses2d, camera_objs, joints_vis, config, nviews=4):
    """
    An group is accepted only if support inliers are not less
    than config.PSEUDO_LABEL.NUM_INLIERS, i.e. num of Trues
    in a 4-view group is not less than config.PSEUDO_LABEL.NUM_INLIERS
    Param:
        poses2d: [N, 16, 2]
        camera_objs: a list of [N]
        joints_vis: [N, 16], only visible joints participate in triangulation
    Return:
        res_vis: [N, 16]
    """
    njoints = poses2d.shape[1]
    ninstances = len(camera_objs) // nviews

    res_vis = np.zeros_like(joints_vis)
    for i in range(ninstances):
        camera_list = []
        for j in range(nviews):
            camera_name = f'camera_{j}'
            camera_list.append((camera_name, camera_objs[i * nviews + j]))
        camera_system = build_multi_camera_system(camera_list, config.DATASET.NO_DISTORTION)

        for k in range(njoints):
            points_2d_set = []

            for j in range(nviews):
                camera_name = f'camera_{j}'
                # select out visible points from all 4 views
                if joints_vis[i * nviews + j, k]:
                    points_2d = poses2d[i * nviews + j, k, :]
                    points_2d_set.append((camera_name, points_2d))

            # points < 2, invalid instance, abandon samples of 1 view
            if len(points_2d_set) < 2:
                continue

            best_inliers = []
            best_error = 10000
            for points_pair in itertools.combinations(points_2d_set, 2):
                point_3d = triangulate_one_point(camera_system, list(points_pair)).T
                in_thre = []
                mean_error = 0
                for j in range(nviews):
                    point_2d_proj = camera_system.find2d(f'camera_{j}', point_3d)
                    error = np.linalg.norm(point_2d_proj - poses2d[i * nviews + j, k, :])
                    if error < config.PSEUDO_LABEL.REPROJ_THRE:
                        in_thre.append(j)
                        mean_error += error
                num_inliers = len(in_thre)
                if num_inliers < config.PSEUDO_LABEL.NUM_INLIERS:
                    continue
                mean_error /= num_inliers
                # update best candidate
                if num_inliers > len(best_inliers):
                    best_inliers = in_thre
                    best_error = mean_error
                elif num_inliers == len(best_inliers):
                    if mean_error < best_error:
                        best_inliers = in_thre
                        best_error = mean_error
            for idx_view in best_inliers:
                res_vis[i * nviews + idx_view, k] = 1
    return res_vis


def reproject_poses(poses2d, camera_objs, joints_vis, no_distortion=True, nviews=4):
    """
    Triangulate 3d points in world coordinates of multi-view 2d poses
    by interatively calling $triangulate_one_point$

    Args:
        camera_objs: a list of camera objectss, each corresponding to
                       one prediction in poses2d
        poses2d: [N, k, 2], len(cameras) == N
        joints_vis: [N, k], only visible joints participate in triangulatioin
    Returns:
        proj_2d: ndarray of shape [N, k, 2]
        res_vis: [N, k]
    """
    njoints = poses2d.shape[1]
    ninstances = len(camera_objs) // nviews
    assert np.all(joints_vis.shape == poses2d.shape[:2])
    proj_2d = np.zeros_like(poses2d)  # [N, 16, 2]
    res_vis = np.zeros_like(joints_vis)

    for i in range(ninstances):
        camera_list = []
        for j in range(nviews):
            camera_name = f'camera_{j}'
            camera_list.append((camera_name, camera_objs[i * nviews + j]))
        camera_system = build_multi_camera_system(camera_list, no_distortion)

        for k in range(njoints):
            points_2d_set = []

            for j in range(nviews):
                if joints_vis[i * nviews + j, k]:
                    camera_name = f'camera_{j}'
                    points_2d = poses2d[i * nviews + j, k, :]
                    points_2d_set.append((camera_name, points_2d))
            if len(points_2d_set) < 2:
                continue
            point_3d = triangulate_one_point(camera_system, points_2d_set).T

            for j in range(nviews):
                point_2d_proj = camera_system.find2d(f'camera_{j}', point_3d)
                proj_2d[i * nviews + j, k, :] = point_2d_proj
                res_vis[i * nviews + j, k] = 1
    return proj_2d, res_vis


def fast_triangulate(camera_objs, poses2d, joints_vis=None):
    """
    Triangulate 3d points with DLT,
    confidence scores can be passed to joints_vis

    Args:
        camera_objs: [C] a list of camera parameters, each corresponding to
                       one prediction in poses2d
        poses2d: [C, N, j, 2], [human1_view1, human1_view2,..., human2_view1, human2_view2,...]
        joints_vis: [C, N, j], only visible joints participate in triangulation
    Returns:
        poses3d: ndarray of shape [N, J, 3]
    """
    num_cams = len(camera_objs)

    if joints_vis is not None:
        # assert np.all(joints_vis.shape == poses2d.shape[:-1])
        pass  # no check for faster
    else:
        joints_vis = np.ones(poses2d.shape[:-1])

    # return 0 if less than 2 covered
    covered_mask = np.sum(joints_vis > 0, axis=0) >= 2  # [N, J]
    joints_vis = covered_mask[None, ...].astype(np.int32) * joints_vis  # [C, N, j]

    # make projection matrix
    P = np.zeros((num_cams, 3, 4))  # [C, 3, 4]

    for idx, CamModel in enumerate(camera_objs):
        P[idx] = CamModel.get_intrinsic() @ CamModel.get_extrinsic(homo=False)

    P = P[None, None, ...]  # [1, 1, C, 3, 4]
    row0 = P[..., 0, :]  # [1, 1, C, 4]
    row1 = P[..., 1, :]  # [1, 1, C, 4]
    row2 = P[..., 2, :]  # [1, 1, C, 4]
    # row0, row1, row2 = np.split(, 3, axis=-2)  # [1, 1, C, 4]

    poses2d = poses2d.transpose(1, 2, 0, 3)  # [N, J, C, 2]

    joints_vis = joints_vis[..., None].transpose(1, 2, 0, 3)  # [N, J, C, 1]
    eq1 = poses2d[..., [0]] * row2 - row0  # [N, J, C, 4]
    eq2 = poses2d[..., [1]] * row2 - row1  # [N, J, C, 4]

    eq1 = eq1 * joints_vis
    eq2 = eq2 * joints_vis

    A = np.concatenate((eq1, eq2), axis=-2)  # [N, J, 2C, 4]

    # batch SVD on [2C, 4]
    u, s, vh = np.linalg.svd(A)  # vh: [N, J, 4, 4]

    points_un = vh[..., -1, :3]  # [N, J, 3]
    points_scale = vh[..., -1, [3]]  # [N, J, 1]
    points = np.divide(
        points_un,
        points_scale,
        where=points_scale != 0,
        out=np.zeros_like(points_un),
    )  # [N, J, 3]

    return points


def fast_ransac(
    camera_objs,
    poses2d,
    joints_vis=None,
    n_iters=10,
    reprojection_error_epsilon=5,
    direct_optimization=False,
):
    """
    Ransac 3d points

    Args:
        camera_objs: [C] a list of camera parameters, each corresponding to
                       one prediction in poses2d
        poses2d: [C, N, j, 2], [human1_view1, human1_view2,..., human2_view1, human2_view2,...]
        joints_vis: [C, N, j], only visible joints participate in triangulation
    Returns:
        poses3d: ndarray of shape [N, J, 3]
    """
    # assert len(proj_matricies) == len(points)
    # assert len(points) >= 2  # number of views

    # proj_matricies = np.array(proj_matricies)
    # points = np.array(points)
    assert len(camera_objs) == len(poses2d)
    n_views, n_human, n_joints = poses2d.shape[:3]
    assert n_views >= 2

    # determine inliers
    all_views = range(n_views)
    # inlier_set = set()
    partial_triangulation_3d_dict = {}
    reprojected_poses2d = []

    if n_views <= 5:
        # sample all pairs
        # calculate all combinations
        # 2, 3, 4, ..., n-1, n; different from partial-tri, add res of n views
        keys = []
        num_cam_idx = [i for i in range(n_views)]
        for group_size in range(2, n_views + 1):
            group_index = list(itertools.combinations(num_cam_idx, group_size))
            keys.extend(group_index)

        whether_inlier = []
        for key in keys:
            part_cam_model_list = [
                CamModel for cam_id, CamModel in enumerate(camera_objs) if cam_id in key
            ]
            part_pred2d = poses2d[key, ...]  # [2, N, j, 2]
            pred3d = fast_triangulate(
                part_cam_model_list,
                part_pred2d,
                None,
            )  # [N, j, 3]
            partial_triangulation_3d_dict[key] = pred3d

            # reproject combination-2 resutls and calculate errors
            if len(key) == 2:
                proj2d_all_views = []
                valid_all_views = []
                pred3d_flat = pred3d.reshape((-1, 3))  # [N*j, 3]
                for cam_id, CamModel in enumerate(camera_objs):
                    proj2d = CamModel.project_to_2d(pred3d_flat, return_depth=True)  # [N*j, 3]
                    proj2d_all_views.append(proj2d[..., :2])  # [N*j, 2]
                    valid_all_views.append(proj2d[..., -1] > CamModel.f)  # [N*j]
                proj2d_all_views = np.array(proj2d_all_views)  # [C, N*j, 2]
                proj2d_all_views = proj2d_all_views.reshape(
                    (n_views, -1, n_joints, 2)
                )  # [C, N, j, 2]
                valid_all_views = np.array(valid_all_views).reshape(
                    (n_views, -1, n_joints)
                )  # [C, N, j]

                reproj_error = np.linalg.norm(poses2d - proj2d_all_views, axis=-1)  # [C, N, j]
                inlier_mask = np.logical_and(
                    reproj_error < reprojection_error_epsilon,
                    valid_all_views.astype(bool),
                )  # [C, N, j]
                whether_inlier.append(inlier_mask)
                # print(f'max reproj_error {np.amax(reproj_error)}')

        # whether_inlier  [S, C, N, j]
        whether_inlier = np.array(whether_inlier)  # [S, C, N, j]
        # print(whether_inlier.shape)
        # select triangulation plan with most inlier views for each point [N, j]
        inlier_plan = np.argmax(np.sum(whether_inlier, axis=1), axis=0)  # [N, j], index between 0-S

        # select triangulation res according to inlier_plan, all inliers participate in the triangulation
        ransac_3d = np.empty((n_human, n_joints, 3), dtype=np.float32)  # [N, j, 3]
        for human_idx in range(n_human):
            for joint_idx in range(n_joints):
                inlier_views = whether_inlier[
                    inlier_plan[human_idx, joint_idx], :, human_idx, joint_idx
                ]  # [C]

                selected_views = tuple(i for i in range(n_views) if inlier_views[i])
                if len(selected_views) < 2:
                    selected_views = tuple(range(n_views))

                ransac_3d[human_idx, joint_idx] = partial_triangulation_3d_dict[selected_views][
                    human_idx, joint_idx
                ]
        return ransac_3d
    else:
        # use random sampling
        raise NotImplementedError

    # for view_tuple in itertools.combinations(all_views, 2):
    #     sel_views = list(view_tuple)
    #     sel_camera_objs = [camera_objs[i] for i in sel_views]  # [2] list
    #     sel_poses2d = poses2d[sel_views]  # [2, N, j, 2]

    #     for

    # for i in range(n_iters):
    #     sampled_views = sorted(random.sample(view_set, 2))

    #     keypoint_3d_in_base_camera = multiview.triangulate_point_from_multiple_views_linear(proj_matricies[sampled_views], points[sampled_views])
    #     reprojection_error_vector = multiview.calc_reprojection_error_matrix(np.array([keypoint_3d_in_base_camera]), points, proj_matricies)[0]

    #     new_inlier_set = set(sampled_views)
    #     for view in view_set:
    #         current_reprojection_error = reprojection_error_vector[view]
    #         if current_reprojection_error < reprojection_error_epsilon:
    #             new_inlier_set.add(view)

    #     if len(new_inlier_set) > len(inlier_set):
    #         inlier_set = new_inlier_set

    # # triangulate using inlier_set
    # if len(inlier_set) == 0:
    #     inlier_set = view_set.copy()

    # inlier_list = np.array(sorted(inlier_set))
    # inlier_proj_matricies = proj_matricies[inlier_list]
    # inlier_points = points[inlier_list]

    # keypoint_3d_in_base_camera = multiview.triangulate_point_from_multiple_views_linear(inlier_proj_matricies, inlier_points)
    # reprojection_error_vector = multiview.calc_reprojection_error_matrix(np.array([keypoint_3d_in_base_camera]), inlier_points, inlier_proj_matricies)[0]
    # reprojection_error_mean = np.mean(reprojection_error_vector)

    # keypoint_3d_in_base_camera_before_direct_optimization = keypoint_3d_in_base_camera
    # reprojection_error_before_direct_optimization = reprojection_error_mean

    # # direct reprojection error minimization
    # if direct_optimization:
    #     def residual_function(x):
    #         reprojection_error_vector = multiview.calc_reprojection_error_matrix(np.array([x]), inlier_points, inlier_proj_matricies)[0]
    #         residuals = reprojection_error_vector
    #         return residuals

    #     x_0 = np.array(keypoint_3d_in_base_camera)
    #     res = least_squares(residual_function, x_0, loss='huber', method='trf')

    #     keypoint_3d_in_base_camera = res.x
    #     reprojection_error_vector = multiview.calc_reprojection_error_matrix(np.array([keypoint_3d_in_base_camera]), inlier_points, inlier_proj_matricies)[0]
    #     reprojection_error_mean = np.mean(reprojection_error_vector)

    # return keypoint_3d_in_base_camera, inlier_list
