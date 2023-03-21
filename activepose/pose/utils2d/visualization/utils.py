import cv2
import numpy as np


def viz_skeleton2d_batch_inplace(batch_image, pose2d, skeleton2d, joints_vis=None):
    """
    Used for debug in 2d pose,
    final visualization functions are defined in another render py file

    all inputs are numpy array
    batch_image: [batch_size, height, width, channel]
    pose2d: [batch_size, num_joints, 2],
    joints_vis: [batch_size, num_joints, 1],
    }
    """
    ndarr = batch_image
    njoints = pose2d.shape[1]

    for frame_idx, pred in enumerate(pose2d):
        points = []
        for point in zip(pred[:, 0], pred[:, 1]):
            points.append(point)

        # plot skeletons
        for u, v in skeleton2d['skeleton']:
            if u in skeleton2d['joints_right']:
                col = [30, 200, 251, 1]  # xkcd:goldenrod
            else:
                col = [244, 164, 31, 1]  # xkcd:azure
                cv2.line(ndarr[frame_idx], points[u], points[v], col, 1)

        # plot joints
        for n in range(njoints):
            if n in skeleton2d['joints_right']:
                col = [30, 200, 251, 1]
            else:
                col = [244, 164, 31, 1]
            cv2.circle(ndarr[frame_idx], points[n], 1, col, -1)

    return ndarr


def draw_str(dst, target, s):
    x, y = target
    cv2.putText(
        dst,
        s,
        (x + 1, y + 1),
        cv2.FONT_HERSHEY_PLAIN,
        1.0,
        (0, 0, 0),
        thickness=2,
        lineType=cv2.LINE_AA,
    )
    cv2.putText(
        dst,
        s,
        (x, y),
        cv2.FONT_HERSHEY_PLAIN,
        1.0,
        (255, 255, 255),
        lineType=cv2.LINE_AA,
    )


def viz_bbx_inplace(img, box_list, color_list, marginal_pix=0):
    """
    img: [h, w, 3]
    box_list: [N_max, 4] [x1, y1, w, h]
    """
    assert len(color_list) >= len(box_list)
    for box, color in zip(box_list, color_list):
        if box is not None and not np.any(box[2:] == 0):
            x1, y1, w, h = box[0], box[1], box[2], box[3]
            x2, y2 = x1 + w, y1 + h
            xc, yc = x1 + w // 2, y1 + h // 2

            if marginal_pix != 0:
                x1 -= marginal_pix
                y1 -= marginal_pix
                x2 += marginal_pix
                y2 += marginal_pix

            cv2.rectangle(img, (x1, y1), (x2, y2), color, 1, cv2.LINE_AA)  # green dt box


def viz_skeleton2d_inplace(image, poses2d, skeleton2d, color_list, joints_vis=None):
    """
    Used for debug in 2d pose,
    final visualization functions are defined in another render py file

    all inputs are numpy array
    image: [height, width, channel]
    poses2d: [N, num_joints, 2], int
    joints_vis: [N, num_joints, 1],
    }
    """
    ndarr = image
    poses2d = np.asarray(poses2d)

    assert len(color_list) >= len(poses2d)

    if poses2d.ndim == 2:
        poses2d = poses2d[None, ...]

    njoints = poses2d.shape[1]

    if poses2d.dtype != np.int32 and poses2d.dtype != np.int:
        poses2d = np.around(poses2d).astype(np.int32)
    else:
        poses2d = poses2d

    for pose2d, color in zip(poses2d, color_list):
        if not np.any(pose2d):  # all zeros
            continue

        points = []
        for point in zip(pose2d[:, 0], pose2d[:, 1]):
            points.append(point)

        # plot skeletons
        for u, v in skeleton2d['skeleton']:
            # if u in skeleton2d['joints_right']:
            #     col = [30, 200, 251, 1]  # xkcd:goldenrod, bgr for cv2
            # else:
            #     col = [244, 164, 31, 1]  # xkcd:azure, bgr for cv2

            # if hand def exists
            if 'joints_right_hand' in skeleton2d and 'joints_left_hand' in skeleton2d:
                if u in skeleton2d['joints_right_hand'] or u in skeleton2d['joints_left_hand']:
                    thickness = 1
                else:
                    thickness = 2
            else:
                thickness = 1
            cv2.line(ndarr, points[u], points[v], color, thickness, cv2.LINE_AA)

    return ndarr
