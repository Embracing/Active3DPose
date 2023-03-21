import math

import cv2
import numpy as np
import torch


class AverageMeter:
    """Computes and stores the average and current value"""

    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count


def my_transform(imgs, mean, std):
    """
    Rewrite the pytorch transforms to include batch processing
    imgs: numpy array [N, H, W, C]
    Return:
        imgs: torch.tensor [N, C, H, W]
    """

    imgs = torch.from_numpy(imgs.transpose((0, 3, 1, 2)))
    imgs = imgs.float().div(255)

    dtype = imgs.dtype
    mean = torch.as_tensor(mean, dtype=dtype, device=imgs.device)
    std = torch.as_tensor(std, dtype=dtype, device=imgs.device)
    if (std == 0).any():
        raise ValueError(
            'std evaluated to zero after conversion to {}, leading to division by zero.'.format(
                dtype
            )
        )
    if mean.ndim == 1:
        mean = mean[None, :, None, None]
    if std.ndim == 1:
        std = std[None, :, None, None]
    imgs.sub_(mean).div_(std)
    return imgs


# def my_transform_numpy(imgs, mean, std):
#     """
#     Rewrite the pytorch transforms to include batch processing
#     imgs: numpy array [N, H, W, C]
#     mean: a list of len 3
#     std: a list of len 3
#     Return:
#         imgs: numpy [N, C, H, W]
#     """

#     imgs = imgs.transpose((0, 3, 1, 2))
#     imgs = imgs.astype(np.float32) / 255

#     mean = np.array(mean, dtype=np.float32)
#     std = np.array(std, dtype=np.float32)

#     mean = mean[None, :, None, None]
#     std = std[None, :, None, None]
#     imgs = (imgs - mean) / std
#     return imgs


def my_transform_np_outer(mean, std):
    """
    mean: a list of len 3
    std: a list of len 3
    """
    mean = np.array(mean, dtype=np.float32)[None, :, None, None]
    std = np.array(std, dtype=np.float32)[None, :, None, None]

    def my_transform_np(imgs):
        """
        Rewrite the pytorch transforms to include batch processing
        imgs: numpy array [N, H, W, C]
        Return:
            imgs: numpy [N, C, H, W]
        """
        imgs = imgs.transpose((0, 3, 1, 2))
        imgs = imgs.astype(np.float32) / 255
        imgs = (imgs - mean) / std
        return imgs

    return my_transform_np


def get_max_preds(batch_heatmaps):
    """
    get predictions from score maps
    heatmaps: numpy.ndarray([batch_size, num_joints, height, width])
    """
    # assert isinstance(batch_heatmaps, np.ndarray), \
    #     'batch_heatmaps should be numpy.ndarray'
    # assert batch_heatmaps.ndim == 4, 'batch_images should be 4-ndim'

    batch_size = batch_heatmaps.shape[0]
    num_joints = batch_heatmaps.shape[1]
    width = batch_heatmaps.shape[3]
    heatmaps_reshaped = batch_heatmaps.reshape((batch_size, num_joints, -1))
    idx = np.argmax(heatmaps_reshaped, 2)
    maxvals = np.amax(heatmaps_reshaped, 2)

    maxvals = maxvals.reshape((batch_size, num_joints, 1))
    idx = idx.reshape((batch_size, num_joints, 1))

    preds = np.tile(idx, (1, 1, 2)).astype(np.float32)

    preds[:, :, 0] = (preds[:, :, 0]) % width
    preds[:, :, 1] = np.floor((preds[:, :, 1]) / width)

    pred_mask = np.tile(np.greater(maxvals, 0.0), (1, 1, 2))
    pred_mask = pred_mask.astype(np.float32)

    preds *= pred_mask
    return preds, maxvals


def get_final_preds(batch_heatmaps, center, scale, rot):
    coords, maxvals = get_max_preds(batch_heatmaps)

    heatmap_height = batch_heatmaps.shape[2]
    heatmap_width = batch_heatmaps.shape[3]

    # coords: [batch, j, 2]
    preds = np.empty_like(coords)

    # Transform back
    for i in range(coords.shape[0]):
        preds[i] = transform_preds(
            coords[i], center[i], scale[i], rot[i], [heatmap_width, heatmap_height]
        )

    return preds, maxvals


def get_bbox_from_kp2d_batch(kp2d, margin=0):
    """
    kp2d: [J, 2] or [N, J, 2]
    Return:
        bb_array: [N, 4], [x1, y1, w, h]
    """
    assert 2 <= kp2d.ndim <= 3

    if kp2d.ndim == 2:
        kp2d = kp2d[None, ...]

    x1y1 = np.amin(kp2d, axis=1)  # [N, 2]
    x2y2 = np.amax(kp2d, axis=1)  # [N, 2]
    wh = x2y2 - x1y1  # [N, 2]

    x1y1 = x1y1 - wh * margin
    wh = wh * (1 + 2 * margin)
    bb_array = np.concatenate([x1y1, wh], axis=1).astype(np.int32)  # [N, 4]
    return bb_array


def transform_preds(coords, center, scale, rot, output_size):
    target_coords = np.zeros(coords.shape)
    trans = get_affine_transform(center, scale, rot, output_size, inv=1)
    # for p in range(coords.shape[0]):
    #     target_coords[p, 0:2] = affine_transform(coords[p, 0:2], trans)
    target_coords[:, :2] = affine_transform(coords[:, :2], trans)
    return target_coords


def get_affine_transform(
    center, scale, rot, output_size, shift=np.array([0, 0], dtype=np.float32), inv=0
):
    if not isinstance(scale, np.ndarray) and not isinstance(scale, list):
        scale = np.array([scale, scale])

    scale_tmp = scale * 200.0
    src_w = scale_tmp[0]
    dst_w = output_size[0]
    dst_h = output_size[1]

    rot_rad = np.pi * rot / 180
    src_dir = get_dir([0, src_w * -0.5], rot_rad)
    dst_dir = np.array([0, dst_w * -0.5], np.float32)

    src = np.zeros((3, 2), dtype=np.float32)
    dst = np.zeros((3, 2), dtype=np.float32)
    src[0, :] = center + scale_tmp * shift
    src[1, :] = center + src_dir + scale_tmp * shift
    dst[0, :] = [dst_w * 0.5, dst_h * 0.5]
    dst[1, :] = np.array([dst_w * 0.5, dst_h * 0.5]) + dst_dir

    src[2:, :] = get_3rd_point(src[0, :], src[1, :])
    dst[2:, :] = get_3rd_point(dst[0, :], dst[1, :])

    if inv:
        trans = cv2.getAffineTransform(np.float32(dst), np.float32(src))
    else:
        trans = cv2.getAffineTransform(np.float32(src), np.float32(dst))

    return trans


def affine_transform(pt, t):
    """
    pt: [N, 2] or [2]
    t: [2, 3]
    """
    if pt.ndim == 1:
        pt = pt[np.newaxis, ...]
    pt = np.concatenate((pt, np.ones((pt.shape[0], 1))), axis=-1)
    return np.dot(pt, t.T).squeeze()


def get_3rd_point(a, b):
    direct = a - b
    return b + np.array([-direct[1], direct[0]], dtype=np.float32)


def get_dir(src_point, rot_rad):
    sn, cs = np.sin(rot_rad), np.cos(rot_rad)

    src_result = [0, 0]
    src_result[0] = src_point[0] * cs - src_point[1] * sn
    src_result[1] = src_point[0] * sn + src_point[1] * cs

    return src_result


def crop(img, center, scale, output_size, rot=0):
    trans = get_affine_transform(center, scale, rot, output_size)

    dst_img = cv2.warpAffine(
        img, trans, (int(output_size[0]), int(output_size[1])), flags=cv2.INTER_LINEAR
    )

    return dst_img
