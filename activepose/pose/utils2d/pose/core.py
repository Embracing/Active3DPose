import logging
import os
import time

import cv2
import numpy as np
import torch
import torch.nn as nn
from tqdm import tqdm

from activepose.pose.models import hrnet

from .utils import get_affine_transform, get_final_preds, my_transform, my_transform_np_outer


# from activepose.utils.transforms import get_affine_transform


my_transform_np = my_transform_np_outer(
    mean=[0.485, 0.456, 0.406],
    std=[0.229, 0.224, 0.225],
)

logger = logging.getLogger(__name__)

try:
    from numba import jit
except ImportError:
    pass
else:
    logger.info('=> numba compile transform')
    my_transform_np = jit(nopython=True)(my_transform_np)


def load_pose2d_model(config, device=torch.device('cuda', 0)):
    # load pose model
    base_model = hrnet.get_pose_net(config, is_train=False)

    # base_model = eval(config.MODEL + '.get_multiview_pose_net')(
    #     backbone_model, config)

    model_dict = nn.ModuleDict()
    model_dict['pose_model'] = base_model

    if config.TEST.MODEL_FILE:
        logger.info(f'=> loading Pose2d model from {config.TEST.MODEL_FILE}')
        state_dict = torch.load(config.TEST.MODEL_FILE)
    else:
        assert 0, 'you must specify a checkpoint'

    if 'state_dict_base_model' in state_dict:
        logger.info('=> new loading mode')
        # delete params of the aggregation layer
        for param_key in list(state_dict['state_dict_base_model'].keys()):
            if 'aggre_layer' in param_key:
                state_dict['state_dict_base_model'].pop(param_key)
        model_dict['pose_model'].load_state_dict(state_dict['state_dict_base_model'])
    elif 'state_dict' in state_dict:
        logger.info('=> new loading mode')
        new_state_dict = {}
        for param_key in list(state_dict['state_dict'].keys()):
            if 'backbone.' in param_key:
                new_state_dict[param_key.replace('backbone.', 'posenet.')] = state_dict[
                    'state_dict'
                ][param_key]
            elif 'keypoint_head.' in param_key:
                new_state_dict[param_key.replace('keypoint_head.', 'posenet.')] = state_dict[
                    'state_dict'
                ][param_key]
            else:
                new_state_dict[param_key] = state_dict['state_dict'][param_key]
        model_dict['pose_model'].load_state_dict(new_state_dict)
    else:
        logger.info('=> old loading mode')
        # delete params of the aggregation layer
        for param_key in list(state_dict.keys()):
            if 'aggre_layer' in param_key:
                state_dict.pop(param_key)
        model_dict['pose_model'].load_state_dict(state_dict)

    # gpus = [int(i) for i in config.GPUS.split(',')]
    # model_dict['pose_model'] = torch.nn.DataParallel(model_dict['pose_model'], device_ids=gpus).cuda()

    model_dict = model_dict.to(device)
    model_dict.device = device

    return model_dict


def load_pose2d_trt(config):
    try:
        from activepose.pose.models.trt_model import ONNXWrapper
    except ImportError:
        logger.info('=> tensorrt env is not installed ! ')
        return

    if not os.path.exists(config.TEST.TRT_FILE):
        logger.info(f'=> trt engine not found: {config.TEST.TRT_FILE}')
        return

    logger.info(f'=> loading trt engine from {config.TEST.TRT_FILE}')
    base_model = ONNXWrapper(
        file=config.TEST.TRT_FILE,
        target_dtype=np.float32,
        max_batch_size=config.TEST.TRT_MAX_BATCH_SIZE,
    )  # only support 3 cameras with a maximum of 21 persons

    model_dict = dict()
    model_dict['pose_model'] = base_model

    return model_dict


class Pose2dStreamingInferencer:
    def __init__(self, config, model_dict, device):
        self.device = device
        self.model_dict = model_dict
        for model in self.model_dict.values():
            # for pytorch model
            if isinstance(model, nn.Module):
                model.eval()

        # pose forward procedure
        if isinstance(self.model_dict['pose_model'], nn.Module):
            self.pose_forward = self._pytorch_pose_forward
        else:
            self.pose_forward = self._trt_pose_forward

        self.njoints = config.NETWORK.NUM_JOINTS
        self.hm_h = int(config.NETWORK.HEATMAP_SIZE[0])
        self.hm_w = int(config.NETWORK.HEATMAP_SIZE[1])
        self.config = config

        self.network_image_width = config.NETWORK.IMAGE_SIZE[0]
        self.network_image_height = config.NETWORK.IMAGE_SIZE[1]
        self.aspect_ratio = self.network_image_width * 1.0 / self.network_image_height

    def _box2cs(self, box):
        x, y, w, h = box[:4]
        return self._xywh2cs(x, y, w, h)

    def _xywh2cs(self, x, y, w, h):
        center = np.zeros((2,), dtype=np.float32)
        center[0] = x + w * 0.5
        center[1] = y + h * 0.5

        if w > self.aspect_ratio * h:
            h = w * 1.0 / self.aspect_ratio
        elif w < self.aspect_ratio * h:
            w = h * self.aspect_ratio
        scale = np.array([w * 1.0 / 200, h * 1.0 / 200], dtype=np.float32)
        if center[0] != -1:
            scale = scale * 1.25

        return center, scale

    def _pytorch_pose_forward(self, data):
        """
        pytorch model inference
        np.ndarray --> tensor --> model --> tensor --> np.ndarray
        """
        # used on BGR imgs
        # start = time.time()
        input_batch = my_transform(data, mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]).to(
            self.device
        )
        # print(f'transform time: {time.time() - start}')

        raw_features = self.model_dict['pose_model'](input_batch)  # [nsamples, njoints, h, w]
        heatmap_batch = raw_features.cpu().numpy()
        return heatmap_batch

    def _trt_pose_forward(self, data):
        """
        trt engine inference: np.ndarray --> model --> np.ndarray
        """
        # used on BGR imgs
        # start = time.time()
        input_batch = my_transform_np(data)
        # print(f'transform time: {time.time() - start}')

        heatmap_batch = self.model_dict['pose_model'](input_batch)  # [nsamples, njoints, h, w]
        return heatmap_batch

    def __call__(self, input_data_dict):
        # C cameras, N targets
        ncameras, ori_frame_h, ori_frame_w, _ = input_data_dict['imgs'].shape  # [C, h, w, 3] BGR
        nsamples = len(
            input_data_dict['bb']
        )  # [img1_obj1, img2_obj1, ..., img1_obj2, img2_obj2]. [N*C]

        # default crop: the entire images
        max_scale = max(ori_frame_w / 200.0, ori_frame_h / 200.0)
        all_rots = np.full((nsamples), 0, dtype=np.float32)  # [N]
        all_scales = np.zeros((nsamples, 2), dtype=np.float32)  # [N, 2]
        all_centers = np.zeros((nsamples, 2), dtype=np.float32)  # [N, 2]
        # all_centers[:, 0] = ori_frame_w // 2
        # all_centers[:, 1] = ori_frame_h // 2

        if input_data_dict.get('bb', None) is not None:
            if nsamples % ncameras == 0:
                for idx, item in enumerate(input_data_dict['bb']):
                    if item is None or np.any(item[2:] == 0):
                        continue
                    else:
                        center, scale = self._box2cs(item)
                        all_centers[idx] = center
                        all_scales[idx] = scale
            else:
                logger.warning(
                    '=>len mismatch, bb not used. det bb len: %d, input len: %d,  '
                    % (len(input_data_dict['bb']), nsamples)
                )

        nimgs = nsamples  # NxC
        input_imgs = input_data_dict['imgs']  # [C, h, w, 3]
        with torch.no_grad():
            cropped_imgs = np.empty(
                (
                    nimgs,
                    self.config.NETWORK.IMAGE_SIZE[1],
                    self.config.NETWORK.IMAGE_SIZE[0],
                    3,
                ),
                dtype=np.float32,
            )

            # generate img fed to the pose network
            # start = time.time()
            for idx in range(nimgs):
                trans = get_affine_transform(
                    all_centers[idx], all_scales[idx], 0, self.config.NETWORK.IMAGE_SIZE
                )
                crop = cv2.warpAffine(
                    input_imgs[idx % ncameras],
                    trans,
                    (
                        self.config.NETWORK.IMAGE_SIZE[0],
                        self.config.NETWORK.IMAGE_SIZE[1],
                    ),
                    flags=cv2.INTER_LINEAR,
                )
                cropped_imgs[idx] = crop
            # print(f'warp time: {time.time() - start}')

            # forward
            # start = time.time()
            heatmap_batch = self.pose_forward(cropped_imgs)  # [nsamples, njoints, h, w]
            # print(f'heatmap shape: {heatmap_batch.shape}')
            # print(f'forward time: {time.time() - start}')

            # # used on BGR imgs
            # input_batch = my_transform(cropped_imgs,
            #                            mean=[0.485, 0.456, 0.406],
            #                            std=[0.229, 0.224, 0.225]).to(self.device)
            # print(input_batch.shape)
            # start = time.time()
            # raw_features = self.model_dict['pose_model'](input_batch)  # [nsamples, njoints, h, w]
            # heatmap_batch = raw_features.cpu().numpy()
            # print(f'heatmap shape: {heatmap_batch.shape}')
            # print(f'=>forward time: {time.time() - start}')
            # # [B, 16, 2], [B, 16, 1]

            # start = time.time()
            pred_batch, maxval = get_final_preds(heatmap_batch, all_centers, all_scales, all_rots)
            # print(f'get_final_preds time: {time.time() - start}')

        record = {
            'pose2d': pred_batch,  # [nsamples, j, 2]
            'pose2d_scores': maxval,  # [nsamples, j, 1]
            'heatmaps': heatmap_batch,  # [nsamples, j, h, w]
        }

        return record


def initialize_streaming_pose_estimator(config, device=torch.device('cuda', 0)):
    model_dict = None
    if config.TEST.TRY_TRT:
        logger.info('=> try loading pose2d trt engine...')
        model_dict = load_pose2d_trt(config)

    # if the trt engine is not available
    if model_dict is None:
        logger.info('=> try loading pose2d pytorch checkpoint...')
        model_dict = load_pose2d_model(config, device=device)

    streaming_inferencer = Pose2dStreamingInferencer(config, model_dict, device=device)

    return streaming_inferencer
