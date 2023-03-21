import os

# import logging
import shutil

import cv2

# max support: 10 colors
from activepose.env_config import human_config_dict
from activepose.pose.utils2d.pose.skeleton_def import get_skeleton

from .utils import viz_bbx_inplace, viz_skeleton2d_inplace


# from .visualization2d_utils import draw_str
# logger = logging.getLogger(__name__)


color_list = human_config_dict[10]['mask_color']


def visualization2d_wrapper(imgs, record, output_dir=None):
    """
    imgs: ndarray, imgs from C views [C, h, w, 3]
    """
    if isinstance(imgs, list):
        viz_images = [img.copy() for img in imgs]
    else:
        viz_images = imgs.copy()
    njoints = record['pose2d'][0].shape[1]
    if njoints < 15 or njoints > 22:
        if njoints == 54:
            skeleton2d = get_skeleton('COCO-WholeBody')
        elif njoints == 12:
            skeleton2d = get_skeleton('COCO-WholeBody-onlybody')
        elif njoints == 14:
            skeleton2d = get_skeleton('COCO-WholeBody-body+head')
        else:
            assert 0, 'invalid number of joints'
    else:
        skeleton2d = get_skeleton('LCN_2D_%d' % njoints)

    # draw
    for idx, img in enumerate(viz_images):
        # # draw flag
        # if 'img_level_bad_flags' in record:
        #     img_level_bad_flag = record['img_level_bad_flags'][idx]
        #     draw_str(img, (20, 20), 'bad flag: %d' % img_level_bad_flag)

        # print(type(img))
        # print(img.shape, img.dtype)

        # draw pose
        if 'pose2d' in record:
            pose2d = record['pose2d'][idx]  # [N_max, j, 2]
            viz_skeleton2d_inplace(img, pose2d, skeleton2d, color_list)

        # draw detected boxes
        if 'bb' in record:
            boxes = record['bb'][idx]  # [N_max, 4]
            viz_bbx_inplace(img, boxes, color_list, marginal_pix=0)

        # if 'lhand_bb' in record:
        #     detected_lhand = record['lhand_bb'][idx]  # [4]
        #     viz_bbx_inplace(config, img, detected_lhand,
        #         target_color=(20, 245, 99), draw_enlarge=False, marginal_pix=0)

        # if 'rhand_bb' in record:
        #     detected_rhand = record['rhand_bb'][idx]  # [4]
        #     viz_bbx_inplace(config, img, detected_rhand,
        #         target_color=(245, 56, 244), draw_enlarge=False, marginal_pix=0)

    # Save
    if output_dir is not None:
        save_dir = os.path.join(output_dir, 'frames')
        shutil.rmtree(save_dir, ignore_errors=True)
        os.makedirs(save_dir)

        for idx, viz_img in enumerate(viz_images):
            cv2.imwrite(os.path.join(save_dir, str(idx) + '.jpg'), viz_img)

    return viz_images
