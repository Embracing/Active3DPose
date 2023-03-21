import logging

import cv2
import numpy as np

from .base import VideoDataset


logger = logging.getLogger(__name__)


class YoloDataset(VideoDataset):
    def __init__(self, yolo_img_size):
        self.yolo_img_size = yolo_img_size

    def __getitem__(self, index):
        frame = type(self).imgs[index]
        darknet_img, _, _, _ = self.letterbox(frame, new_shape=self.yolo_img_size)
        # Normalize RGB
        darknet_img = darknet_img[:, :, ::-1].transpose(2, 0, 1)  # BGR to RGB
        darknet_img = np.ascontiguousarray(darknet_img, dtype=np.float32)  # uint8 to float32
        darknet_img /= 255.0  # 0 - 255 to 0.0 - 1.0
        return darknet_img  # frame: BGR, darknet_img: RGB

    def letterbox(self, img, new_shape=416, color=(127.5, 127.5, 127.5), mode='auto'):
        # Resize a rectangular image to a 32 pixel multiple rectangle
        # https://github.com/ultralytics/yolov3/issues/232
        shape = img.shape[:2]  # current shape [height, width]
        if isinstance(new_shape, int):
            ratio = float(new_shape) / max(shape)
        else:
            ratio = max(new_shape) / max(shape)  # ratio  = new / old
        new_unpad = (int(round(shape[1] * ratio)), int(round(shape[0] * ratio)))

        # Compute padding https://github.com/ultralytics/yolov3/issues/232
        if mode == 'auto':  # minimum rectangle
            dw = np.mod(new_shape - new_unpad[0], 32) / 2  # width padding
            dh = np.mod(new_shape - new_unpad[1], 32) / 2  # height padding
        elif mode == 'square':  # square
            dw = (new_shape - new_unpad[0]) / 2  # width padding
            dh = (new_shape - new_unpad[1]) / 2  # height padding
        elif mode == 'rect':  # square
            dw = (new_shape[1] - new_unpad[0]) / 2  # width padding
            dh = (new_shape[0] - new_unpad[1]) / 2  # height padding

        top, bottom = int(round(dh - 0.1)), int(round(dh + 0.1))
        left, right = int(round(dw - 0.1)), int(round(dw + 0.1))
        img = cv2.resize(img, new_unpad, interpolation=cv2.INTER_LINEAR)  # resized, no border
        img = cv2.copyMakeBorder(
            img, top, bottom, left, right, cv2.BORDER_CONSTANT, value=color
        )  # padded square
        return img, ratio, dw, dh
