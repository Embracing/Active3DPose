import glob
import logging
import os

import cv2
import numpy as np
from torch.utils.data import Dataset

from activepose.utils.video import get_metadata, read_video_pyav


logger = logging.getLogger(__name__)


class VideoDataset(Dataset):
    """
    Base dataset to read videos. Note that all children instances of this class
    share the video data and other metadata.

    You can customize it (personalized __getitem()__) by inheritance.
    """

    @classmethod
    def read_video(cls, args):
        cls.metadata = None
        if os.path.isdir(args.input):
            cls.len = len(glob.glob(os.path.join(args.input, '*.jpg')))
            cls.imgs = []
            for i in range(1, 1 + cls.len):
                img = cv2.imread(args.input + str(i) + '.jpg')
                cls.imgs.append(img)
        elif os.path.splitext(args.input)[-1][1:].lower() in ['jpg', 'png']:
            cls.len = 1
            cls.imgs = [cv2.imread(args.input)]
        else:
            # video
            cls.imgs = read_video_pyav(args.input, bgr=True)
            cls.len = len(cls.imgs)
            try:
                cls.metadata = get_metadata(args.input)
            except Exception as e:
                logger.warning('==== Error when reading metadata =====')
                logger.warning(e.message)
                logger.warning(e.args)
                logger.warning('==== Skip reading ====================')
        cls.imgs = np.array(cls.imgs)  # [N, h, w, c]
        cls.hw = cls.imgs.shape[1:3]

    def __getitem__(self, index):
        frame = type(self).imgs[index]
        return frame

    def __len__(self):
        return type(self).len
