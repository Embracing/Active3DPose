import copy
import sys

from PyQt5.QtWidgets import *

from activepose.config import config
from activepose.env_config import get_human_config
from activepose.pose.utils2d.pose.skeleton_def import get_skeleton
from activepose.visualization.ui.window import Window


def preprocess_data(data_dict):
    """
    Preprocess 3d data
    Transformation between unreal coordination (example data) and qt coordination
    y_qt = -y_unreal
    """
    data_dict = copy.deepcopy(data_dict)

    for k, v in data_dict.items():
        if k in ['gt3d', 'pred3d', 'camera', 'map_center']:
            data_dict[k][..., 1] *= -1
    return data_dict


if __name__ == '__main__':
    meta_info = {
        'skeleton': get_skeleton('COCO-WholeBody-body+head')['skeleton'],
        'preprocess_func': preprocess_data,
        # RGB
        'camera_colors': config.ENV.PLOT_CAMERA_COLORS,
        'human_colors': get_human_config(num_humans=10)[1]['mask_color'],
        'symbols': ['t', 'o', '+', 's', 'p', 'h', 'star', 'd', 't1', 't2', 't3'],
    }

    app = QApplication(sys.argv)
    win = Window(meta_info)
    win.show()
    sys.exit(app.exec())  # block
