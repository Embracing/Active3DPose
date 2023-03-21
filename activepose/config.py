import glob
import os

from yacs.config import CfgNode as CN

from activepose import ROOT_DIR


config = CN()

config.OUTPUT_DIR = 'output'
config.LOG_DIR = 'log'
config.DATA_DIR = ''
config.BACKBONE_MODEL = 'pose_resnet'
config.MODEL = 'multiview_pose_resnet'
config.GPUS = (0,)
config.WORKERS = 8
config.PRINT_FREQ = 100
config.PRODUCTION = False

config.MODEL_CONFIG_FILE = os.path.join(ROOT_DIR, 'configs', 'w32_256x192_17j_coco.yaml')

# Cudnn related params
config.CUDNN = CN()
# True for cudnn to choose faster implementation. False for reproduce
config.CUDNN.BENCHMARK = False
config.CUDNN.DETERMINISTIC = True
config.CUDNN.ENABLED = True

# common params for NETWORK
config.NETWORK = CN()
config.NETWORK.PRETRAINED = 'models/pytorch/imagenet/resnet50-19c8e357.pth'
config.NETWORK.NUM_JOINTS = 16
config.NETWORK.HEATMAP_SIZE = [80, 80]
config.NETWORK.IMAGE_SIZE = [320, 320]
config.NETWORK.SIGMA = 2
config.NETWORK.TARGET_TYPE = 'gaussian'
config.NETWORK.AGGRE = False

# HRNet related params
config.MODEL_EXTRA = CN()
config.MODEL_EXTRA.PRETRAINED_LAYERS = ['conv1', 'bn1']
config.MODEL_EXTRA.FINAL_CONV_KERNEL = 1
config.MODEL_EXTRA.STAGE2 = CN()
config.MODEL_EXTRA.STAGE2.NUM_MODULES = 1
config.MODEL_EXTRA.STAGE2.NUM_BRANCHES = 2
config.MODEL_EXTRA.STAGE2.BLOCK = 'BASIC'
config.MODEL_EXTRA.STAGE2.NUM_BLOCKS = [4, 4]
config.MODEL_EXTRA.STAGE2.NUM_CHANNELS = [48, 96]
config.MODEL_EXTRA.STAGE2.FUSE_METHOD = 'SUM'
config.MODEL_EXTRA.STAGE3 = CN()
config.MODEL_EXTRA.STAGE3.NUM_MODULES = 4
config.MODEL_EXTRA.STAGE3.NUM_BRANCHES = 3
config.MODEL_EXTRA.STAGE3.BLOCK = 'BASIC'
config.MODEL_EXTRA.STAGE3.NUM_BLOCKS = [4, 4, 4]
config.MODEL_EXTRA.STAGE3.NUM_CHANNELS = [48, 96, 192]
config.MODEL_EXTRA.STAGE3.FUSE_METHOD = 'SUM'
config.MODEL_EXTRA.STAGE4 = CN()
config.MODEL_EXTRA.STAGE4.NUM_MODULES = 3
config.MODEL_EXTRA.STAGE4.NUM_BRANCHES = 4
config.MODEL_EXTRA.STAGE4.BLOCK = 'BASIC'
config.MODEL_EXTRA.STAGE4.NUM_BLOCKS = [4, 4, 4, 4]
config.MODEL_EXTRA.STAGE4.NUM_CHANNELS = [48, 96, 192, 384]
config.MODEL_EXTRA.STAGE4.FUSE_METHOD = 'SUM'

# DATASET related params
config.DATASET = CN()
config.DATASET.ROOT = 'data/'
config.DATASET.TRAIN_DATASET = 'mixed_dataset'
config.DATASET.TEST_DATASET = 'multiview_h36m'
config.DATASET.TRAIN_SUBSET = 'train'
config.DATASET.TEST_SUBSET = 'validation'
config.DATASET.PSEUDO_LABEL_PATH = ''
config.DATASET.NO_DISTORTION = False
config.DATASET.ROOTIDX = 0
config.DATASET.DATA_FORMAT = 'jpg'
config.DATASET.BBOX = 2000

# training data augmentation
config.DATASET.MPII_SCALE_FACTOR = 0.0
config.DATASET.MPII_ROT_FACTOR = 0
config.DATASET.MPII_FLIP = False

config.DATASET.H36M_SCALE_FACTOR = 0.0
config.DATASET.H36M_ROT_FACTOR = 0
config.DATASET.H36M_FLIP = False

config.DATASET.COCO_SCALE_FACTOR = 0.0
config.DATASET.COCO_ROT_FACTOR = 0
config.DATASET.COCO_FLIP = False

# params for mpii dataset and h36m dataset
config.DATASET.MPII_ROOTIDX = 6
config.DATASET.H36M_ROOTIDX = 0

# params for mixed dataset, balance batch smaples
# h36m:mpii = 39:1
config.DATASET.IF_SAMPLE = False
config.DATASET.H36M_WEIGHT = 1
config.DATASET.MPII_WEIGHT = 10

config.DATASET.COLOR_JITTER = False

# for compatibility
config.LOSS = CN()
config.LOSS.USE_TARGET_WEIGHT = True

config.TRAIN = CN(new_allowed=True)
config.DEBUG = CN(new_allowed=True)

# testing
config.TEST = CN()
config.TEST.BATCH_SIZE = 8
config.TEST.STATE = ''
config.TEST.FLIP_TEST = False
config.TEST.POST_PROCESS = False
config.TEST.SHIFT_HEATMAP = False
config.TEST.USE_GT_BBOX = False
config.TEST.IMAGE_THRE = 0.1
config.TEST.NMS_THRE = 0.6
config.TEST.OKS_THRE = 0.5
config.TEST.IN_VIS_THRE = 0.0
config.TEST.BBOX_FILE = ''
config.TEST.BBOX_THRE = 1.0
config.TEST.MATCH_IOU_THRE = 0.3
config.TEST.DETECTOR = 'fpn_dcn'
config.TEST.DETECTOR_DIR = ''
config.TEST.MODEL_FILE = ''
config.TEST.FUSE_OUTPUT = True
# use trt engine instead of pytorch checkpoints
# if tensorrt runtime is avaliable and trt file exists
config.TEST.TRY_TRT = True
config.TEST.TRT_FILE = 'checkpoints/hr_w32_256x192_coco_fp16_maxb35_trtexec.engine'
config.TEST.TRT_MAX_BATCH_SIZE = 35

# # Yolo model
# config.YOLO = CN()
# config.YOLO.ARCH_CFG_PATH = 'checkpoints/yolov3-spp.cfg'
# config.YOLO.DATA_CFG_PATH = 'checkpoints/coco.data'
# config.YOLO.WEIGHTS_PATH = 'checkpoints/yolov3-spp_4000_tuned1x.weights'
# config.YOLO.IMG_SIZE = 416
# config.YOLO.CONF_THRES = 0.5
# config.YOLO.NMS_THRES = 0.4

# config.YOLO.SINGLE_PERSON_THRES = 3
# config.YOLO.BBOX_ENLARGE = 1.25
# config.YOLO.MAX_INTERPOLATE_FRAMES = 15


config.ENV = CN()
config.ENV.VERSION = 13
config.ENV.BINARY_VERSION = 'blank'  # blank, building, grass, gym

config.ENV.PLOT_CAMERA_COLORS = [
    (255, 0, 0),
    (124, 252, 0),
    (0, 0, 255),
    (255, 0, 255),
    (0, 255, 255),
]
# config.ENV.PLOT_CAMERA_COLORS = ['r']*8 + ['g']*8 + ['b']*8

config.ENV.RESOLUTION = (320, 240)
config.ENV.ENV_NAME = 'C4_6x6_h30_p35'
config.ENV.MAP_NAME = 'Blank'
config.ENV.ALL_MAPS = [
    'Blank',
    'SchoolGymDay',
    'Building',
    'Wilderness',
    'Building_small',
]
config.ENV.HUMAN_MODEL = 'Default'
config.ENV.NUM_OF_HUMANS = 7
human_model_match = {
    'Blank': 'Default',
    'SchoolGymDay': 'Sportman',
    'Building': 'Businessman',
    'Wilderness': 'Default',
}

binary_version_match = {
    'Blank': 'blank',
    'SchoolGymDay': 'gym',
    'Building': 'building',
    'Wilderness': 'grass',
}

unreal_map_match = {
    'Blank': 'Blank',
    'SchoolGymDay': 'SchoolGymDay',
    'Building': 'Building_small',
    'Wilderness': 'grass_plane',
}


config.ENV.MOVE_DISTANCE_MATRIX = [[-30, 0, 30], [-30, 0, 30], [-30, 0, 30]]
config.ENV.ROTATION_ANGLE_MATRIX = [[-3, 0, 3], [-5, 0, 5]]
config.ENV.MAX_NUM_OF_HUMANS = 7
config.ENV.MAX_NUM_OF_CAMERAS = 20
config.ENV.WALK_SPEED_RANGE = [20, 30]
config.ENV.ROTATION_SPEED_RANGE = [80, 100]
config.ENV.EVOLUTION_STEPS = 1
config.ENV.MAX_STEPS = 5000
config.ENV.YAW_PID_COEF = 0.8
config.ENV.PITCH_PID_COEF = 0.8
config.ENV.FOV_PID_COEF = 0.2
config.ENV.YAW_PID_COEF_3D = 0.8
config.ENV.PITCH_PID_COEF_3D = 0.8
config.ENV.FOV_PID_COEF_3D = 0.2

config.ENV.BODY_EXPECTED_RATIO = 0.1  # 0.05 ~ 0.2, body size / img size
config.ENV.HAND_EXPECTED_RATIO = 0.002  #
config.ENV.WALK_SPEED = 125
config.ENV.ROT_SPEED = 200
config.ENV.ACTION_MODE = 'walk'  # mixed; anim; walk;
config.ENV.DONE_LOST_LAST_STEPS = 20  # triggered after 20 steps.

config.ENV.FREEZE_WALK_MODE = 'pause_game'  # pause_game or zero_speed

config.ENV.RENDER_DRIVER = 'opengl4'  # or 'opengl4', only works on linux

config.REWARD = CN()
# smaller than this value will lead to negative reward, same as REC3D.VISIBILITY_THRESH
config.REWARD.VISIBILITY_THRESH = 0.3
config.REWARD.REC3D_HAND_COEF = 1  # 3d reconstruction reward, weight for hand
config.REWARD.REC3D_COEF = 1  # 3d reconstruction reward, weight for hand
config.REWARD.VIS_COEF = 1
config.REWARD.MULTISELECTION_COEF = 1
config.REWARD.REC3D_FUNC = 'gemen'  # gemen or l2

config.REC3D = CN()
config.REC3D.USE_RANSAC = False  # otherwise Triangulation
config.REC3D.TRIANGULATION_STRATEGY = 'confidence'  # confidence, all, rl-select
config.REC3D.VISIBILITY_THRESH = 0.3  # joints whose confidence > thresh involved in
config.REC3D.DONE_LOST_JOINTS_RATIO = (
    0.55  # env returns done when 60% joints are not reconstructed ..
)

config.REC3D.USE_TEMPORAL_SMOOTHING = False
config.REC3D.USE_ONE_EURO = True
config.REC3D.COPY_LAST_PRED = False  # Temporal Fusion

config.REMOTE_RENDER = CN()
config.REMOTE_RENDER.URL = 'http://127.0.0.1:1234/data'

config.OBS_DIM = CN()
config.OBS_DIM['3D_WORLD'] = 10
config.OBS_DIM['3D_LOCAL'] = 6
config.OBS_DIM['3D_PART'] = config.OBS_DIM['3D_WORLD'] + config.OBS_DIM['3D_LOCAL']
config.OBS_DIM['2D_PART'] = 6
config.OBS_DIM['FLAG'] = 6
config.OBS_DIM['ENV'] = 8
config.OBS_DIM['CAMERA'] = 9
config.OBS_DIM['HUMAN'] = (
    config.OBS_DIM['3D_PART'] + config.OBS_DIM['2D_PART'] + config.OBS_DIM['FLAG']
)

config.OBS_SLICES = CN()
config.OBS_SLICES['ENV'] = slice(0, config.OBS_DIM['ENV'])
config.OBS_SLICES['CAMERA'] = slice(
    config.OBS_SLICES['ENV'].stop,
    config.OBS_SLICES['ENV'].stop + config.OBS_DIM['CAMERA'],
)
config.OBS_SLICES['HUMAN'] = slice(
    config.OBS_SLICES['CAMERA'].stop,
    config.OBS_SLICES['CAMERA'].stop + config.OBS_DIM['HUMAN'] * config.ENV.NUM_OF_HUMANS,
)
config.OBS_SLICES['TARGET'] = slice(
    config.OBS_SLICES['CAMERA'].stop,
    config.OBS_SLICES['CAMERA'].stop + config.OBS_DIM['HUMAN'] * 1,
)
config.OBS_SLICES['OBSTACLES'] = slice(config.OBS_SLICES['TARGET'].stop, None)


def update_config(cfg, args):
    cfg.defrost()

    if args is None:
        cfg.merge_from_file(cfg.MODEL_CONFIG_FILE)
    else:
        if hasattr(args, 'cfg') and args.cfg is not None:
            cfg.MODEL_CONFIG_FILE = args.cfg
            cfg.merge_from_file(args.cfg)
        else:
            cfg.merge_from_file(cfg.MODEL_CONFIG_FILE)

        if hasattr(args, 'opts'):
            cfg.merge_from_list(args.opts)

        if getattr(args, 'num_humans', None):
            config.ENV.NUM_OF_HUMANS = args.num_humans

        if getattr(args, 'walk_speed_range', None):
            config.ENV.WALK_SPEED_RANGE = args.walk_speed_range

        if getattr(args, 'rot_speed_range', None):
            config.ENV.ROTATION_SPEED_RANGE = args.rot_speed_range

        if getattr(args, 'max_num_humans', None):
            config.ENV.MAX_NUM_OF_HUMANS = args.max_num_humans
        config.ENV.MAX_NUM_OF_HUMANS = max(config.ENV.MAX_NUM_OF_HUMANS, config.ENV.NUM_OF_HUMANS)

        if getattr(args, 'env_name', None):
            config.ENV.ENV_NAME = args.env_name

        if hasattr(args, 'rec3d_func') and args.rec3d_func is not None:
            config.REWARD.REC3D_FUNC = args.rec3d_func

        if getattr(args, 'map_name', None):
            config.ENV.MAP_NAME = unreal_map_match[args.map_name]
            config.ENV.BINARY_VERSION = binary_version_match[args.map_name]
            config.ENV.HUMAN_MODEL = human_model_match[args.map_name]

            path_to_default_binary = sorted(
                glob.glob(
                    '{base}{sep}binary{sep}{version}{sep}*{sep}Binaries{sep}**{sep}AnimalParsing*'.format(
                        base=os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                        sep=os.path.sep,
                        version=config.ENV.BINARY_VERSION,
                    ),
                    recursive=True,
                )
            )[0]
            assert path_to_default_binary, 'Default binary not found'

            if hasattr(args, 'worker_index') and args.worker_index is not None:
                from pathlib import Path

                p_default_binary = Path(path_to_default_binary)

                # find the parent which has the name Binaries
                p_binary = p_default_binary
                while p_binary.name != 'Binaries':
                    p_binary = p_binary.parent
                    assert p_binary.name, 'Binaries folder not found'

                subfolder_name = p_default_binary.parent.name + '_' + str(args.worker_index)

                binary_path = sorted(
                    glob.glob(
                        '{base}{sep}binary{sep}{version}{sep}*{sep}Binaries{sep}{subfolder_name}{sep}AnimalParsing*'.format(
                            base=os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                            sep=os.path.sep,
                            version=config.ENV.BINARY_VERSION,
                            subfolder_name=subfolder_name,
                        ),
                        recursive=True,
                    )
                )

                # If no binary found, create
                if len(binary_path) == 0:
                    import shutil

                    Path.mkdir(p_binary / subfolder_name, parents=True, exist_ok=True)

                    # make a hard link from p_default_binary to p_binary / new_folder_name
                    p_default_binary.link_to(p_binary / subfolder_name / p_default_binary.name)

                    # cp unrealcv.ini from p_default_binary.parent to p_binary / new_folder_name with shutil
                    shutil.copy(
                        p_default_binary.parent / 'unrealcv.ini',
                        p_binary / subfolder_name / 'unrealcv.ini',
                    )

                    binary_path = str(Path(p_binary / subfolder_name / p_default_binary.name))
                else:
                    binary_path = binary_path[0]

            else:
                # Likely in local debug mode so route to any binary
                binary_path = path_to_default_binary

            config.ENV.BINARY_PATH = binary_path
            assert binary_path, 'Binary Path Not Found'

        if getattr(args, 'render', None):
            config.ENV.RENDER_DRIVER = args.render

    cfg.freeze()
