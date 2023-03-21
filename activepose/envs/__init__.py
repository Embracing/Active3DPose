import os

import gym

from activepose import ROOT_DIR
from activepose.config import config, update_config
from activepose.envs.wrappers import *

from .base import MultiviewPose


def make_env(
    args=None,
    air_wall_outer=False,
    air_wall_inner=None,
    gt_observation=None,
    place_cam=None,
    fix_human=None,
    movable_cam=None,
    rule_based_rot=None,
    scale_human=None,
    ego_action=None,
    ema_action=None,
    rot_limit=None,
    rand_reset_human=None,
    shuffle_cam_id=None,
    partial_triangulation=None,
    **kwargs,
):
    if partial_triangulation is None:
        partial_triangulation = {'use': False}
    if shuffle_cam_id is None:
        shuffle_cam_id = {'use': False}
    if rand_reset_human is None:
        rand_reset_human = {'use': False, 'args': {'random_init_loc_rot': True}}
    if rot_limit is None:
        rot_limit = {
            'use': False,
            'args': {
                'pitch_low': -85,
                'pitch_high': 85,
                'yaw_low': -360,
                'yaw_high': 360,
            },
        }
    if ema_action is None:
        ema_action = {'use': False, 'args': {}}
    if ego_action is None:
        ego_action = {'use': False}
    if scale_human is None:
        scale_human = {'use': False, 'args': {'scale': 2, 'index_list': None}}
    if rule_based_rot is None:
        rule_based_rot = {
            'use': False,
            'args': {'use_gt3d': True, 'num_rule_cameras': None, 'index_list': [0]},
        }
    if movable_cam is None:
        movable_cam = {'use': False, 'args': {'num_movable_cameras': 1}}
    if fix_human is None:
        fix_human = {
            'use': False,
            'args': {'random_init_states': False, 'index_list': None},
        }
    if place_cam is None:
        place_cam = {
            'use': False,
            'args': {'num_place_cameras': 1, 'place_on_slider': False},
        }
    if gt_observation is None:
        gt_observation = {
            'use': False,
            'args': {'gt_noise_scale': 0, 'affected_by_visibility': False},
        }
    if air_wall_inner is None:
        air_wall_inner = {
            'use': False,
            'args': {'lower_bound': [-200, -200, 0], 'higher_bound': [200, 200, 500]},
        }

    update_config(config, args)
    if config.REC3D.USE_RANSAC:
        print(f'RANSAC has set to {True}')
    if config.REC3D.USE_TEMPORAL_SMOOTHING:
        print(f'Temporal Smoothing has set to {True}')
        print('USE_ONE_EURO: ', config.REC3D.USE_ONE_EURO)
        print('COPY_LAST_PRED: ', config.REC3D.COPY_LAST_PRED)

    config['NETWORK']['PRETRAINED'] = os.path.join(ROOT_DIR, config['NETWORK']['PRETRAINED'])
    config['TEST']['MODEL_FILE'] = os.path.join(ROOT_DIR, config['TEST']['MODEL_FILE'])

    worker_index = getattr(args, 'worker_index', 0)
    vector_index = getattr(args, 'vector_index', 0)
    num_envs = getattr(args, 'num_envs', 1)
    print('worker_index: ', worker_index)
    env = MultiviewPose(
        config,
        worker_index=worker_index,
        vector_index=vector_index,
        num_envs=num_envs,
        **kwargs,
    )

    if air_wall_outer:
        env = AirWallOuter(env)

    if air_wall_inner.get('use', False):
        env = AirWallInner(env, **air_wall_inner.get('args', dict()))

    if gt_observation.get('use', False):
        env = GroundTruthObservation(env, **gt_observation.get('args', dict()))

    if place_cam.get('use', False):
        env = PlaceCam(env, **place_cam.get('args', dict()))

    if fix_human.get('use', False):
        env = FixHuman(env, **fix_human.get('args', dict()))

    if movable_cam.get('use', False):
        env = MovableCam(env, **movable_cam.get('args', dict()))

    if rule_based_rot.get('use', False):
        env = RuleBasedRot(env, **rule_based_rot.get('args', dict()))

    if scale_human.get('use', False):
        env = ScaleHuman(env, **scale_human.get('args', dict()))

    if ego_action.get('use', False):
        env = EgoAction(env)

    if ema_action.get('use', False):
        env = EMAAction(env, ema_action.get('args', dict()))

    if rot_limit.get('use', False):
        env = RotLimit(env, **rot_limit.get('args', dict()))

    if rand_reset_human.get('use', False):
        env = RandResetHuman(env, **rand_reset_human.get('args', dict()))

    if shuffle_cam_id.get('use', False):
        env = ShuffleCamID(env)

    if partial_triangulation.get('use', False):
        env = PartialTriangulation(env)

    return env


gym.register(id='base-v0', entry_point=make_env)
