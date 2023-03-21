import os
from argparse import Namespace

from ray.rllib.policy.policy import PolicySpec
from ray.tune import Experiment

from activepose import DEBUG, ROOT_DIR
from activepose.control.callbacks import CustomMetricCallback
from activepose.control.utils import SHARED_POLICY_ID, shared_policy_mapping_fn


def make_ctcr_wdl_experiment(n_cams=5, EXP_MODE='MAPPO+CTCR+WDL'):
    assert 2 <= n_cams <= 5, 'n_cams must be in [2, 5]'
    assert EXP_MODE in [
        'MAPPO+CTCR+WDL',
        'MAPPO+CTCR',
        'MAPPO+WDL',
        'MAPPO',
    ], "EXP_MODE must be in ['MAPPO+CTCR+WDL', 'MAPPO+CTCR', 'MAPPO+WDL', 'MAPPO']"

    # ===== Experiment Settings =====
    NUM_CAMERAS = n_cams
    INDEPENDENT = False

    # ===== Resource Settings =====
    NUM_CPUS_FOR_DRIVER = 5
    TRAINER_GPUS = 0.5  # Trainer GPU amount
    NUM_GPUS_PER_WORKER = 0.5
    NUM_ENVS_PER_WORKER = 4  # memory requirement grows with number of humans in the environment
    NUM_CPUS_PER_WORKER = 1  # 1 is enough
    NUM_WORKERS = 7 if not DEBUG else 4

    # ===== Sampler Settings =====
    ROLLOUT_FRAGMENT_LENGTH = 25
    NUM_SAMPLING_ITERATIONS = 1
    TRAIN_BATCH_SIZE = (
        NUM_SAMPLING_ITERATIONS * ROLLOUT_FRAGMENT_LENGTH * NUM_WORKERS * NUM_ENVS_PER_WORKER
    )
    SGD_MINIBATCH_SIZE = TRAIN_BATCH_SIZE // 2

    REWARD_DICT = {
        'teamonly': {
            'team_reward': 1.0,
        },
        'teamsmalliot': {
            'team_reward': 1.0,
            'iot_reward': 0.1,
        },
    }

    if not INDEPENDENT:
        MULTI_AGENT = {
            'policies': {SHARED_POLICY_ID: PolicySpec(observation_space=None, action_space=None)},
            'policy_mapping_fn': shared_policy_mapping_fn,
        }
    else:
        MULTI_AGENT = {
            'policies': {
                f'camera_{c}': PolicySpec(observation_space=None, action_space=None)
                for c in range(NUM_CAMERAS)
            },
            'policy_mapping_fn': lambda agent_id, **kwargs: agent_id,
        }

    ENV_CONFIG = {
        'id': 'MultiviewPose-v0',
        'pose_model_config': os.path.join('configs', 'w32_256x192_17j_coco.yaml'),
        'algo': 'MAPPO_CTCR_WDL',
        # ===== Main Settings =====
        'in_evaluation': False,  # This is set to False differentiate between training and evaluation
        'reward_dict': ('teamonly', REWARD_DICT['teamonly']),
        # key arguments
        'args': Namespace(
            num_humans=7,
            env_name=f'C{NUM_CAMERAS}_training',
            # Currently supported environments: Blank, SchoolGymDay, grass_plane, Building_small
            map_name='Blank',
            num_envs=NUM_ENVS_PER_WORKER,
        ),
        'gt_observation': {
            'use': False,
            # 'args': {
            #     'gt_noise_scale': 0.,
            #     'affected_by_visibility': False
            # }
        },
        'rule_based_rot': {
            'use': True,
            'args': {'use_gt3d': True, 'num_rule_cameras': None, 'index_list': [0]},
        },
        'ema_action': {
            'use': False,
            # 'args': {
            #     'k_EMA': 0.5,
            #     'action_noise': [1.0, 1.0],
            #     'add_extra_force': False,
            #     'force_configs': {
            #         'F': 30,
            #         'human_safe_distance': 150.,
            #         'camera_safe_distance': 90.,
            #     }
            # }
        },
        'partial_triangulation': {
            'use': True if (NUM_CAMERAS > 2 and 'CTCR' in EXP_MODE) else False,
        },
        'shapley_reward': True if (NUM_CAMERAS > 2 and 'CTCR' in EXP_MODE) else False,
        'shuffle_cam_id': {
            'use': True,
        },
        # ===== Other Misc Settings =====
        'done_when_colliding': {
            'use': False,
            # 'args': {
            #     'threshold': 50.0,
            #     'collision_tolerance': 10,
            # },
        },
        'rot_limit': {
            'use': True,
            'args': {
                'pitch_low': -85.0,
                'pitch_high': 85.0,
                'yaw_low': -360,
                'yaw_high': 360,
            },
        },
        'fix_human': {
            'use': False,
            # 'args': {
            #     'random_init_states': False,
            #     'index_list': None,
            # },
        },
        'place_cam': {
            'use': False,
            # 'args': {
            #     'num_place_cameras': 1,
            #     'place_on_slider': False,
            # },
        },
        'movable_cam': {
            'use': True,
            'args': {'num_movable_cameras': NUM_CAMERAS},
        },
        'scale_human': {
            'use': False,
            # 'args': {
            #     'scale': 1,
            #     'index_list': None,
            # },
        },
        'ego_action': {
            'use': True,
        },
        'air_wall_outer': True,
        'air_wall_inner': {
            'use': False,
            # "args": {
            #     "lower_bound": [-200, -200, 0],
            #     "higher_bound": [200, 200, 500]
            # },
        },
        'aux_rewards': {
            'use': False,
            # 'args': {
            #     'centering': True,
            #     'distance': True,
            #     'obstruction': True,
            #     'iot': True,
            #     'anti_collision': False
            # }
        },
        'remove_info': True,
        'convert_multi_discrete_to_discrete': False,
        'force_single_agent': False,
        'running_normalized_reward': False,
    }

    mappo_ctcr_wdl = Experiment(
        run='PPO',
        name='mappo_ctcr_wdl',
        stop={'timesteps_total': 1.0e6},
        checkpoint_freq=50,
        checkpoint_at_end=True,
        keep_checkpoints_num=None,
        max_failures=0,
        # checkpoint_score_attr="min-custom_metrics/mpjpe_3d_mean",
        local_dir=os.path.join(ROOT_DIR, 'ray_results'),
        config={
            'env': 'active-pose-parallel',
            'framework': 'torch',
            'callbacks': CustomMetricCallback,
            'env_config': ENV_CONFIG,
            # == Sampling ==
            'horizon': 500,
            'rollout_fragment_length': ROLLOUT_FRAGMENT_LENGTH,
            'batch_mode': 'truncate_episodes',
            # == Training ==
            'num_cpus_for_driver': NUM_CPUS_FOR_DRIVER,
            'num_gpus': TRAINER_GPUS,
            'num_workers': NUM_WORKERS,
            'num_gpus_per_worker': NUM_GPUS_PER_WORKER,
            'num_envs_per_worker': NUM_ENVS_PER_WORKER,
            'num_cpus_per_worker': NUM_CPUS_PER_WORKER,
            'train_batch_size': TRAIN_BATCH_SIZE,  # how many steps to collect for training per iteration
            'sgd_minibatch_size': SGD_MINIBATCH_SIZE,
            'gamma': 0.99,
            'shuffle_sequences': False,
            'entropy_coeff': 0,
            'num_sgd_iter': 16,
            'vf_loss_coeff': 0.1,
            'vf_clip_param': 1000.0,
            'grad_clip': 50.0,
            'lr': 5.0e-4,
            'lr_schedule': [
                (0, 5e-4),
                (200e3, 5e-4),
                (200e3, 1e-4),
                (400e3, 1e-4),
                (600e3, 5e-5),
                (600e3, 5e-5),
            ],
            'model': {
                'lstm_use_prev_action': False,
                'lstm_use_prev_reward': False,
                'custom_model': 'aux_rnn_ma_partial',
                'custom_model_config': {
                    # ===== Env Settings =====
                    'num_cameras': NUM_CAMERAS,
                    'max_num_humans': 7,
                    'masking_target': True,  # to assert observability mask on unobservable targets
                    # ===== Model Architecture =====
                    'cell_size': 128,
                    'actnet_hiddens': [128],
                    'vfnet_hiddens': [128],
                    'fcnet_hiddens': [128, 128, 128],
                    'mdn_hiddens': [128, 128],
                    'mdn_num_gaussians': 16,
                    # ===== WDL related =====
                    'prediction_steps': 1,
                    'merge_back': True if 'WDL' in EXP_MODE else False,
                    'coordinate_scale': 500.0,
                    'prediction_loss_coeff': (5.0 if 'WDL' in EXP_MODE else 0.0),  # Total Coeff
                    'pred_coeff_dict': {  # Sub Coeffs
                        'coeff_cam_pred': (1.0 if 'WDL' in EXP_MODE else 0.0),
                        'coeff_other_cam_pred': (1.0 if 'WDL' in EXP_MODE else 0.0),
                        'coeff_reward_pred': (1.0 if 'WDL' in EXP_MODE else 0.0),
                        'coeff_human_pred': (1.0 if 'WDL' in EXP_MODE else 0.0),
                        'coeff_obstructor_pred': (0.1 if 'WDL' in EXP_MODE else 0.0),
                    },
                },
                # # == Post-processing LSTM ==
                'max_seq_len': ROLLOUT_FRAGMENT_LENGTH,
            },
            'multiagent': MULTI_AGENT,
        },
    )

    return mappo_ctcr_wdl
