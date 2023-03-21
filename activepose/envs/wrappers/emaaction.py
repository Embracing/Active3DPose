from .base import WrapperBase


class EMAAction(WrapperBase):
    def __init__(self, env, config_dict):
        super().__init__(env)
        env.unwrapped.EMA_action = True

        if config_dict:
            env.unwrapped.k_EMA = config_dict['k_EMA']
        else:
            env.unwrapped.k_EMA = 0.0

        if config_dict.get('action_noise', False):
            env.unwrapped.action_noise = config_dict['action_noise']

        if config_dict.get('add_extra_force', False):
            env.unwrapped.add_extra_force = True
            if config_dict.get('force_configs', False):
                env.unwrapped.force_configs = config_dict['force_configs']
            else:
                env.unwrapped.force_configs = {
                    'F': 50,
                    'human_safe_distance': 150.0,
                    'camera_safe_distance': 90.0,
                }
        else:
            env.unwrapped.add_extra_force = False
