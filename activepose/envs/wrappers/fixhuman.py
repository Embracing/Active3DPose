import time

from .base import WrapperBase
from .utils import bind


# from activepose.envs.internal import MultiHumanController


# def reset_human_overwritten(self):
#     """
#     Reset Humans
#     Overwritter
#     """
#     # if not hasattr(self, 'controller'):
#     #     # don not create again
#     #     # if-clause assumes number of humans does not change.
#     #     # Otherwise, remove this if-caluse

#     self.controller = MultiHumanController(
#         self.env_interaction,
#         self.human_config,
#         self.env_config,
#         action_setting=self.human_config['action']
#     )

#     # waiting for humans to be ready
#     print('Waiting for humans to be ready (1s)')
#     time.sleep(1)

#     # random reset loc, rot and mesh
#     # set mesh color according to human_config
#     if hasattr(self, 'random_init_states') and self.random_init_states:
#         self.controller.random_set_human_states_all()

#     # reset action
#     # freeze walking human, restore walking insided env.step()
#     self.controller.reset_action_all()
#     self.controller.freeze_walk_all()


class FixHuman(WrapperBase):
    """
    Can be placed in any order
    1: Stationary scene.
    """

    def __init__(
        self,
        env,
        # random_init_states=False,
        # random_init_loc_rot,
        index_list=None,
        **kwargs,
    ):
        super().__init__(env)

        # bind(env.unwrapped, reset_human_overwritten)

        # env.unwrapped.random_init_states = random_init_states
        # env.unwrapped.num_fixed_humans = num_fixed_humans

        if index_list is None:
            # all humans are fixed
            env.human_config['action'] = ['fixed' for _ in range(env.num_humans)]
        else:
            assert len(index_list) <= env.num_humans
            assert max(index_list) < env.num_humans

            for idx in index_list:
                env.human_config['action'][idx] = 'fixed'
