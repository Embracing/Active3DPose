import gym


class WrapperBase(gym.Wrapper):
    def __init__(self, env):
        self.env = env
        self.reward_range = self.env.reward_range
        self.metadata = self.env.metadata

    @property
    def observation_space(self):
        return self.env.observation_space

    @observation_space.setter
    def observation_space(self, value):
        self.env.observation_space = value

    @property
    def action_space(self):
        return self.env.action_space

    @action_space.setter
    def action_space(self, value):
        self.env.action_space = value
