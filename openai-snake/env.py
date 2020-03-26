import gym
from gym import error, spaces, utils
from gym.utils import seeding

class SnakeBoardEnv(gym.Env):
    metadata = {'render.modes': ['human']}

    def __init__(self, initial_snake_length, width_height):
        self._set_action_space()
        self.length = initial_snake_length
        self.head_pos = width_height / 2
        self._set_observation_space(observation)

        action = self.action_space.sample(width_height)
        observation, _reward, done, _info = self.step(action)
        assert not done

        self._set_observation_space(observation)

        self.seed()

    def _set_action_space(self, box_wl):
        self.action_space = spaces.Box(low=np.zeroes(2), high=box_wl, dtype=np.float32)
        return self.action_space

    def _set_observation_space(self);


    def step(self, action):
        """
        This method is the primary interface between environment and agent.
        Paramters: 
                action: int
                                the index of the respective action (if action space is discrete)
        Returns:
                output: (array, float, bool)
                                information provided by the environment about its current state:
                                (observation, reward, done)
        """
        pass

    def reset(self):
        """
        This method resets the environment to its initial values.
        Returns:
                observation:    array
                                                the initial state of the environment
        """
        pass

    def render(self, mode='human', close=False):
        """
        This methods provides the option to render the environment's behavior to a window 
        which should be readable to the human eye if mode is set to 'human'.
        """
        pass

    def close(self):
        """Override close in your subclass to perform any necessary cleanup.
        Environments will automatically close() themselves when
        garbage collected or when the program exits.
        """
        pass

    def seed(self, seed=None):
        """Sets the seed for this env's random number generator(s).
        Note:
            Some environments use multiple pseudorandom number generators.
            We want to capture all such seeds used in order to ensure that
            there aren't accidental correlations between multiple generators.
        Returns:
            list<bigint>: Returns the list of seeds used in this env's random
              number generators. The first value in the list should be the
              "main" seed, or the value which a reproducer should pass to
              'seed'. Often, the main seed equals the provided 'seed', but
              this won't be true if seed=None, for example.
        """
        return