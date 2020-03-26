import gym
from gym import error, spaces, utils
from gym.utils import seeding

class Snake(object):
    actions = ['UP', 'DOWN', 'LEFT', 'RIGHT']

    def __init__(self, length, epsilon=0.01):
        """
        Constructs a new snake.
        
        :param      position:  The array of positions of snakes body, head indexed at 0
        :type       position:  np.array
        """
        self.intial_position = box_dimensions[1] / 2
        self.body_position = [
            np.array( [self.intial_position[0] - x, self.intial_position[1] ])
            for x in range(length)
        ]
        self.epsilon = epsilon

    def act(self):
        if np.random.rand() < self.epsilon:
            action = self.sample_move(self)
        else:
            # until we have an actual strategy, random sampling
            action = self.sample_move(self)
            return action

    def sample_move(self):
        move = np.random.choice(actions)
        while self._convert_move_to_point(move) == self.body_position[1]

    def _convert_move_to_point(self, move):
        x_pos, y_pos = self.body_position[0]
        if move == 'UP':
            return x_pos, y_pos + 1
        elif move == 'DOWN':
            return x_pos, y_pos - 1
        elif move == 'LEFT':
            return x_pos - 1, y_pos
        elif move == 'RIGHT':
            return x_pos + 1, y_pos
        else:
            raise ValueError('Invalid move')


    def is_colliding(self, new_position):
        # can only collide 2 units of length from head
        for inx, elt in enumerate(self.position[2:]):
            if new_position == elt:
                return True
        return False

class SnakeBoardEnv(gym.Env):

    def __init__(self, box_dimensions, snake):
        # set observation state, equal to action space as we assume snake sees everything
        self._set_observation_space(box_dimensions)
        self._prize_position = self._observation_space.sample()
        self._snake = snake
        self.seed()

    def _set_observation_space(self, box_wl);
        self.observation_space = spaces.Box(low=np.zeroes(2), high=box_wl, dtype=np.float32)
        return self.observation_space

    def _sample_action(self):
        return self._snake.act()

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
        if self._snake.is_colliding(action):
            return -1
        else:
        pass

    def reset(self):
        """
        This method resets the environment to its initial values.
        Returns:
                observation:    array
                                                the initial state of the environment
        """
        pass


if __name__ == '__main__':
    # initialize snake agent 
    snake = Snake(initial_snake_length)
    env = SnakeBoardEnv(box_dimensions, snake)
    done = False
    while not sim_done:
    action = self.snake.act()
    observation, _reward, done, _info = self.step(action)
    assert not done
