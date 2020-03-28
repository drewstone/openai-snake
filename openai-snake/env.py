import gym
from gym import error, spaces, utils
from gym.utils import seeding
import numpy as np


class Snake(object):
    actions = ['UP', 'DOWN', 'LEFT', 'RIGHT']

    def __init__(self, length, box_dimensions, epsilon=0.01):
        """
        Constructs a new snake.
        
        :param      position:  The array of positions of snakes body, head indexed at 0
        :type       position:  np.array
        """
        self.i_pos = box_dimensions / 2
        self.body_position = [np.array([
            self.i_pos[0] - x,
            self.i_pos[1]
        ]) for x in range(length)]
        self.epsilon = epsilon

    def set_prize_position(self, position):
        self.prize_position = position

    def act(self):
        if np.random.rand() < self.epsilon:
            action = self.sample_move()
        else:
            action = self.move()
        if np.array_equal(action, self.prize_position):
            self.body_position = [action] + self.body_position
        else:
            self.body_position.pop()
            self.body_position = [action] + self.body_position
        return action

    def move(self):
        # until we have an actual, strategy sample random move
        return self.sample_move()

    def sample_move(self):
        move = np.random.choice(self.actions)
        # cant move into the portion one before the head,
        # we sample moves until that isn't the case
        while np.array_equal(self._convert_move_to_point(move), np.array(self.body_position[1])):
            move = np.random.choice(self.actions)
        return self._convert_move_to_point(move)

    def _convert_move_to_point(self, move):
        x_pos, y_pos = self.body_position[0]
        if move == 'UP':
            return np.array([x_pos, y_pos + 1])
        elif move == 'DOWN':
            return np.array([x_pos, y_pos - 1])
        elif move == 'LEFT':
            return np.array([x_pos - 1, y_pos])
        elif move == 'RIGHT':
            return np.array([x_pos + 1, y_pos])
        else:
            raise ValueError('Invalid move')


    def is_colliding(self, new_position):
        # can only collide 2 units of length from head
        for inx, elt in enumerate(self.body_position[2:]):
            if np.array_equal(np.array(new_position), np.array(elt)):
                return True
        return False

class SnakeBoardEnv(gym.Env):

    def __init__(self, box_dimensions, snake):
        # set observation state, equal to action space as we assume snake sees everything
        self.width = box_dimensions[0]
        self.height = box_dimensions[1]
        self._set_observation_space(box_dimensions)
        self._prize_position = self._observation_space.sample()
        self._snake = snake
        self.seed()

    def _set_observation_space(self, box_wl):
        self._observation_space = spaces.Box(low=np.zeros(2), high=np.array(box_wl), dtype=np.float32)
        return self.observation_space

    def _out_of_bounds(self, action):
        if action[0] >= self.width or action[1] >= self.height:
            return True
        elif action[0] < 0 or action[1] < 0:
            return True
        else:
            return False

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
        if self._snake.is_colliding(action) or self._out_of_bounds(action):
            return [], -1.0, True
        else:
            return [self._prize_position, self._observation_space], 0, False

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
    initial_snake_length = 3
    box_dimensions = np.array([10, 10])
    snake = Snake(initial_snake_length, box_dimensions)
    env = SnakeBoardEnv(box_dimensions, snake)
    prize_position = np.random.randint(10, size=2)
    while snake.is_colliding(prize_position):
        prize_position = np.random.randint(10, size=2)
    snake.set_prize_position(prize_position)
    done = False
    for _ in range(10):
        action = snake.act()
        print(snake.body_position)
        observation, _reward, done = env.step(action)
        if done:
            print('Crashed into oneself: {}', env.snake.body_position)


