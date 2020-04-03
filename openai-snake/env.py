import gym
from gym import error, spaces, utils
from gym.utils import seeding
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import colors
from snake import Snake

class SnakeBoardEnv(gym.Env):

    def __init__(self, box_dimensions, snake):
        # set observation state, equal to action space as we assume snake sees everything
        self.height = box_dimensions[0]
        self.width = box_dimensions[1]
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

    def _select_prize_pos(self):
        prize_position = np.random.randint(10, size=2)
        while this._snake.is_colliding(prize_position):
            prize_position = np.random.randint(10, size=2)
        return prize_position

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
        new_position = snake._convert_move_to_point(action)
        # update body position of the snake
        if np.array_equal(new_position, self._prize_position):
            self._snake.body_position = [new_position] + self._snake.body_position
            self._prize_position = self._select_prize_position()

        else:
            self._snake.body_position.pop()
            self._snake.body_position = [new_position] + self._snake.body_position
        # 
        if self._snake.is_colliding(new_position) or self._out_of_bounds(new_position):
            return [], -1.0, True
        else:
            return [self._prize_position, self._observation_space], 0, False

    def reset(self):
        """
        This method resets the environment to its initial values.
        """
        pass

    def render(self):
        """
        This method renders the environment in a matplotlib plot.
        """
        harvest = np.zeros((self.width, self.height))
        for inx, elt in enumerate(self._snake.body_position):
            print(elt)
            harvest[int(elt[0])][int(elt[1])] = 10.0
        harvest[int(prize_position[0])][int(prize_position[1])] = 5.0
        fig, ax = plt.subplots()
        ax.imshow(harvest)

        # draw gridlines
        ax.grid(which='major', axis='both', linestyle='-', color='k', linewidth=2)
        ax.set_xticks(np.arange(-0.5, 10, 1));
        ax.set_yticks(np.arange(-0.5, 10, 1));
        plt.setp(ax.get_xticklabels(), rotation=45, ha="right",
         rotation_mode="anchor")

        plt.show()



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
        env.render()
        action = snake.act()
        print(action)
        observation, _reward, done = env.step(action)
        if done:
            print('Crashed into oneself or the barrier: {}', env._snake.body_position)
            break
