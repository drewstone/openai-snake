import torch
import gym
from gym import error, spaces, utils
from gym.utils import seeding
import numpy as np
from snake import Snake


class SnakeBoardEnv(gym.Env):
    def __init__(self, box_dimensions, snake):
        # set observation state, equal to action space as we assume snake sees everything
        self.width = box_dimensions[0]
        self.height = box_dimensions[1]
        self._set_observation_space(box_dimensions)
        self._snake = snake
        self._prize_position = self._select_prize_pos()
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
        prize_position = np.random.randint(min(self.width, self.height), size=2)
        while self._snake.is_colliding(np.flip(np.array(prize_position, dtype=int))):
            prize_position = np.random.randint(10, size=2)
        return prize_position

    def _get_snake_board(self):
        board = np.zeros((self.height, self.width))
        head = self._snake.body_position[0]
        board[int(head[1])][int(head[0])] = 15.0
        for inx, elt in enumerate(self._snake.body_position[1:]):
            board[int(elt[1])][int(elt[0])] = 10.0
        board[int(self._prize_position[0])][int(self._prize_position[1])] = 3.0
        return board

    def _get_last_screen(self):
        return self._last_screen

    def step(self, action):
        self._last_screen = self._get_snake_board()
        new_position = self._snake._convert_move_to_point(action)
        if self._snake.is_colliding(new_position) or self._out_of_bounds(new_position):
            return None, -1.0, True

        # TODO: Figuring out why the prize position needs to be inverted
        # TODO: (there's some bad axes setting on the game)
        # update body position of the snake
        flipped_prize_pos = np.flip(np.array(self._prize_position, dtype=int))
        if np.array_equal(np.array(new_position, dtype=int), flipped_prize_pos):
            print("The snake captured the prize!")
            self._snake.body_position = [new_position] + self._snake.body_position
            self._prize_position = self._select_prize_pos()
            return self._get_snake_board(), 1, True
        else:
            self._snake.body_position.pop()
            self._snake.body_position = [new_position] + self._snake.body_position
            return self._get_snake_board(), 0, False

    def reset(self):
        """
        This method resets the environment to its initial values.
        """
        self._snake._reset()
        self._prize_position = self._select_prize_pos()


if __name__ == '__main__':
    BATCH_SIZE = 128
    GAMMA = 0.999
    EPS_START = 0.9
    EPS_END = 0.05
    EPS_DECAY = 200
    TARGET_UPDATE = 10

    # if gpu is to be used
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    # initialize snake agent
    initial_snake_length = 3
    # width by height
    box_dimensions = np.array([20, 10])
    snake = Snake(
        initial_snake_length,
        box_dimensions,
        BATCH_SIZE,
        GAMMA,
        EPS_START,
        EPS_END,
        EPS_DECAY,
        TARGET_UPDATE
    )
    env = SnakeBoardEnv(box_dimensions, snake)
    done = False
    for _ in range(10):
        env.render()
        action = snake.act(env._get_snake_board())
        observation, _reward, done = env.step(action)
        if done:
            print('Crashed into oneself or the barrier: {}', env._snake.body_position)
            break
