import torch
import numpy as np
from env import SnakeBoardEnv
from snake import Snake
from train import train


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
box_dimensions = np.array([10, 10])
snake = Snake(
    initial_snake_length,
    box_dimensions,
    device,
    BATCH_SIZE,
    GAMMA,
    EPS_START,
    EPS_END,
    EPS_DECAY,
    TARGET_UPDATE
)
env = SnakeBoardEnv(box_dimensions, snake)

train(env, snake, device)