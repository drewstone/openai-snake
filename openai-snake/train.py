import torch
import torch.nn.functional as F
import numpy as np
from itertools import count
from model import Transition
import matplotlib.pyplot as plt
from matplotlib import colors


def render(env):
    """
    This method renders the environment in a matplotlib plot.
    """
    board = env._get_snake_board()
    fig, ax = plt.subplots()
    ax.imshow(board)

    # draw gridlines
    ax.grid(which='major', axis='both', linestyle='-', color='k', linewidth=2)
    ax.set_xticks(np.arange(-0.5, env._snake.width, 1));
    ax.set_yticks(np.arange(-0.5, env._snake.height, 1));
    plt.setp(ax.get_xticklabels(), rotation=45, ha="right",
        rotation_mode="anchor")

    plt.show()

def train(
    env,
    snake,
    device,
    num_episodes=1
):
    episode_durations = []
    for i_episode in range(num_episodes):
        print("Starting episode {}, steps done {}...".format(i_episode, snake.steps_done))
        # Initialize the environment and state
        env.reset()
        render(env)
        for t in count():
            state = env._get_snake_board()
            # Select and perform an action
            action = snake.act(state)
            observation, reward, done = env.step(action)
            reward = torch.tensor([reward], device=device)

            # Observe new state
            if not done:
                next_state = env._get_snake_board()
                render(env)
            else:
                break

            # Store the transition in memory

            
            snake.memory.push(state, snake._convert_action_to_tensor(action), next_state, reward)

            # Move to the next state
            state = next_state

            # Perform one step of the optimization (on the target network)
            snake.optimize_model()
            if done:
                episode_durations.append(t + 1)
                # plot_durations()
                break
        # Update the target network, copying all weights and biases in DQN
        if i_episode % snake.TARGET_UPDATE == 0:
            snake.target_net.load_state_dict(snake.policy_net.state_dict())
    print("Complete")
