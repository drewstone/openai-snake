import torch
import torch.nn.functional as F
from itertools import count
from model import Transition

def train(
    env,
    snake,
    device,
    num_episodes=50
):
    episode_durations = []
    for i_episode in range(num_episodes):
        print("Starting episode {}...".format(i_episode))
        # Initialize the environment and state
        env.reset()
        # env.render()
        for t in count():
            state = env._get_snake_board()
            # Select and perform an action
            action = snake.act(state)
            observation, reward, done = env.step(action)
            reward = torch.tensor([reward], device=device)

            # Observe new state
            if not done:
                next_state = env._get_snake_board()
                # env.render()
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
