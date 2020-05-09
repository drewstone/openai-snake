import torch
import numpy as np
from env import SnakeBoardEnv
from snake import Snake
import matplotlib.pyplot as plt 
import matplotlib.animation as animation 

plt.style.use('dark_background')

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

fig = plt.figure() 
ax = fig.gca()

def render(env, episode, score, steps):
    """
    This method renders the environment in a matplotlib plot.
    """
    board = env._get_snake_board()
    # draw gridlines
    ax.grid(which='major', axis='both', linestyle='-', color='k', linewidth=2)
    ax.set_xticks(np.arange(-0.5, env._snake.width, 1));
    ax.set_yticks(np.arange(-0.5, env._snake.height, 1));
    plt.setp(ax.get_xticklabels(), visible=False)
    plt.setp(ax.get_yticklabels(), visible=False)
    ax.tick_params(axis='both', which='both', length=0)
    ax.set_title("Episode {} | Score {} | Steps {}".format(episode, score, steps))
    # return image
    im = ax.imshow(board, animated=True)
    return [im]

# initialization function 
def init(): 
	return render(env, snake.episode, snake.cumulative_reward, snake.steps_done)

# animation function 
def animate(i):
    state = env._get_snake_board()
    # Select and perform an action
    action = snake.act(state)
    observation, reward, done = env.step(action)
    snake.process_reward(reward)
    reward = torch.tensor([reward], device=device)

    # Observe new state
    if not done:
        next_state = env._get_snake_board()
        render(env, snake.episode, snake.cumulative_reward, snake.steps_done)
    else:
        env.reset()

    # Observe new state
    next_state = env._get_snake_board()

    # Store the transition in memory
    snake.memory.push(state, snake._convert_action_to_tensor(action), next_state, reward)

    # Perform one step of the optimization (on the target network)
    snake.optimize_model()
    # Update the target network, copying all weights and biases in DQN
    if snake.steps_done % snake.TARGET_UPDATE == 0:
        snake.target_net.load_state_dict(snake.policy_net.state_dict())
    
    return render(env, snake.episode, snake.cumulative_reward, snake.steps_done)
	
# setting a title for the plot 
plt.title('Creating a growing coil with matplotlib!') 
# hiding the axis details 
plt.axis('off') 

# call the animator	 
anim = animation.FuncAnimation(
    fig,
    animate,
    np.arange(1000),
    init_func=init,
    # blit=True,
    # repeat=False
)

# save the animation as mp4 video file 
anim.save('anim.mp4', writer='imagemagick', fps=30)