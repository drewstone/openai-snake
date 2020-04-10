import torch
import torch.nn.functional as F
from itertools import count
from model import Transition


def optimize_model(
	optimizer,
	memory,
	policy_net,
	target_net,
	device,
	BATCH_SIZE,
	GAMMA
):
	if len(memory) < BATCH_SIZE:
		return
	transitions = memory.sample(BATCH_SIZE)
	# Transpose the batch (see https://stackoverflow.com/a/19343/3343043 for
	# detailed explanation). This converts batch-array of Transitions
	# to Transition of batch-arrays.
	batch = Transition(*zip(*transitions))

	# Compute a mask of non-final states and concatenate the batch elements
	# (a final state would've been the one after which simulation ended)
	non_final_mask = torch.tensor(tuple(map(lambda s: s is not None,
										  batch.next_state)), device=device, dtype=torch.bool)
	non_final_next_states = torch.cat([s for s in batch.next_state
												if s is not None])
	state_batch = torch.cat(batch.state)
	action_batch = torch.cat(batch.action)
	reward_batch = torch.cat(batch.reward)

	# Compute Q(s_t, a) - the model computes Q(s_t), then we select the
	# columns of actions taken. These are the actions which would've been taken
	# for each batch state according to policy_net
	state_action_values = policy_net(state_batch).gather(1, action_batch)

	# Compute V(s_{t+1}) for all next states.
	# Expected values of actions for non_final_next_states are computed based
	# on the "older" target_net; selecting their best reward with max(1)[0].
	# This is merged based on the mask, such that we'll have either the expected
	# state value or 0 in case the state was final.
	next_state_values = torch.zeros(BATCH_SIZE, device=device)
	next_state_values[non_final_mask] = target_net(non_final_next_states).max(1)[0].detach()
	# Compute the expected Q values
	expected_state_action_values = (next_state_values * GAMMA) + reward_batch

	# Compute Huber loss
	loss = F.smooth_l1_loss(state_action_values, expected_state_action_values.unsqueeze(1))

	# Optimize the model
	optimizer.zero_grad()
	loss.backward()
	for param in policy_net.parameters():
		param.grad.data.clamp_(-1, 1)
	optimizer.step()

def train(
	env,
	memory,
	device,
	policy_net,
	target_net,
	TARGET_UPDATE,
	num_episodes=50
):
  	episode_durations = []
	for i_episode in range(num_episodes):
		# Initialize the environment and state
		env.reset()
		for t in count():
  			state = env._get_snake_board()
			# Select and perform an action
			action = env._snake.act(env._get_snake_board())
			observation, _reward, done = env.step(action)
			reward = torch.tensor([reward], device=device)

			# Observe new state
			if not done:
				next_state = env._get_snake_board()
			else:
				next_state = None

			# Store the transition in memory
			memory.push(state, action, next_state, reward)

			# Move to the next state
			state = next_state

			# Perform one step of the optimization (on the target network)
			optimize_model()
			if done:
				episode_durations.append(t + 1)
				# plot_durations()
				break
		# Update the target network, copying all weights and biases in DQN
		if i_episode % TARGET_UPDATE == 0:
			target_net.load_state_dict(policy_net.state_dict())
	print("Complete")