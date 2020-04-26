import numpy as np
import torch
import torch.optim as optim
import torch.nn.functional as F
from itertools import count
from model import DQN, ReplayMemory, Transition
import random
import math as m


class Snake(object):
    actions = ['FORWARD', 'LEFT', 'RIGHT']
    orientations = ['UP', 'DOWN', 'LEFT', 'RIGHT']
    def __init__(
        self,
        length, 
        box_dimensions,
        device,
        BATCH_SIZE=128,
        GAMMA=0.999,
        EPS_START=0.9,
        EPS_END=0.05,
        EPS_DECAY=200,
        TARGET_UPDATE=10,
    ):
        self.length = length
        self.box_dimensions = box_dimensions
        self.width = box_dimensions[0]
        self.height = box_dimensions[1]
        self.epsilon = 0.01
        self.orientation = 'LEFT'
        self.device = device

        self.policy_net = DQN(self.height, self.width, len(self.actions)).to(device)
        self.target_net = DQN(self.height, self.width, len(self.actions)).to(device)
        self.optimizer = optim.RMSprop(self.policy_net.parameters())
        self.memory = ReplayMemory(10000)
        self.steps_done = 0
        self.cumulative_reward = 0.0
        self.target_net.load_state_dict(self.policy_net.state_dict())
        self.target_net.eval()


        self.BATCH_SIZE = BATCH_SIZE
        self.GAMMA = GAMMA
        self.EPS_START = EPS_START
        self.EPS_END = EPS_END
        self.EPS_DECAY = EPS_DECAY
        self.TARGET_UPDATE = TARGET_UPDATE
        self._reset()

    def _reset(self):
        self.i_pos = self.box_dimensions / 2
        self.body_position = [np.array([ self.i_pos[0] - x, self.i_pos[1] ]) for x in range(self.length)]

    def act(self, state):
        a = self.select_action(state)
        return self.actions[a.item()]

    def _convert_action_to_tensor(self, action):
        return torch.tensor(self.actions.index(action)).view(1, -1)

    def _convert_move_to_point(self, move):
        if move == 'FORWARD':
            return self._handle_forward(move)
        elif move == 'LEFT':
            return self._handle_left(move)
        elif move == 'RIGHT':
            return self._handle_right(move)
        else:
            raise ValueError('Invalid move')

    def is_colliding(self, new_position):
        # can only collide 2 units of length from head
        for inx, elt in enumerate(self.body_position[2:]):
            if np.array_equal(np.array(new_position, dtype=int), np.array(elt, dtype=int)):
                return True
        return False

    def select_action(self, state):
        sample = np.random.random()
        eps_threshold = self.EPS_END + (self.EPS_START - self.EPS_END) * \
            m.exp(-1. * self.steps_done / self.EPS_DECAY)
        self.steps_done += 1
        if sample > eps_threshold:
            with torch.no_grad():
                state_tensor = torch.tensor(
                    state,
                    device=self.device
                ).double()
                state_shape = list(state_tensor.size())
                state_tensor = state_tensor.view(1, 1, state_shape[0], state_shape[1])
                # t.max(1) will return largest column value of each row.
                # second column on max result is index of where max element was
                # found, so we pick action with the larger expected reward.
                return self.policy_net(state_tensor).max(1)[1].view(1, 1)
        else:
            return torch.tensor(
                [[random.randrange(len(self.actions))]],
                device=self.device,
                dtype=torch.long)

    def _handle_forward(self, move):
        x_pos, y_pos = self.body_position[0]
        if self.orientation == 'UP':
            return np.array([x_pos, y_pos + 1])
        elif self.orientation == 'DOWN':
            return np.array([x_pos, y_pos - 1])
        elif self.orientation == 'LEFT':
            return np.array([x_pos + 1, y_pos])
        elif self.orientation == 'RIGHT':
            return np.array([x_pos - 1, y_pos])
        else:
            raise ValueError('Invalid orientation')

    def _handle_left(self, move):
        x_pos, y_pos = self.body_position[0]
        if self.orientation == 'UP':
            self.orientation = 'LEFT'
            return np.array([x_pos + 1, y_pos])

        elif self.orientation == 'DOWN':
            self.orientation = 'RIGHT'
            return np.array([x_pos - 1, y_pos])

        elif self.orientation == 'LEFT':
            self.orientation = 'DOWN'
            return np.array([x_pos, y_pos - 1])

        elif self.orientation == 'RIGHT':
            self.orientation = 'UP'
            return np.array([x_pos, y_pos + 1])
        else:
            raise ValueError('Invalid orientation')

    def _handle_right(self, move):
        x_pos, y_pos = self.body_position[0]
        if self.orientation == 'UP':
            self.orientation = 'RIGHT'
            return np.array([x_pos - 1, y_pos])

        elif self.orientation == 'DOWN':
            self.orientation = 'LEFT'
            return np.array([x_pos + 1, y_pos])

        elif self.orientation == 'LEFT':
            self.orientation = 'UP'
            return np.array([x_pos, y_pos + 1])

        elif self.orientation == 'RIGHT':
            self.orientation = 'DOWN'
            return np.array([x_pos, y_pos - 1])
        else:
            raise ValueError('Invalid orientation')

    def optimize_model(self):
        if len(self.memory) < self.BATCH_SIZE:
            return
        transitions = self.memory.sample(self.BATCH_SIZE)
        # Transpose the batch (see https://stackoverflow.com/a/19343/3343043 for
        # detailed explanation). This converts batch-array of Transitions
        # to Transition of batch-arrays.
        batch = Transition(*zip(*transitions))
        # Compute a mask of non-final states and concatenate the batch elements
        # (a final state would've been the one after which simulation ended)
        non_final_mask = torch.tensor(
            tuple(map(lambda s: s is not None, batch.next_state)),
            device=self.device,
            dtype=torch.bool
        )
        non_final_next_states = torch.cat([torch.from_numpy(s) for s in batch.next_state if s is not None])
        state_batch = torch.cat([torch.from_numpy(s) for s in batch.state if s is not None])
        action_batch = torch.cat(batch.action)
        reward_batch = torch.cat(batch.reward)

        # Compute Q(s_t, a) - the model computes Q(s_t), then we select the
        # columns of actions taken. These are the actions which would've been taken
        # for each batch state according to policy_net
        state_action_values = self.policy_net(state_batch).gather(1, action_batch)

        # Compute V(s_{t+1}) for all next states.
        # Expected values of actions for non_final_next_states are computed based
        # on the "older" target_net; selecting their best reward with max(1)[0].
        # This is merged based on the mask, such that we'll have either the expected
        # state value or 0 in case the state was final.
        next_state_values = torch.zeros(self.BATCH_SIZE, device=device)
        next_state_values[non_final_mask] = self.target_net(non_final_next_states).max(1)[0].detach()
        # Compute the expected Q values
        expected_state_action_values = (next_state_values * GAMMA) + reward_batch

        # Compute Huber loss
        loss = F.smooth_l1_loss(state_action_values, expected_state_action_values.unsqueeze(1))

        # Optimize the model
        self.optimizer.zero_grad()
        loss.backward()
        for param in self.policy_net.parameters():
            param.grad.data.clamp_(-1, 1)
        self.optimizer.step()
