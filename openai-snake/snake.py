import torch
import torch.optim as optim
import numpy as np
import model


class Snake(object):
    actions = ['UP', 'DOWN', 'LEFT', 'RIGHT']

    def __init__(
        self,
        length, 
        box_dimensions,
        BATCH_SIZE=128,
        GAMMA=0.999,
        EPS_START=0.9,
        EPS_END=0.05,
        EPS_DECAY=200,
        TARGET_UPDATE=10,
    ):
        """
        Constructs a new snake.
        
        :param      position:  The array of positions of snakes body, head indexed at 0
        :type       position:  np.array
        """
        self.length = length
        self.box_dimensions = box_dimensions
        self.height = box_dimensions[0]
        self.width = box_dimensions[1]
        self.epsilon = 0.01
        self._reset()

    def _reset(self):
        self.i_pos = self.box_dimensions / 2
        self.body_position = [np.array([ self.i_pos[1], self.i_pos[0] - x ]) for x in range(self.length)]

    def _set_prize_position(self, position):
        self.prize_position = position

    def act(self, state):
        if np.random.rand() < self.epsilon:
            action = self.sample_move()
        else:
            action = self.move()
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
        return move

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


    def setup_model(self):
        self.policy_net = DQN(self.height, self.width, len(self.actions) - 1).to(self.device)
        self.target_net = DQN(self.height, self.width, len(self.actions) - 1).to(self.device)
        self.target_net.load_state_dict(self.policy_net.state_dict())
        self.target_net.eval()

        self.optimizer = optim.RMSprop(self.policy_net.parameters())
        self.memory = ReplayMemory(10000)
        self.steps_done = 0

    def select_action(state, device):
        sample = np.random.random()
        eps_threshold = self.EPS_END + (self.EPS_START - self.EPS_END) * \
            math.exp(-1. * self.steps_done / self.EPS_DECAY)
        self.steps_done += 1
        if sample > eps_threshold:
            with torch.no_grad():
                # t.max(1) will return largest column value of each row.
                # second column on max result is index of where max element was
                # found, so we pick action with the larger expected reward.
                return policy_net(state).max(1)[1].view(1, 1)
        else:
            return torch.tensor([[random.randrange(n_actions)]], device=device, dtype=torch.long)
