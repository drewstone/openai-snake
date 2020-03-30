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
            self.i_pos[1],
            self.i_pos[0] - x,
        ]) for x in range(length)]
        self.epsilon = epsilon

    def set_prize_position(self, position):
        self.prize_position = position

    def act(self):
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