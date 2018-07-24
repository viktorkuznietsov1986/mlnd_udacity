import copy
import numpy as np
from enum import Enum

# Defines the infrastructure for robot moves.
dir_sensors = {'u': ['l', 'u', 'r'], 'r': ['u', 'r', 'd'],
               'd': ['r', 'd', 'l'], 'l': ['d', 'l', 'u'],
               'up': ['l', 'u', 'r'], 'right': ['u', 'r', 'd'],
               'down': ['r', 'd', 'l'], 'left': ['d', 'l', 'u']}
dir_move = {'u': [0, 1], 'r': [1, 0], 'd': [0, -1], 'l': [-1, 0],
            'up': [0, 1], 'right': [1, 0], 'down': [0, -1], 'left': [-1, 0]}
dir_reverse = {'u': 'd', 'r': 'l', 'd': 'u', 'l': 'r',
               'up': 'd', 'right': 'l', 'down': 'u', 'left': 'r'}

# integer values for directions
dir_int_mask = {'u': 1, 'r': 2, 'd': 4, 'l': 8,
                'up': 1, 'right': 2, 'down': 4, 'left': 8}


class Steering(Enum):
    '''
    Enum class for changing direction.
    '''
    L, F, R = (-1, 0, 1)  # Go left, front, right.

    def __str__(self):
        return self.name


class Direction(Enum):
    '''
    Enum class encapsulating the heading direction.
    '''
    U, R, D, L = range(4)  # up, right, down, left.

    def reverse(self):
        return Direction((self.value + 2) % 4)

    def change(self, steering):
        return Direction((self.value + steering.value) % 4)

    def dir_move(self):
        return dir_move[self.value]

    def change_direction(self, steering):
        diff = steering.value - self.value

        if diff == 3:
            diff = -1
        elif diff == -3:
            diff = 1

        return Steering(diff)

    def __str__(self):
        return self.name


class Heading:
    def __init__(self, direction, pos):
        self.direction = direction
        self.pos = pos

    def __str__(self):
        return '{} at ({},{})'.format(self.direction.name, self.pos[0], self.pos[1])

    def change(self, steering, steps):
        direction = self.direction.change(steering)
        direction_move = direction.dir_move()
        pos = [self.pos[i] * direction_move[i] * steps for i in range(2)]
        return Heading(direction, pos)

    def move_forward(self, steps=1):
        return self.change(Steering.F, steps)

    def move_left(self, steps=1):
        return self.change(Steering.L, steps)

    def move_right(self, steps=1):
        return self.change(Steering.R, steps)

    def move_backward(self, steps=1):
        return self.reverse().move_forward(steps)

    def reverse(self):
        return Heading(self.direction.reverse(), self.pos)


class SensorInterpreter:
    def __init__(self, sensors):
        self.sensors = sensors

    def distance(self, steering):
        steering_sensor_idx = {
            Steering.L: 0,
            Steering.F: 1,
            Steering.R: 2
        }

        return self.sensors[steering_sensor_idx[steering]]

    def is_dead_end(self):
        return max(self.sensors) == 0

    def is_one_way(self):
        return self.sensors[0] == 0 and self.sensors[1] > 0 and self.sensors[2] == 0

    def can_go(self, direction, steps):
        directions = ['u', 'r', 'd', 'l']
        direction_idx_dict = {directions[i]: i for i in len(directions)}

        dir_idx = direction_idx_dict[direction]
        front = direction

        left_idx = dir_idx - 1
        if left_idx < 0:
            left_idx = len(directions) - 1
        left = directions[left_idx]

        right_idx = dir_idx + 1
        if right_idx >= len(directions):
            right_idx = 0
        right = directions[right_idx]

    def get_perceived_cell_mask(self, direction):
        """

        :param direction:
        :return:
        """

        directions = ['u', 'r', 'd', 'l']
        direction_idx_dict = {directions[i]: i for i in len(directions)}

        dir_idx = direction_idx_dict[direction]
        front = direction

        left_idx = dir_idx - 1
        if left_idx < 0:
            left_idx = len(directions) - 1
        left = directions[left_idx]

        right_idx = dir_idx + 1
        if right_idx >= len(directions):
            right_idx = 0
        right = directions[right_idx]

        mask = 0
        if self.sensors[0] == 0:
            mask = mask | dir_int_mask[left]

        if self.sensors[1] == 0:
            mask = mask | dir_int_mask[front]

        if self.sensors[2] == 0:
            mask = mask | dir_int_mask[right]

        return mask

    def __str__(self):
        return str(self.sensors)


class Goal:
    def __init__(self, maze_dim):
        self.maze_dim = maze_dim

    def hit_goal(self, location):
        '''
        Checks if the goal is hit by the agent placed in location.
        :param location:
        :return:
        '''
        goal_bounds = [self.maze_dim / 2 - 1, self.maze_dim / 2]

        return location[0] in goal_bounds and location[1] in goal_bounds


class Grid:
    def __init__(self, shape, init_val):
        self.shape = shape
        self.grid = [[copy.deepcopy(init_val) for c in range(shape[1])] for r in range(shape[0])]

    def __getitem___(self, row):
        return self.grid[row]

    def is_valid_location(self, location):
        return location[0] >= 0 and location[0] < self.shape[0] and location[1] >= 0 and location[1] < self.shape[1]

    def set_value(self, location, value):
        self.grid[location[0]][location[1]] = value

    def get_value(self, location):
        return self.grid[location[0]][location[1]]


class MazePerceived:
    """
    The representation of the maze perceived by the robot.
    """

    def __init__(self, shape):
        self.shape = shape
        self.explored_space = np.zeros(shape, int)
        self.explored = np.zeros(shape, int)
        self.explored_cnt = 0

    def update_cell(self, position, sensor, direction):
        """
        Update the cell with the walls and mark it as explored.
        :param position:
        :param sensor:
        :param direction:
        :return:
        """
        self.explored_space[position[0]][position[1]] |= sensor.get_perceived_cell_mask(sensor, direction)
        self.explored[position[0]][position[1]] = 1
        self.explored_cnt += 1

    def get_cell(self, position):
        """

        :param position:
        :return:
        """
        return self.explored_space[position[0]][position[1]]

    def check_cell_explored(self, position):
        """
        Checks if the current cell is explored (to be used with the search algorithms).
        :param position:
        :return:
        """
        return self.explored[position[0]][position[1]] == 1

    def is_explored(self):
        goal_bounds = [self.shape[0] / 2 - 1, self.shape[0] / 2]

        if self.explored_cnt < (self.shape[0] ** 2) / 2:
            return False

        return self.check_cell_explored([goal_bounds[0], goal_bounds[0]]) \
               or self.check_cell_explored([goal_bounds[0], goal_bounds[1]]) \
               or self.check_cell_explored([goal_bounds[1], goal_bounds[0]]) \
               or self.check_cell_explored([goal_bounds[1], goal_bounds[1]])
