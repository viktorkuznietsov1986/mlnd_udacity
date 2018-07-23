import copy
from enum import Enum

# Defines the infrastructure for robot moves.

# directions to move: pair of (row direction, column direction).
dir_move = [[0, 1], # up
            [1, 0], # right
            [0, -1], # down
            [-1, 0]] # left


class Steering(Enum):
    '''
    Enum class for changing direction.
    '''
    L, F, R = (-1, 0, 1) # Go left, front, right.

    def __str__(self):
        return self.name


class Direction(Enum):
    '''
    Enum class encapsulating the heading direction.
    '''
    U, R, D, L = range(4) # up, right, down, left.

    def reverse(self):
        return Direction((self.value+2)%4)

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
        pos = [self.pos[i]*direction_move[i]*steps for i in range(2)]
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


class Sensor:
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

    def __str__(self):
        return str(self.sensors)


class Goal:
    def __init__(self, maze_dim):
        self.maze_dim = maze_dim

    def hit_goal(self, location):
        '''Checks if the goal is hit by the agent placed in location.'''
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