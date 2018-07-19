from enum import Enum

# Defines the infrastructure for robot moves.

# directions to move: pair of (row direction, column direction).
dir_move = [[0, 1], # up
            [1, 0], # right
            [0, -1], # down
            [-1, 0]] # left


class DirectionControl(Enum):
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

    def change(self, direction_control):
        return Direction((self.value + direction_control.value)%4)

    def dir_move(self):
        return dir_move[self.value]

    def change_direction(self, direction_control):
        diff = direction_control.value - self.value

        if diff == 3:
            diff = -1
        elif diff == -3:
            diff = 1

        return DirectionControl(diff)

    def __str__(self):
        return self.name