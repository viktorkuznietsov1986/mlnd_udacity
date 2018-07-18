import queue

import numpy as np

orientations = {'u': 1, 'r': 2, 'd': 4, 'l': 8,
                   'up': 1, 'right': 2, 'down': 4, 'left': 8}

class Robot(object):
    def __init__(self, maze_dim):
        '''
        Use the initialization function to set up attributes that your robot
        will use to learn and navigate the maze. Some initial attributes are
        provided based on common information, including the size of the maze
        the robot is placed in.
        '''

        self.location = [0, 0]
        self.heading = 'up'
        self.maze_dim = maze_dim
        self.explored_space = np.zeros([maze_dim, maze_dim], int)
        self.explored = False
        self.training = True
        self.orientation = 'u'

    def next_move(self, sensors):
        '''
        Use this function to determine the next move the robot should make,
        based on the input from the sensors after its previous move. Sensor
        inputs are a list of three distances from the robot's left, front, and
        right-facing sensors, in that order.

        Outputs should be a tuple of two values. The first value indicates
        robot rotation (if any), as a number: 0 for no rotation, +90 for a
        90-degree rotation clockwise, and -90 for a 90-degree rotation
        counterclockwise. Other values will result in no rotation. The second
        value indicates robot movement, and the robot will attempt to move the
        number of indicated squares: a positive number indicates forwards
        movement, while a negative number indicates backwards movement. The
        robot may move a maximum of three units per turn. Any excess movement
        is ignored.

        If the robot wants to end a run (e.g. during the first training run in
        the maze) then returing the tuple ('Reset', 'Reset') will indicate to
        the tester to end the run and return the robot to the start.
        '''
        rotation = 0
        movement = 0

        # if explored
        if self.training:
            rotation, movement, done = self.get_training_step(sensors)

            if done:
                self.explored = True
                return 'Reset', 'Reset'
        else:
            rotation, movement = self.get_step(sensors)

        return rotation, movement

    def get_training_step(self, sensors):
        '''Gets the training step and updates the beleif space.'''
        return 0,0,True

    def get_step(self, sensors):
        '''Gets the optimal step.'''
        return 0,0

    def hit_goal(self):
        '''Checks if the goal is hit by the agent.'''
        goal_bounds = [self.maze_dim / 2 - 1, self.maze_dim / 2]

        return self.location[0] in goal_bounds and self.location[1] in goal_bounds


class RobotBFS(Robot):
    def __init__(self, maze_dim):
        Robot.__init__(self, maze_dim)
        self.rotations = np.array([-90, 0, 90])
        self.movements = np.array([i for i in range(-3,4)])
        self.frontier = queue.Queue()
        self.move_controller = MoveController()

    def get_training_step(self, sensors):
        '''Gets the training step and updates the beleif space.'''
        actions = self.move_controller.get_actions(sensors)
        for a in actions:
            self.frontier.put(a)

        rotation, movement = self.frontier.get_nowait()


        return rotation, movement, self.hit_goal()

    def get_step(self, sensors):
        '''Gets the optimal step.'''
        return 0, 0

class MoveController():
    def get_actions(self, sensors):
        actions = [(0, 1), (0, -1), (90, 1), (-90, 1), (90, 0), (-90, 0)]

        return actions




