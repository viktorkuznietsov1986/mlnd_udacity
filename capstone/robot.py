import queue
from random import randint

from robot_utils import Goal, Grid, Direction, SensorInterpreter, MazePerceived, dir_sensors, rotation_idx_dict, \
    Steering, dir_move, dir_reverse

rotation_to_steering = {-90:Steering.L, 0:Steering.F, 90:Steering.R}


class Robot(object):
    def __init__(self, maze_dim):
        '''
        Use the initialization function to set up attributes that your robot
        will use to learn and navigate the maze. Some initial attributes are
        provided based on common information, including the size of the maze
        the robot is placed in.
        '''

        self.robot_pos = self.get_initial_robot_pos()
        self.maze_dim = maze_dim
        self.explored_space = MazePerceived(maze_dim)
        self.training = True
        self.direction = Direction.U
        self.goal = Goal(maze_dim)
        #self.grid = Grid(maze_dim, ' ')
        self.exploration = RandomMove(maze_dim)

    def get_initial_robot_pos(self):
        return {'location': [0, 0], 'heading': 'up'}

    def can_go(self, rotation, movement):
        return True # todo implement

    def update_location(self, rotation, movement):
        s = rotation_to_steering[rotation]

        movement = max(min(int(movement), 3), -3)  # fix to range [-3, 3]
        while movement:
            if movement > 0:
                if self.can_go(rotation, movement):
                    self.robot_pos['location'][0] += dir_move[self.robot_pos['heading']][0]
                    self.robot_pos['location'][1] += dir_move[self.robot_pos['heading']][1]
                    movement -= 1
                else:
                    print("Movement stopped by wall.")
                    movement = 0
            else:
                rev_heading = dir_reverse[self.robot_pos['heading']]
                if self.can_go(rotation, movement):
                    self.robot_pos['location'][0] += dir_move[rev_heading][0]
                    self.robot_pos['location'][1] += dir_move[rev_heading][1]
                    movement += 1
                else:
                    print("Movement stopped by wall.")
                    movement = 0

    def build_optimal_path(self):
        # todo
        frontier = []


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
        the maze) then returning the tuple ('Reset', 'Reset') will indicate to
        the tester to end the run and return the robot to the start.
        '''
        rotation = 0
        movement = 0

        # if explored
        if self.training:
            rotation, movement, done = self.get_training_step(sensors)

            # obtain direction from heading and rotation values
            direction = dir_sensors[self.robot_pos['heading']][rotation_idx_dict[rotation]]

            #self.explored_space.update_cell(self.robot_pos['location'], sensors, direction)

            #print ("perceived position is: {}".format(self.robot_pos))

            s = SensorInterpreter(sensors)

            if self.can_go(rotation, movement):
                self.update_location(rotation, movement)

            if done:
                self.robot_pos = self.get_initial_robot_pos()
                self.build_optimal_path()
                self.training = False
                return 'Reset', 'Reset'
        else:
            rotation, movement = self.get_step(sensors)

        return rotation, movement

    def get_training_step(self, sensors):
        '''Gets the training step and updates the beleif space.'''
        rotation, movement = self.exploration.get_step(sensors)
        done = self.explored_space.is_explored()
        return rotation, movement, done

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

        self.frontier = queue.Queue()


    def get_training_step(self, sensors):
        '''Gets the training step and updates the beleif space.'''


    def get_step(self, sensors):
        '''Gets the optimal step.'''
        return 0, 0


class Exploration:
    def __init__(self, maze_dim):
        self.maze_dim = maze_dim
        self.grid = Grid(maze_dim, '') # maybe need to start using another ds for exploration
        self.location = [0, 0]

    def get_step(self, sensors):
        return 0,0


class RandomMove(Exploration):
    def __init__(self, maze_dim):
        Exploration.__init__(self, maze_dim)

    def get_step(self, sensors):
        rotations = [-90, 0, 90]
        steps = [i for i in range(1, 3)]

        rotation_idx = randint(0, len(rotations)-1)
        rotation = rotations[rotation_idx]

        step_idx = randint(0, len(steps)-1)
        step = steps[step_idx]

        return rotation, step




