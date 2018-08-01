import queue
from random import randint

from robot_utils import Goal, Grid, Direction, SensorInterpreter, MazePerceived, dir_sensors, rotation_idx_dict, \
    dir_move, dir_reverse, Graph, BFS


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
        self.training = True
        self.goal = Goal(maze_dim)
        self.exploration = RandomMoveWallsDetection(self, maze_dim)
        self.maze_graph = Graph(maze_dim*maze_dim)
        self.path = None
        self.path_idx = 0

    def get_initial_robot_pos(self):
        return {'location': [0, 0], 'heading': 'up'}

    def robot_position_2_graph_vertex(self):
        return int(int(self.robot_pos['location'][1]*self.maze_dim) + self.robot_pos['location'][0])

    def update_position(self, sensor, rotation, movement):
        steps_available = sensor[rotation_idx_dict[rotation]]

        # update heading based on the rotation values
        self.robot_pos['heading'] = dir_sensors[self.robot_pos['heading']][rotation_idx_dict[rotation]]

        # update location then
        movement = min(max(min(int(movement), 3), -3), steps_available)  # fix to range [-3, 3]

        u = self.robot_position_2_graph_vertex()

        while movement:
            if movement > 0:
                self.robot_pos['location'][0] += dir_move[self.robot_pos['heading']][0]
                self.robot_pos['location'][1] += dir_move[self.robot_pos['heading']][1]
                movement -= 1
            else:
                rev_heading = dir_reverse[self.robot_pos['heading']]
                self.robot_pos['location'][0] += dir_move[rev_heading][0]
                self.robot_pos['location'][1] += dir_move[rev_heading][1]
                movement += 1

            v = self.robot_position_2_graph_vertex()

            self.maze_graph.connect(u, v, 1, self.robot_pos['heading'], self.hit_goal())
            u = v

    def build_optimal_path(self):
        raise NotImplemented

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

            print ("perceived position is: {}".format(self.robot_pos))

            self.update_position(sensors, rotation, movement)

            if done:
                self.robot_pos = self.get_initial_robot_pos()
                self.training = False
                self.build_optimal_path()
                return 'Reset', 'Reset'
        else:
            rotation, movement = self.get_step(sensors)

        return rotation, movement

    def get_training_step(self, sensors):
        '''Gets the training step and updates the beleif space.'''
        rotation, movement = self.exploration.get_step(sensors)
        done = self.exploration.is_explored()
        return rotation, movement, done

    def get_step(self, sensors):
        '''Gets the optimal step.'''
        step = self.path[self.path_idx]
        self.path_idx += 1
        return step

    def hit_goal(self):
        '''Checks if the goal is hit by the agent.'''
        goal_bounds = [int(self.maze_dim / 2) - 1, int(self.maze_dim / 2)]

        return self.robot_pos['location'][0] in goal_bounds and self.robot_pos['location'][1] in goal_bounds


class RobotBFS(Robot):
    def __init__(self, maze_dim):
        Robot.__init__(self, maze_dim)

    def build_optimal_path(self):
        bfs = BFS(self.maze_graph)
        bfs.search()
        self.path = bfs.path
        print (self.path)


class Exploration:
    def __init__(self, robot, maze_dim):
        self.maze_dim = maze_dim
        self.robot = robot
        #self.grid = Grid(maze_dim, '') # maybe need to start using another ds for exploration
        self.explored_space = MazePerceived(maze_dim)

    def get_step(self, sensors):
        return 0,0

    def is_explored(self):
        return self.explored_space.is_explored()


class BlindRandomMove(Exploration):
    def __init__(self, robot, maze_dim):
        Exploration.__init__(self, robot, maze_dim)

    def get_step(self, sensors):
        rotations = [-90, 0, 90]
        steps = [i for i in range(1,2)]

        rotation_idx = randint(0, len(rotations)-1)
        rotation = rotations[rotation_idx]

        step_idx = randint(0, len(steps)-1)
        step = steps[step_idx]

        self.explored_space.update_cell(self.robot.robot_pos, sensors)

        return rotation, step


class RandomMoveWallsDetection(BlindRandomMove):
    def __init__(self, robot, maze_dim):
        BlindRandomMove.__init__(self, robot, maze_dim)

    def get_step(self, sensors):
        s = SensorInterpreter(sensors)

        if s.is_dead_end():
            return 90,0

        can_go = False

        while not can_go:
            rotation, movement = BlindRandomMove.get_step(self, sensors)

            if self.is_permissible(sensors, rotation, movement):
                can_go = True

        return rotation, movement

    def is_permissible(self, sensors, rotation, movement):
        if movement <= 0:
            return True

        s = SensorInterpreter(sensors)
        return  s.distance(rotation) >= movement