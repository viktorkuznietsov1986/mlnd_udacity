from random import randint
import numpy as np

from robot_utils import Goal, SensorInterpreter, MazePerceived, dir_sensors, rotation_idx_dict, \
    dir_move, dir_reverse, Graph, BFS, AStar, try_explore_random


class Robot(object):
    def __init__(self, maze_dim):
        """
        Use the initialization function to set up attributes that your robot
        will use to learn and navigate the maze. Some initial attributes are
        provided based on common information, including the size of the maze
        the robot is placed in.
        """
        self.robot_pos = self.get_initial_robot_pos()
        self.maze_dim = maze_dim
        self.training = True
        self.goal = Goal(maze_dim)
        self.exploration = RandomMoveVisitCounter(self, maze_dim)
        self.maze_graph = Graph(maze_dim*maze_dim)
        self.path = None
        self.path_idx = 0

    @staticmethod
    def get_initial_robot_pos():
        """
        Gets the initial robot position: location: [0, 0] and heading: up.
        :return: the initial robot position: location: [0, 0] and heading: up.
        """
        return {'location': [0, 0], 'heading': 'up'}

    def robot_position_2_graph_vertex(self):
        """
        Converts the robot's location to a vertex on the graph.
        :return: a vertex on the graph corresponding to robot's location.
        """
        return int(int(self.robot_pos['location'][1]*self.maze_dim) + self.robot_pos['location'][0])

    def update_position(self, sensor, rotation, movement):
        """
        Update the robot's beleif state.
        :param sensor: sensory information.
        :param rotation: the rotation value.
        :param movement: number of steps to move.
        """
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
        """
        Builds the optimal path. Should be implemented in the child classes.
        """
        raise NotImplemented

    def next_move(self, sensors):
        """
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
        """
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
        """
        Gets the training step and updates the beleif space.
        :param sensors: sensors information.
        :return: values for rotation, movement and whether we're done with the exploration.
        """
        rotation, movement = self.exploration.get_step(sensors)
        done = self.exploration.is_explored()
        return rotation, movement, done

    def get_step(self, sensors):
        """
        Gets the optimal step based on path found or sensory information.
        :param sensors: sensory information.
        :return: optimal (rotation, movement) based on the path found.
        """
        step = self.path[self.path_idx]
        self.path_idx += 1
        return step

    def hit_goal(self):
        """
        Checks if the goal is hit by the agent.
        :return: True if the goal is hit by the agent, False otherwise.
        """
        goal_bounds = [int(self.maze_dim / 2) - 1, int(self.maze_dim / 2)]

        return self.robot_pos['location'][0] in goal_bounds and self.robot_pos['location'][1] in goal_bounds


class RobotBFS(Robot):
    """
    The BFS implementation for the robot.
    """
    def __init__(self, maze_dim):
        Robot.__init__(self, maze_dim)

    def build_optimal_path(self):
        bfs = BFS(self.maze_graph)
        bfs.search()
        self.path = bfs.path
        print (self.path)


class RobotAStar(Robot):
    """
    The A* implementation for the robot.
    """
    def __init__(self, maze_dim):
        Robot.__init__(self, maze_dim)

    def build_optimal_path(self):
        astar = AStar(self.maze_graph)
        astar.search()
        self.path = astar.path
        print (self.path)


class RobotFactory:
    """
    The factory class to create a different kinds of robots.
    """
    def __init__(self, maze_dim):
        self.maze_dim = maze_dim

    def get_robot(self, robot_type):
        """
        Retrieve the robot of the specified type.
        :param robot_type: type of the robot to create. here's the range: ('bfs', 'astar')
        :return: the robot of the specified type or None if no robot created.
        """
        if robot_type == 'bfs':
            return RobotBFS(self.maze_dim)
        elif robot_type == 'astar':
            return RobotAStar(self.maze_dim)
        else:
            return None


class Exploration:
    """
    Base class for exploration.
    """
    def __init__(self, robot, maze_dim):
        """
        Ctor.
        :param robot: an intelligent agent.
        :param maze_dim: the dimensions of the maze.
        """
        self.maze_dim = maze_dim
        self.robot = robot
        self.explored_space = MazePerceived(maze_dim)

    def get_step(self, sensors):
        """
        Should be implemented in the nested classes.
        :param sensors: sensory information perceived by the robot.
        :return: a step in format (rotation, movement).
        """
        raise NotImplemented

    def is_explored(self):
        """
        Check if the exploration is done.
        :return: True if the exploration is done, False otherwise.
        """
        return self.explored_space.is_explored()


class BlindRandomMove(Exploration):
    """
    The simplest class for exploration. Does a random move without looking at the sensory information.
    """
    def __init__(self, robot, maze_dim):
        Exploration.__init__(self, robot, maze_dim)

    def get_step(self, sensors):
        rotations = [-90, 0, 90]
        steps = [i+1 for i in range(1)]

        rotation_idx = randint(0, len(rotations)-1)
        rotation = rotations[rotation_idx]

        step_idx = randint(0, len(steps)-1)
        step = steps[step_idx]

        self.explored_space.update_cell(self.robot.robot_pos, sensors)

        return rotation, step

    def is_permissible(self, sensors, rotation, movement):
        """
        Checks if the agent can accomplish the chosen move.
        :param sensors: sensory information.
        :param rotation: the value for rotation.
        :param movement: the value for movement.
        :return: True if the agent can go, False otherwise.
        """
        if movement <= 0:
            return True

        s = SensorInterpreter(sensors)
        return s.distance(rotation) >= movement


class RandomMoveWallsDetection(BlindRandomMove):
    def __init__(self, robot, maze_dim):
        BlindRandomMove.__init__(self, robot, maze_dim)
        self.dead_ends = np.zeros([maze_dim, maze_dim], int)

    def get_step(self, sensors):
        s = SensorInterpreter(sensors)

        if s.is_dead_end():
            self.dead_ends[self.robot.robot_pos['location'][0]][self.robot.robot_pos['location'][1]] = 1
            return 90, 0

        can_go = False

        while not can_go:
            rotation, movement = BlindRandomMove.get_step(self, sensors)

            if self.is_permissible(sensors, rotation, movement):
                can_go = True

        return rotation, movement


class RandomMoveVisitCounter(RandomMoveWallsDetection):
    """
    Counts the number of visits to any specified cell in order to visit the less visited first.
    """
    def __init__(self, robot, maze_dim):
        BlindRandomMove.__init__(self, robot, maze_dim)
        self.visits = np.zeros([maze_dim, maze_dim], int)

    def get_step(self, sensors):
        s = SensorInterpreter(sensors)

        self.visits[self.robot.robot_pos['location'][0]][self.robot.robot_pos['location'][1]] += 1

        if s.is_dead_end():
            return 90, 0

        can_go = False

        explore_anyway_prob = .2

        rotations = [-90, 0, 90]
        visited_cnt = []

        for r in rotations:
            robot_pos = {'location': [self.robot.robot_pos['location'][0], self.robot.robot_pos['location'][1]],
                         'heading': dir_sensors[self.robot.robot_pos['heading']][rotation_idx_dict[r]]}
            robot_pos['location'][0] += dir_move[robot_pos['heading']][0]
            robot_pos['location'][1] += dir_move[robot_pos['heading']][1]
            if 0 <= robot_pos['location'][0] < self.maze_dim and \
                    0 <= robot_pos['location'][1] < self.maze_dim:
                visited_cnt.append(int(self.visits[robot_pos['location'][0]][robot_pos['location'][1]]))
            else:
                visited_cnt.append(np.inf)

        visited_dict = {visited: i for i, visited in enumerate(visited_cnt)}

        steps = [i+1 for i in range(1)]

        while not can_go:

            explore_at_random = try_explore_random(explore_anyway_prob)

            visited = np.inf

            if explore_at_random:
                rotation, movement = BlindRandomMove.get_step(self, sensors)
            else:
                visited = np.min(visited_cnt)

                if visited == np.inf:
                    continue

                visited = int(visited)

                rotation = rotations[visited_dict[visited]]

                step_idx = randint(0, len(steps) - 1)
                movement = steps[step_idx]

            if self.is_permissible(sensors, rotation, movement):
                can_go = True
            else:
                if visited != np.inf:
                    visited_cnt[visited_dict[visited]] = np.inf

        return rotation, movement


class RandomMoveDeadEndMemorization(RandomMoveWallsDetection):
    def __init__(self, robot, maze_dim):
        RandomMoveWallsDetection.__init__(self, robot, maze_dim)

    def get_step(self, sensors):
        s = SensorInterpreter(sensors)
        if s.is_dead_end():
            return 90, 0

        can_go = False

        while not can_go:
            rotation, movement = BlindRandomMove.get_step(self, sensors)

            permissible, dead_end = self.is_permissible(sensors, rotation, movement)

            if permissible:
                break

            if dead_end:
                return 90, 0

        return rotation, movement

    def is_permissible(self, sensors, rotation, movement):
        if not BlindRandomMove.is_permissible(self, sensors, rotation, movement):
            return False, False

        robot_pos = {'location': [self.robot.robot_pos['location'][0], self.robot.robot_pos['location'][1]],
                     'heading':  dir_sensors[self.robot.robot_pos['heading']][rotation_idx_dict[rotation]]}

        while movement:
            if movement > 0:
                robot_pos['location'][0] += dir_move[robot_pos['heading']][0]
                robot_pos['location'][1] += dir_move[robot_pos['heading']][1]
                movement -= 1
            else:
                rev_heading = dir_reverse[robot_pos['heading']]
                robot_pos['location'][0] += dir_move[rev_heading][0]
                robot_pos['location'][1] += dir_move[rev_heading][1]
                movement += 1

            if self.dead_ends[robot_pos['location'][0]][robot_pos['location'][1]] == 1:
                return False, True

        if self.dead_ends[robot_pos['location'][0]][robot_pos['location'][1]] == 1:
            return False, True

        return True, False
