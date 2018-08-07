import heapq
from collections import deque
import numpy as np

# Defines the infrastructure for robot moves.
dir_sensors = {'u': ['l', 'u', 'r'], 'r': ['u', 'r', 'd'],
               'd': ['r', 'd', 'l'], 'l': ['d', 'l', 'u'],
               'up': ['l', 'u', 'r'], 'right': ['u', 'r', 'd'],
               'down': ['r', 'd', 'l'], 'left': ['d', 'l', 'u']}
dir_move = {'u': [0, 1], 'r': [1, 0], 'd': [0, -1], 'l': [-1, 0],
            'up': [0, 1], 'right': [1, 0], 'down': [0, -1], 'left': [-1, 0]}
dir_reverse = {'u': 'd', 'r': 'l', 'd': 'u', 'l': 'r',
               'up': 'd', 'right': 'l', 'down': 'u', 'left': 'r'}

rotation_idx_dict = {-90: 0, 0: 1, 90: 2}

# integer values for directions
dir_int_mask = {'u': 1, 'r': 2, 'd': 4, 'l': 8,
                'up': 1, 'right': 2, 'down': 4, 'left': 8}


def try_explore_random(prob):
    """
    Returns the choice whether to explore or not.
    :param prob: probability of letting explore.
    :return: the choice whether to explore or not.
    """
    return np.random.choice([False, True], p=[1.0 - prob, prob])


class Direction:
    def __init__(self, direction):
        self.direction = direction

    def reverse(self):
        return Direction(dir_reverse[self.direction])

    def change(self, rotation):
        rotation_idx = rotation_idx_dict[rotation]
        return Direction(dir_sensors[self.direction][rotation_idx])

    def dir_move(self):
        return dir_move[self.direction]

    def __str__(self):
        return self.direction


class Heading:
    def __init__(self, direction, pos):
        self.direction = direction
        self.pos = pos

    def __str__(self):
        return '{} at ({},{})'.format(self.direction.name, self.pos[0], self.pos[1])

    def change(self, rotation, movement):
        direction = self.direction.change(rotation)
        direction_move = direction.dir_move()
        pos = [self.pos[i] * direction_move[i] * movement for i in range(2)]
        return Heading(direction, pos)

    def move_forward(self, steps=1):
        return self.change(0, steps)

    def move_left(self, steps=1):
        return self.change(-90, steps)

    def move_right(self, steps=1):
        return self.change(90, steps)

    def move_backward(self, steps=1):
        return self.reverse().move_forward(steps)

    def reverse(self):
        return Heading(self.direction.reverse(), self.pos)


class SensorInterpreter:
    def __init__(self, sensors):
        self.sensors = sensors

    def distance(self, rotation):
        return self.sensors[rotation_idx_dict[rotation]]

    def is_dead_end(self):
        return max(self.sensors) == 0

    def is_one_way(self):
        return self.sensors[0] == 0 and self.sensors[1] > 0 and self.sensors[2] == 0

    def can_go(self, rotation, movement):
        # apply steering and check number of steps
        return self.distance(rotation) >= movement

    def get_perceived_cell_mask(self, direction):
        """

        :param direction:
        :return:
        """

        directions = ['u', 'r', 'd', 'l']
        direction_idx_dict = {directions[i]: i for i in range(len(directions))}

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
        self.goal_bounds = [int(self.maze_dim / 2) - 1, int(self.maze_dim / 2)]

    def hit_goal(self, location):
        '''
        Checks if the goal is hit by the agent placed in location.
        :param location:
        :return:
        '''
        return location[0] in self.goal_bounds and location[1] in self.goal_bounds

    def get_distance(self, x):
        # returns the distance to the minimum distance to the goal
        goal_pos = int(int(self.goal_bounds[1] * self.maze_dim) + self.goal_bounds[0])
        return abs(x-goal_pos)

    def get_squared_distance(self, x):
        goal_pos = self.goal_bounds[1] * self.maze_dim + self.goal_bounds[0]
        return int(abs((x-goal_pos)**2))


class Grid:
    def __init__(self, shape, init_val):
        self.shape = shape

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

    def __init__(self, dim):
        self.shape = (dim, dim)
        self.explored_space = np.zeros(self.shape, int)
        self.explored = np.zeros(self.shape, int)
        self.explored_cnt = 0

    def update_cell(self, robot_pos, sensor):

        mask = 0
        sensor_heading = dir_sensors[robot_pos['heading']]

        for i in range(len(sensor)):
            if sensor[i] == 0:
                mask |= dir_int_mask[sensor_heading[i]]

        self.explored_space[robot_pos['location'][0]][robot_pos['location'][1]] |= mask
        self.explored[robot_pos['location'][0]][robot_pos['location'][1]] = 1
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
        goal_bounds = [int(self.shape[0] / 2) - 1, int(self.shape[0] / 2)]

        if self.explored_cnt < (self.shape[0] ** 2) / 2:
            return False

        return self.check_cell_explored([goal_bounds[0], goal_bounds[0]]) \
               or self.check_cell_explored([goal_bounds[0], goal_bounds[1]]) \
               or self.check_cell_explored([goal_bounds[1], goal_bounds[0]]) \
               or self.check_cell_explored([goal_bounds[1], goal_bounds[1]])

    def is_permissible(self, cell, direction):
        """
        Returns a boolean designating whether or not a cell is passable in the
        given direction. Cell is input as a list. Directions may be
        input as single letter 'u', 'r', 'd', 'l', or complete words 'up',
        'right', 'down', 'left'.
        """
        try:
            return (self.explored_space[tuple(cell)] & dir_int_mask[direction] != 0)
        except:
            print ('Invalid direction provided!')


class Edge:
    def __init__(self, direction, u, v, w, contains_goal):
        self.direction = direction
        self.w = w
        self.u = u
        self.v = v
        self.contains_goal = contains_goal

    def either(self):
        return self.u

    def other(self, u):
        if u == self.u:
            return self.v

        return self.u

    def weight(self):
        return self.w

    def direction(self):
        return self.direction

    def contains_goal(self):
        return self.contains_goal()


class Graph:
    def __init__(self, dim):
        self.dim = dim
        self.edges = np.empty(dim, dtype=list)
        self.edges_count = 0

    def is_connected(self, u, v):
        if self.edges[u] is None or self.edges[v] is None:
            return False

        for e in self.edges[u]:
            if e.either() == v or e.other(u) == v:
                return True

        return False

    def connect(self, u, v, w, direction, contains_goal):
        if not self.is_connected(u,w):
            edge_v = Edge(direction, u, v, w, contains_goal)
            edge_u = Edge(dir_reverse[direction], v, u, w, contains_goal)

            if self.edges[v] is None:
                self.edges[v] = []

            if self.edges[u] is None:
                self.edges[u] = []

            self.edges[v].append(edge_v)
            self.edges[u].append(edge_u)

            self.edges_count += 2


class GraphSearch:
    class Node:
        def __init__(self, parent, edge, cost=1, goal=None):
            self.parent = parent
            self.edge = edge
            self.cost = cost
            self.goal = goal

        def get_cost(self):
            u = self.edge.either()
            v = self.edge.other(u)

            if self.goal is not None:
                return self.cost + min(self.goal.get_squared_distance(u), self.goal.get_squared_distance(v))

            return self.cost

        def __lt__(self, other):
            return self.get_cost() < other.get_cost()

    def __init__(self, graph, starting_point=0):
        self.graph = graph
        self.starting_point = starting_point
        self.visited = [False for i in range(graph.dim)]
        self.path = []

    def search(self):
        node = self.build_node()
        self.path = self.convert_node_to_path(node)

        # print the debug info
        if len(self.path) > 0:
            print("the cost of the path is: {}".format(node.cost))
            print("the number of moves is: {}".format(len(self.path)))

    def build_node(self):
        raise NotImplemented

    @staticmethod
    def convert_directions_to_rotation(previous_direction, current_direction):
        if previous_direction == current_direction:
            return 0

        directions = dir_sensors[previous_direction]

        if current_direction == directions[0]:
            return 90
        elif current_direction == directions[2]:
            return -90

        return 180

    def convert_node_to_path(self, node):
        path = []

        if node is not None:
            previous_direction = node.edge.direction

            while node is not None:
                current_direction = node.edge.direction

                if len(path) == 0:
                    object_to_add = [0, 1]
                    path.insert(0, object_to_add)
                else:
                    if current_direction == previous_direction:
                        if path[0][1] < 3:
                            path[0][1] += 1
                        else:
                            rotation = path[0][0]
                            movement = 1
                            path[0][0] = 0
                            object_to_add = [rotation, movement]
                            path.insert(0, object_to_add)
                    else:
                        rotation = self.convert_directions_to_rotation(previous_direction, current_direction)
                        path[0][0] = rotation
                        rotation = 0
                        movement = 1
                        obj_to_add = [rotation, movement]
                        path.insert(0, obj_to_add)

                previous_direction = current_direction
                node = node.parent

        return path


class BFS(GraphSearch):
    def __init__(self, graph, starting_point=0):
        GraphSearch.__init__(self, graph, starting_point)

    def build_node(self):
        frontier = deque()

        for e in self.graph.edges[self.starting_point]:
            frontier.append(GraphSearch.Node(None, e))

        self.visited[self.starting_point] = True

        while len(frontier) > 0:
            n = frontier.popleft()
            e = n.edge

            u = e.either()
            v = e.other(u)

            if e.contains_goal:
                return n

            if self.visited[u] and self.visited[v]:
                continue

            vertex = u
            if self.visited[u]:
                vertex = v

            self.visited[vertex] = True

            for e in self.graph.edges[vertex]:
                if not self.visited[e.other(vertex)]:
                    frontier.append(GraphSearch.Node(n, e))

        return None


class AStar(GraphSearch):
    def __init__(self, graph, starting_point=0):
        GraphSearch.__init__(self, graph, starting_point)

    def build_node(self):
        frontier = []

        goal = Goal(int(self.graph.dim/2))

        for e in self.graph.edges[self.starting_point]:
            item_to_add = GraphSearch.Node(None, e, 1, goal)
            heapq.heappush(frontier, item_to_add)

        self.visited[self.starting_point] = True

        while len(frontier) > 0:
            n = heapq.heappop(frontier)
            e = n.edge

            u = e.either()
            v = e.other(u)

            if e.contains_goal:
                return n

            if self.visited[u] and self.visited[v]:
                continue

            vertex = u
            if self.visited[u]:
                vertex = v

            self.visited[vertex] = True

            for e in self.graph.edges[vertex]:
                if not self.visited[e.other(vertex)]:
                    item_to_add = GraphSearch.Node(n, e, n.cost+1, goal)
                    heapq.heappush(frontier, item_to_add)

        return None


class Dijkstra(GraphSearch):
    """
    Todo implement
    """
    def __init__(self, graph, starting_point=0):
        GraphSearch.__init__(self, graph, starting_point)
        self.find_all_paths()

    def find_all_paths(self):
        raise NotImplemented

    def build_node(self):
        raise NotImplemented










