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


class SensorInterpreter:
    """
    Wrapper for the sensory information.
    """
    def __init__(self, sensors):
        """
        Ctor.
        :param sensors: sensory information in the given direction.
        """
        self.sensors = sensors

    def distance(self, rotation):
        """
        Distance to the walls in the given direction (based on the rotation).
        :param rotation: Rotation value (-90, 0, 90).
        :return: Distance to the walls in the given direction (based on the rotation).
        """
        return self.sensors[rotation_idx_dict[rotation]]

    def is_dead_end(self):
        """
        Checks if the agent reached the dead end.
        :return: True if the agent reached the dead end, False otherwise.
        """
        return max(self.sensors) == 0

    def is_one_way(self):
        """
        Checks if the agent can go one way.
        :return: True if the agent can go one way, False otherwise.
        """
        return self.sensors[0] == 0 and self.sensors[1] > 0 and self.sensors[2] == 0

    def can_go(self, rotation, movement):
        """
        Apply rotation and see if the agent can go in the given direction.
        :param rotation: Rotation value (-90, 0, 90).
        :param movement: Number of steps to go (the range is: [0, 3]).
        :return: True if the agent can move in the specified direction, False otherwise.
        """
        return self.distance(rotation) >= movement

    def get_perceived_cell_mask(self, direction):
        """
        Build the mask to apply to a cell, based on the information perceived.
        :param direction: heading direction.
        :return: the mask to apply to a cell, based on the information perceived.
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
        """
        The string representation of the sensors (for debugging).
        :return: the string representation of the sensors.
        """
        return str(self.sensors)


class Goal:
    """
    Incapsulates the goal.
    """
    def __init__(self, maze_dim):
        """
        Ctor.
        :param maze_dim: the dimension of the maze.
        """
        self.maze_dim = maze_dim
        self.goal_bounds = [int(self.maze_dim / 2) - 1, int(self.maze_dim / 2)]

    def hit_goal(self, location):
        """
        Checks if the goal is hit by the agent placed in location.
        :param location: location on the grid.
        :return: True if the goal is hit at the given location, False otherwise.
        """
        return location[0] in self.goal_bounds and location[1] in self.goal_bounds


class MazePerceived:
    """
    The representation of the maze perceived by the robot.
    """
    def __init__(self, dim):
        """
        Ctor.
        :param dim: the dimension of the maze.
        """
        self.shape = (dim, dim)
        self.explored_space = np.zeros(self.shape, int)
        self.explored = np.zeros(self.shape, int)
        self.explored_cnt = 0

    def update_cell(self, robot_pos, sensor):
        """
        Updates the cell with the sensory information.
        :param robot_pos: robot position and heading.
        :param sensor: sensory information (distance to the walls).
        :return: no return value.
        """
        if self.explored[robot_pos['location'][0]][robot_pos['location'][1]] != 1:
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
        Gets the value of the cell.
        :param position: Position on the grid.
        :return: the value of the cell.
        """
        return self.explored_space[position[0]][position[1]]

    def check_cell_explored(self, position):
        """
        Checks if the current cell is explored (to be used with the search algorithms).
        :param position: Position on the grid.
        :return: True if the current cell is explored, False otherwise.
        """
        return self.explored[position[0]][position[1]] == 1

    def is_explored(self):
        """
        Checks if the maze can be marked as explored.
        :return: True if the maze is explored, False otherwise.
        """
        goal_bounds = [int(self.shape[0] / 2) - 1, int(self.shape[0] / 2)]

        if self.explored_cnt < 4*(self.shape[0] ** 2)/5:
            return False

        return self.check_cell_explored([goal_bounds[0], goal_bounds[0]]) \
               or self.check_cell_explored([goal_bounds[0], goal_bounds[1]]) \
               or self.check_cell_explored([goal_bounds[1], goal_bounds[0]]) \
               or self.check_cell_explored([goal_bounds[1], goal_bounds[1]])

    def is_permissible(self, cell, direction):
        """
        Returns a boolean designating whether or not a cell is passable in the
        given direction.
        :param cell: Cell is input as a list.
        :param direction: Directions may be
        input as single letter 'u', 'r', 'd', 'l', or complete words 'up',
        'right', 'down', 'left'.
        :return: True if a cell is passable in the given direction, False otherwise.
        """
        try:
            return (self.explored_space[tuple(cell)] & dir_int_mask[direction] != 0)
        except:
            print ('Invalid direction provided!')


class Edge:
    """
    Represents an edge for the graph data structure, simply the connection between 2 vertices.
    """
    def __init__(self, direction, u, v, w, contains_goal):
        """
        Ctor.
        :param direction: heading direction.
        :param u: start vertex.
        :param v: end vertex.
        :param w: weight.
        :param contains_goal: is there a goal on any side.
        """
        self.direction = direction
        self.w = w
        self.u = u
        self.v = v
        self.contains_goal = contains_goal

    def either(self):
        """
        Get a single vertex on the edge.
        :return: a single vertex on the edge.
        """
        return self.u

    def other(self, u):
        """
        Gets the vertex other than an input.
        :param u: the vertex on the edge.
        :return: the vertex other than an input.
        """
        if u == self.u:
            return self.v

        return self.u

    def weight(self):
        """
        Gets the weight of the edge.
        :return: the weight of the edge.
        """
        return self.w

    def direction(self):
        """
        Get the heading direction of the edge.
        :return: the heading direction of the edge.
        """
        return self.direction


class Graph:
    """
    Represents the directed graph data structure.
    """
    def __init__(self, V):
        """
        Ctor.
        :param V: number of vertices on the graph.
        """
        self.V = V
        self.edges = np.empty(V, dtype=list)
        self.edges_count = 0

    def is_connected(self, u, v):
        """
        Checks if 2 vertices are connected in any direction.
        :param u: the first vertex.
        :param v: the second vertex.
        :return: True if 2 vertices are connected, False otherwise.
        """
        if self.edges[u] is None or self.edges[v] is None:
            return False

        for e in self.edges[u]:
            if e.either() == v or e.other(u) == v:
                return True

        return False

    def connect(self, u, v, w, direction, contains_goal):
        """
        Connect 2 vertices if they are not connected.
        :param u: first vertex.
        :param v: second vertex.
        :param w: the weight of the edge.
        :param direction: the heading direction.
        :param contains_goal: whether or not the edge contains a goal.
        """
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
    """
    A base class for the search algorithms on the graph.
    """
    class Node:
        """
        The data structure for bookkeeping during the path search.
        """
        def __init__(self, parent, edge, cost=1, goal=None):
            """
            Ctor.
            :param parent: the parent Node.
            :param edge: edge on the Node.
            :param cost: cost to choose the Node.
            :param goal: the goal.
            """
            self.parent = parent
            self.edge = edge
            self.cost = cost
            self.goal = goal

        def get_cost(self):
            """
            Get the cost, based on the initial cost + the heuristics (squared distance to the goal).
            :return: the cost.
            """
            # we'll use the following heuristics: do penalize the agent for a turn.
            if self.parent is not None and (self.parent.edge.direction != self.edge.direction):
                return self.cost + 1

            return self.cost

        def __lt__(self, other):
            """
            The comparison operator (<).
            :param other: other node.
            :return: True if current node's cost is lower than the other node's cost.
            """
            return self.get_cost() < other.get_cost()

    def __init__(self, graph, starting_point=0):
        """
        Ctor.
        :param graph: Graph ready to be explored.
        :param starting_point: The starting point on the graph.
        """
        self.graph = graph
        self.starting_point = starting_point
        self.visited = [False for i in range(graph.V)]
        self.path = []
        self.nodes_explored_cnt = 0

    def search(self):
        """
        Search on the graph.
        :return:
        """
        node = self.build_node()
        self.path = self.convert_node_to_path(node)

        # print the debug info
        if len(self.path) > 0:
            print("the cost of the path is: {}".format(node.cost))
            print("the number of moves is: {}".format(len(self.path)))

    def build_node(self):
        """
        Builds the goal node.
        :return: the goal node if the goal is found, Nothing otherwise.
        """
        raise NotImplemented

    @staticmethod
    def convert_directions_to_rotation(previous_direction, current_direction):
        """
        Builds the rotation value based on the 2 directions.
        :param previous_direction: previous direction.
        :param current_direction: current direction.
        :return: the rotation value in degrees: [-90, 0, 90].
        """
        if previous_direction == current_direction:
            return 0

        directions = dir_sensors[previous_direction]

        if current_direction == directions[0]:
            return 90
        elif current_direction == directions[2]:
            return -90

        return 180

    def convert_node_to_path(self, node):
        """
        Converts the goal node to the path ((rotation, movement) sequence used by the agent).
        :param node: the goal node.
        :return: the list of the (rotation, movement) tuples to be used by the agent.
        """
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
    """
    The breadth first search algorithm implementation.
    It's used as the benchmark model.
    """
    def __init__(self, graph, starting_point=0):
        GraphSearch.__init__(self, graph, starting_point)

    def build_node(self):
        frontier = deque()

        for e in self.graph.edges[self.starting_point]:
            frontier.append(GraphSearch.Node(None, e, 1))

        self.visited[self.starting_point] = True

        goal_node = None

        while len(frontier) > 0:
            n = frontier.popleft()
            e = n.edge

            u = e.either()
            v = e.other(u)

            if self.visited[u] and self.visited[v]:
                continue

            self.nodes_explored_cnt += 1

            if e.contains_goal:
                if goal_node is None:
                    goal_node = n
                else:
                    if goal_node.cost > n.cost:
                        goal_node = n

            vertex = u
            if self.visited[u]:
                vertex = v

            self.visited[vertex] = True

            for e in self.graph.edges[vertex]:
                if not self.visited[e.other(vertex)]:
                    frontier.append(GraphSearch.Node(n, e, n.cost+1))

        return goal_node


class AStar(GraphSearch):
    """
    Implements the A* algorithm.
    """
    def __init__(self, graph, starting_point=0):
        GraphSearch.__init__(self, graph, starting_point)

    def build_node(self):
        frontier = []

        goal = Goal(int(self.graph.V/2))

        for e in self.graph.edges[self.starting_point]:
            item_to_add = GraphSearch.Node(None, e, 1, goal)
            heapq.heappush(frontier, item_to_add)

        self.visited[self.starting_point] = True

        while len(frontier) > 0:
            n = heapq.heappop(frontier)

            e = n.edge

            u = e.either()
            v = e.other(u)

            if self.visited[u] and self.visited[v]:
                continue

            self.nodes_explored_cnt += 1

            if e.contains_goal:
                return n

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
    Implements the Dijkstra algorithm.
    Note: shouldn't be implemented as the part of the project as it does the redundant work of finding and storing all
    the unnecessary paths.
    """
    def __init__(self, graph, starting_point=0):
        GraphSearch.__init__(self, graph, starting_point)
        self.find_all_paths()

    def find_all_paths(self):
        raise NotImplemented

    def build_node(self):
        raise NotImplemented


class GridVisualization:
    """
    The class is used to visualize the path on the maze.
    The grid is shown in 0s with 1s as the robot movements.
    """
    def __init__(self, maze_dim, path):
        self.shape = (maze_dim, maze_dim)
        self.grid = np.chararray(self.shape, unicode=True)
        self.grid[:] = '.'
        self.update_grid_with_path(path)

    def update_grid_with_path(self, path):
        robot_pos = {'location': [self.shape[0]-1, 0], 'heading': 'u'}
        self.grid[robot_pos['location'][0]][robot_pos['location'][1]] = robot_pos['heading']

        for rotation, movement in path:
            # update heading based on the rotation values
            robot_pos['heading'] = dir_sensors[robot_pos['heading']][rotation_idx_dict[rotation]]

            while movement > 0:
                robot_pos['location'][0] -= dir_move[robot_pos['heading']][1]
                robot_pos['location'][1] += dir_move[robot_pos['heading']][0]
                movement -= 1
                self.grid[robot_pos['location'][0]][robot_pos['location'][1]] = robot_pos['heading']

    def show_grid(self):
        #print(self.grid)
        for row in self.grid:
            str_value = '| '
            for column in row:
                str_value += str(column) + ' '

            str_value += '|'
            print(str_value)










