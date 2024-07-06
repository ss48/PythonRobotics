import copy
import matplotlib.pyplot as plt
import networkx as nx
import numpy as np
import numpy.random as rnd
import tsplib95
import tsplib95.distances as distances

from alns import ALNS
from alns.accept import HillClimbing
from alns.select import RouletteWheel
from alns.stop import MaxRuntime

SEED = 7654
DATA = tsplib95.load('/home/dell/ALNS/examples/data/xqf131.tsp')
CITIES = list(DATA.node_coords.keys())

COORDS = DATA.node_coords.values()
DIST = np.empty((len(COORDS) + 1, len(COORDS) + 1))

for row, coord1 in enumerate(COORDS, 1):
    for col, coord2 in enumerate(COORDS, 1):
        DIST[row, col] = distances.euclidean(coord1, coord2)

SOLUTION = tsplib95.load('/home/dell/ALNS/examples/data/xqf131.opt.tour')
OPTIMAL = DATA.trace_tours(SOLUTION.tours)[0]

print(f"Total optimal tour length is {OPTIMAL}.")


def draw_graph(graph, only_nodes=False, ax=None, title=None):
    """
    Helper method for drawing TSP (tour) graphs.
    """
    if ax is None:
        fig, ax = plt.subplots(figsize=(12, 6))

    if only_nodes:
        nx.draw_networkx_nodes(graph, DATA.node_coords, node_size=25, ax=ax)
    else:
        nx.draw_networkx(graph, DATA.node_coords, node_size=25, with_labels=False, ax=ax)

    if title:
        ax.set_title(title)


class TspState:
    """
    Solution class for the TSP problem. It has two data members, nodes, and edges.
    nodes is a list of IDs. The edges data member, then, is a mapping from each node
    to their only outgoing node.
    """

    def __init__(self, nodes, edges):
        self.nodes = nodes
        self.edges = edges

    def objective(self):
        """
        The objective function is simply the sum of all individual edge lengths,
        using the rounded Euclidean norm.
        """
        return sum(DIST[node, self.edges[node]] for node in self.nodes)

    def to_graph(self):
        """
        NetworkX helper method.
        """
        graph = nx.Graph()

        for node in self.nodes:
            graph.add_node(node, pos=DATA.node_coords[node])

        for node_from, node_to in self.edges.items():
            graph.add_edge(node_from, node_to)

        return graph


DEGREE_OF_DESTRUCTION = 0.1


def edges_to_remove(state):
    return int(len(state.edges) * DEGREE_OF_DESTRUCTION)


def worst_removal(current, rnd_state):
    """
    Worst removal iteratively removes the 'worst' edges, that is,
    those edges that have the largest distance.
    """
    destroyed = copy.deepcopy(current)

    worst_edges = sorted(destroyed.nodes,
                         key=lambda node: DIST[node, destroyed.edges[node]])

    for idx in range(edges_to_remove(current)):
        del destroyed.edges[worst_edges[-(idx + 1)]]

    return destroyed


def path_removal(current, rnd_state):
    """
    Removes an entire consecutive sub-path, that is, a series of
    contiguous edges.
    """
    destroyed = copy.deepcopy(current)

    node_idx = rnd_state.choice(len(destroyed.nodes))
    node = destroyed.nodes[node_idx]

    for _ in range(edges_to_remove(current)):
        node = destroyed.edges.pop(node)

    return destroyed


def random_removal(current, rnd_state):
    """
    Random removal iteratively removes random edges.
    """
    destroyed = copy.deepcopy(current)

    for idx in rnd_state.choice(len(destroyed.nodes),
                                edges_to_remove(current),
                                replace=False):
        del destroyed.edges[destroyed.nodes[idx]]

    return destroyed


def would_form_subcycle(from_node, to_node, state):
    """
    Ensures the proposed solution would not result in a cycle smaller
    than the entire set of nodes.
    """
    for step in range(1, len(state.nodes)):
        if to_node not in state.edges:
            return False

        to_node = state.edges[to_node]

        if from_node == to_node and step != len(state.nodes) - 1:
            return True

    return False


def greedy_repair(current, rnd_state):
    """
    Greedily repairs a tour, stitching up nodes that are not departed
    with those not visited.
    """
    visited = set(current.edges.values())

    shuffled_idcs = rnd_state.permutation(len(current.nodes))
    nodes = [current.nodes[idx] for idx in shuffled_idcs]

    while len(current.edges) != len(current.nodes):
        node = next(node for node in nodes
                    if node not in current.edges)

        unvisited = {other for other in current.nodes
                     if other != node
                     if other not in visited
                     if not would_form_subcycle(node, other, current)}

        nearest = min(unvisited,
                      key=lambda other: DIST[node, other])

        current.edges[node] = nearest
        visited.add(nearest)

    return current


# Initialize random state and initial solution
random_state = rnd.RandomState(SEED)
state = TspState(CITIES, {})

# Initial greedy repair solution
init_sol = greedy_repair(state, random_state)
print(f"Initial solution objective is {init_sol.objective()}.")

# Create subplots
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6))

# Plot both graphs
draw_graph(init_sol.to_graph(), ax=ax1, title="Initial Solution")
draw_graph(DATA.get_graph(), only_nodes=True, ax=ax2, title="Original Graph with Nodes Only")

plt.tight_layout()
plt.show()

