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

# Set seed for reproducibility
SEED = 7654

# Load TSP data
DATA = tsplib95.load('/home/dell/ALNS/examples/data/xqf131.tsp')
CITIES = list(DATA.node_coords.keys())

# Precompute distance matrix
COORDS = DATA.node_coords.values()
DIST = np.empty((len(COORDS) + 1, len(COORDS) + 1))
for row, coord1 in enumerate(COORDS, 1):
    for col, coord2 in enumerate(COORDS, 1):
        DIST[row, col] = distances.euclidean(coord1, coord2)

# Load optimal solution for comparison
SOLUTION = tsplib95.load('/home/dell/ALNS/examples/data/xqf131.opt.tour')
OPTIMAL = DATA.trace_tours(SOLUTION.tours)[0]
print(f"Total optimal tour length is {OPTIMAL}.")

# Helper function to draw TSP graphs
def draw_graph(graph, only_nodes=False, ax=None):
    if ax is None:
        fig, ax = plt.subplots(figsize=(12, 6))
    
    pos = DATA.node_coords
    if only_nodes:
        nodes_to_draw = [node for node in graph.nodes if node in pos]
        nx.draw_networkx_nodes(graph, pos, nodelist=nodes_to_draw, node_size=25, ax=ax)
    else:
        nx.draw_networkx(graph, pos, with_labels=False, ax=ax)


# Solution class for TSP problem
class TspState:
    def __init__(self, nodes, edges):
        self.nodes = nodes
        self.edges = edges
    
    def objective(self):
        return sum(DIST[node, self.edges[node]] for node in self.nodes)
    
    def to_graph(self):
        graph = nx.Graph()
        for node in self.nodes:
            graph.add_node(node, pos=DATA.node_coords[node])
        for node_from, node_to in self.edges.items():
            graph.add_edge(node_from, node_to)
        return graph

# Operators for ALNS
DEGREE_OF_DESTRUCTION = 0.1

def edges_to_remove(state):
    return int(len(state.edges) * DEGREE_OF_DESTRUCTION)

def worst_removal(current, rnd_state):
    destroyed = copy.deepcopy(current)
    worst_edges = sorted(destroyed.nodes, key=lambda node: DIST[node, destroyed.edges[node]])
    for idx in range(edges_to_remove(current)):
        del destroyed.edges[worst_edges[-(idx + 1)]]
    return destroyed

def path_removal(current, rnd_state):
    destroyed = copy.deepcopy(current)
    node_idx = rnd_state.choice(len(destroyed.nodes))
    node = destroyed.nodes[node_idx]
    for _ in range(edges_to_remove(current)):
        node = destroyed.edges.pop(node)
    return destroyed

def random_removal(current, rnd_state):
    destroyed = copy.deepcopy(current)
    for idx in rnd_state.choice(len(destroyed.nodes), edges_to_remove(current), replace=False):
        del destroyed.edges[destroyed.nodes[idx]]
    return destroyed

def would_form_subcycle(from_node, to_node, state):
    for step in range(1, len(state.nodes)):
        if to_node not in state.edges:
            return False
        to_node = state.edges[to_node]
        if from_node == to_node and step != len(state.nodes) - 1:
            return True
    return False

def greedy_repair(current, rnd_state):
    visited = set(current.edges.values())
    shuffled_idcs = rnd_state.permutation(len(current.nodes))
    nodes = [current.nodes[idx] for idx in shuffled_idcs]
    while len(current.edges) != len(current.nodes):
        node = next(node for node in nodes if node not in current.edges)
        unvisited = {other for other in current.nodes if other != node
                     if other not in visited
                     if not would_form_subcycle(node, other, current)}
        nearest = min(unvisited, key=lambda other: DIST[node, other])
        current.edges[node] = nearest
        visited.add(nearest)
    return current

# Initialize random state
random_state = rnd.RandomState(SEED)
state = TspState(CITIES, {})

# Initial solution using greedy repair
init_sol = greedy_repair(state, random_state)
print(f"Initial solution objective is {init_sol.objective()}.")

# ALNS initialization and parameters
alns = ALNS(random_state)
alns.add_destroy_operator(random_removal)
alns.add_destroy_operator(path_removal)
alns.add_destroy_operator(worst_removal)
alns.add_repair_operator(greedy_repair)

# ALNS parameters and execution
select = RouletteWheel([3, 2, 1, 0.5], 0.8, 3, 1)
accept = HillClimbing()
stop = MaxRuntime(60)

# Run ALNS on the initial solution
result = alns.iterate(init_sol, select, accept, stop)
solution = result.best_state
objective = solution.objective()
pct_diff = 100 * (objective - OPTIMAL) / OPTIMAL

print(f"Best heuristic objective is {objective}.")
print(f"This is {pct_diff:.1f}% worse than the optimal solution, which is {OPTIMAL}.")

# Plotting all graphs side by side
fig, axs = plt.subplots(1, 3, figsize=(18, 6))

# Plot the initial solution
draw_graph(init_sol.to_graph(), ax=axs[0])
axs[0].set_title('Initial Solution')

# Plot the ALNS objectives
result.plot_objectives(ax=axs[1], lw=2)
axs[1].set_title('ALNS Objectives')

# Plot the optimal solution
draw_graph(SOLUTION.get_graph(), only_nodes=True, ax=axs[2])
axs[2].set_title('Optimal Solution')

# Adjust layout and display the plots
plt.tight_layout()
plt.show()

