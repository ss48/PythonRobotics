import copy
import matplotlib.pyplot as plt
import networkx as nx
import numpy as np
import numpy.random as rnd
import tsplib95
import tsplib95.distances as distances
from math import radians, sin, cos, sqrt, atan2

from alns import ALNS
from alns.accept import HillClimbing
from alns.select import RouletteWheel
from alns.stop import MaxRuntime

# Set seed for reproducibility
SEED = 7654

# Load TSP data
DATA = tsplib95.load('/home/dell/ALNS/Destinations.tsp')
CITIES = list(DATA.node_coords.keys())

# Load fences data
def load_fences(filename):
    with open(filename, 'r') as file:
        lines = file.readlines()
    fences = []
    read_fences = False
    for line in lines:
        if line.strip() == "EDGE_COORD_SECTION":
            read_fences = True
            continue
        if read_fences:
            if line.strip() == "EOF":
                break
            parts = list(map(float, line.strip().split()))
            if len(parts) == 4:
                fences.append(parts)
    return fences

FENCES = load_fences('/home/dell/ALNS/Fences.tsp')

def intersects_fence(coord1, coord2, fence):
    # Function to check if a line segment from coord1 to coord2 intersects a fence
    x1, y1 = coord1
    x2, y2 = coord2
    fx1, fy1, fx2, fy2 = fence

    def ccw(A, B, C):
        return (C[1] - A[1]) * (B[0] - A[0]) > (B[1] - A[1]) * (C[0] - A[0])

    return ccw((x1, y1), (fx1, fy1), (fx2, fy2)) != ccw((x2, y2), (fx1, fy1), (fx2, fy2)) and \
           ccw((x1, y1), (x2, y2), (fx1, fy1)) != ccw((x1, y1), (x2, y2), (fx2, fy2))

def path_valid(coord1, coord2):
    return not any(intersects_fence(coord1, coord2, fence) for fence in FENCES)

def haversine_distance(coord1, coord2):
    # Convert latitude and longitude from degrees to radians
    lat1, lon1 = map(radians, coord1)
    lat2, lon2 = map(radians, coord2)

    # Haversine formula
    dlat = lat2 - lat1
    dlon = lon2 - lon1
    a = sin(dlat / 2)**2 + cos(lat1) * cos(lat2) * sin(dlon / 2)**2
    c = 2 * atan2(sqrt(a), sqrt(1 - a))
    radius_of_earth_km = 6371.0  # Radius of the Earth in kilometers
    distance = radius_of_earth_km * c

    return distance

# Precompute distance matrix with fences
COORDS = list(DATA.node_coords.values())
DIST = np.empty((len(COORDS) + 1, len(COORDS) + 1))
for row, coord1 in enumerate(COORDS, 1):
    for col, coord2 in enumerate(COORDS, 1):
        if row == col:
            DIST[row, col] = 0
        else:
            dist = haversine_distance(coord1, coord2)
            if not path_valid(coord1, coord2):
                dist = np.inf  # Use a very large number to indicate an impassable path
            DIST[row, col] = dist

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

    # Draw fences
    for fence in FENCES:
        x_values = [fence[0], fence[2]]
        y_values = [fence[1], fence[3]]
        ax.plot(x_values, y_values, 'r-')

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
                     if not would_form_subcycle(node, other, current)
                     if path_valid(DATA.node_coords[node], DATA.node_coords[other])}
        if not unvisited:
            # If no valid unvisited nodes remain, we need to backtrack
            continue
        nearest = min(unvisited, key=lambda other: DIST[node, other])
        current.edges[node] = nearest
        visited.add(nearest)
    return current

# Initialize random state
random_state = rnd.RandomState(SEED)
state = TspState(CITIES, {})

# Initial solution using greedy repair
init_sol = greedy_repair(state, random_state)
initial_objective = init_sol.objective()
print(f"Initial solution objective (distance): {initial_objective:.2f} km")

# ALNS initialization and parameters
alns = ALNS(random_state)
alns.add_destroy_operator(random_removal)
alns.add_destroy_operator(path_removal)
alns.add_destroy_operator(worst_removal)
alns.add_repair_operator(greedy_repair)

# ALNS parameters and execution
select = RouletteWheel([3, 2, 1, 0.5], 0.8, 3, 1)
accept = HillClimbing()
stop = MaxRuntime(600)  # Increased runtime for better chances of finding improvements

# Run ALNS on the initial solution
result = alns.iterate(init_sol, select, accept, stop)
solution = result.best_state
solution_objective = solution.objective()

print(f"Best solution objective (distance) avoiding fences: {solution_objective:.2f} km")

# Check if the best solution is different from the initial solution
if initial_objective == solution_objective:
    print("ALNS did not find a better solution.")

# Plotting the initial solution, ALNS objectives, and the best solution
fig, axs = plt.subplots(1, 3, figsize=(24, 8))

# Plot the initial solution
draw_graph(init_sol.to_graph(), ax=axs[0])
axs[0].set_title('Initial Solution')

# Plot the ALNS objectives
result.plot_objectives(ax=axs[1], lw=2)
axs[1].set_title('ALNS Objectives')

# Plot the best solution found by ALNS
draw_graph(solution.to_graph(), ax=axs[2])
axs[2].set_title('Best Solution Found by ALNS')

# Draw fences on all plots for clarity
for ax in axs:
    for fence in FENCES:
        x_values = [fence[0], fence[2]]
        y_values = [fence[1], fence[3]]
        ax.plot(x_values, y_values, 'r-')

# Adjust layout and display the plots
plt.tight_layout()
plt.show()

