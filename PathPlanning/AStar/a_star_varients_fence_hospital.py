import numpy as np
import matplotlib.pyplot as plt

show_animation = True

DATA_FILE = '/home/dell/ALNS/Destinations.tsp'
FENCES_FILE = '/home/dell/ALNS/Fences.tsp'

def read_positions(file_path):
    """ Reads positions from a TSP file format. """
    positions = []
    with open(file_path, 'r') as f:
        reading_nodes = False
        for line in f:
            if "NODE_COORD_SECTION" in line:
                reading_nodes = True
            elif "EOF" in line:
                break
            elif reading_nodes:
                _, x, y = line.split()
                positions.append((float(x), float(y)))
    return positions

def setup_obstacle_dict(fences):
    """ Creates a dictionary of obstacles from fence coordinates. """
    obs_dict = {}
    for x, y in fences:
        obs_dict[(int(x * 100), int(y * 100))] = True  # Increase resolution by scaling coordinates
    return obs_dict

def in_line_of_sight(obs_grid, x1, y1, x2, y2):
    t = 0
    while t <= 1:
        xt = (1 - t) * x1 + t * x2
        yt = (1 - t) * y1 + t * y2
        if obs_grid.get((int(xt), int(yt)), False):
            return False
        t += 0.01
    return True

class AStarPlanner:
    """ Implements the A* search algorithm. """
    def __init__(self, obstacles):
        self.obstacles = obstacles

    def heuristic(self, x1, y1, x2, y2):
        """ Euclidean distance heuristic for A* """
        return np.hypot(x1 - x2, y1 - y2)

    def a_star_search(self, start, goal):
        """ Performs A* search from start to goal. """
        start = (int(start[0] * 100), int(start[1] * 100))
        goal = (int(goal[0] * 100), int(goal[1] * 100))

        open_set = {start: (0, self.heuristic(*start, *goal), -1, -1)}
        closed_set = {}
        path_found = False

        fig, ax = plt.subplots(figsize=(12, 8), dpi=150)
        plt.xlim(5100, 5250)
        plt.ylim(-90, 120)
        for (x, y) in self.obstacles:
            plt.plot(x, y, 'xr')

        plt.plot(start[0], start[1], 'ob', markersize=10)
        plt.plot(goal[0], goal[1], 'or', markersize=10)

        while open_set:
            current = min(open_set, key=lambda o: open_set[o][0] + open_set[o][1])
            if current == goal:
                path_found = True
                path = []
                while current in closed_set:
                    path.append((current[0] / 100, current[1] / 100))
                    current = (closed_set[current][1], closed_set[current][2])
                path.append((start[0] / 100, start[1] / 100))
                path = path[::-1]
                break
            g, _, cx, cy = open_set.pop(current)
            closed_set[current] = (g, cx, cy)
            for dx, dy in [(-1, 0), (1, 0), (0, -1), (0, 1), (-1, -1), (1, 1), (-1, 1), (1, -1)]:
                neighbor = (current[0] + dx, current[1] + dy)
                if neighbor in closed_set or self.obstacles.get(neighbor, False):
                    continue
                if in_line_of_sight(self.obstacles, current[0], current[1], neighbor[0], neighbor[1]):
                    g_cost = g + np.hypot(dx, dy)
                    if neighbor not in open_set or g_cost < open_set[neighbor][0]:
                        h_cost = self.heuristic(neighbor[0], neighbor[1], goal[0], goal[1])
                        open_set[neighbor] = (g_cost, h_cost, current[0], current[1])

                        # Animation step: plot the current node
                        if show_animation:
                            plt.plot(neighbor[0], neighbor[1], 'xc')
                            plt.pause(0.001)

        if path_found:
            xs, ys = zip(*path)
            plt.plot(xs, ys, '-g')
        else:
            print(f"No path found between {start} and {goal}")

        plt.grid(True)
        plt.xlabel('Longitude')
        plt.ylabel('Latitude')
        plt.title('Path Planning with A*')
        plt.legend(['Obstacle', 'Start', 'Goal', 'Path', 'Visited Node'], loc='best')
        plt.show()

        return path if path_found else []

def main():
    print("Starting path planning...")
    destinations = read_positions(DATA_FILE)
    fences = read_positions(FENCES_FILE)
    obstacles = setup_obstacle_dict(fences)

    a_star = AStarPlanner(obstacles)

    # Loop through all destinations
    for i in range(len(destinations) - 1):
        start, goal = destinations[i], destinations[i + 1]
        print(f"Searching path from {start} to {goal}...")
        path = a_star.a_star_search(start, goal)
        if path:
            print(f"Path found: {path}")
        else:
            print(f"No path found between {start} and {goal}")

if __name__ == '__main__':
    main()

