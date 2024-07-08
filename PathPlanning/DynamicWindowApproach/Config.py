#!/usr/bin/env python
# -*- coding: utf-8 -*-

from __future__ import print_function
def read_fences(file_path):
    """Reads fence positions from a TSP file format and converts them to obstacles."""
    fences = []
    with open(file_path, 'r') as f:
        reading_nodes = False
        for line in f:
            if "NODE_COORD_SECTION" in line:
                reading_nodes = True
            elif "EOF" in line:
                break
            elif reading_nodes:
                _, x, y = line.split()
                fences.append([float(x), float(y)])
    return fences

class Config:
    def __init__(self, fences_file):
        self.max_speed = 1.0  # [m/s]
        self.min_speed = -0.5  # [m/s]
        self.max_yaw_rate = 40.0 * math.pi / 180.0  # [rad/s]
        self.max_accel = 0.2  # [m/ss]
        self.max_delta_yaw_rate = 40.0 * math.pi / 180.0  # [rad/ss]
        self.v_resolution = 0.01  # [m/s]
        self.yaw_rate_resolution = 0.1 * math.pi / 180.0  # [rad/s]
        self.dt = 0.1  # [s] Time tick for motion prediction
        self.predict_time = 3.0  # [s]
        self.to_goal_cost_gain = 0.15
        self.speed_cost_gain = 1.0
        self.obstacle_cost_gain = 1.0
        self.robot_stuck_flag_cons = 0.001  # constant to prevent robot stucked
        self.robot_type = RobotType.circle
        self.robot_radius = 1.0  # [m] for collision check
        self.robot_width = 0.5  # [m] for collision check
        self.robot_length = 1.2  # [m] for collision check
        
        # Read the obstacles from the fences file
        self.ob = np.array(read_fences(fences_file))
        # Usage example
FENCES_FILE = '/home/dell/ALNS/Fences.tsp'
config = Config(FENCES_FILE)

print("Obstacle coordinates:")
print(config.ob)
