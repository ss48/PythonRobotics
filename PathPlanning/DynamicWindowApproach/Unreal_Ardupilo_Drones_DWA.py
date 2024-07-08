#!/usr/bin/env python
# -*- coding: utf-8 -*-

from __future__ import print_function
import numpy as np
import math
import matplotlib.pyplot as plt
import matplotlib.animation as animation
import threading
import queue
import random
import argparse
from dronekit import connect, VehicleMode, LocationGlobalRelative
from pymavlink import mavutil  # Needed for command message definitions
from dwa import dwa_control, motion, RobotType, plot_robot

# Files for destinations and fences
DATA_FILE = '/home/dell/ALNS/Destinations.tsp'
FENCES_FILE = '/home/dell/ALNS/Fences.tsp'

class RobotType:
    circle = 1
    rectangle = 2

def read_positions(file_path):
    """Reads positions from a TSP file format."""
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

def read_fences(file_path, radius=0.003, num_points=20):
    """Reads fence positions from a TSP file format and converts them to polygonal obstacles."""
    fences = []
    centers = []
    with open(file_path, 'r') as f:
        reading_nodes = False
        for line in f:
            if "NODE_COORD_SECTION" in line:
                reading_nodes = True
            elif "EOF" in line:
                break
            elif reading_nodes:
                _, x, y = line.split()
                center_x, center_y = float(x), float(y)
                centers.append((center_x, center_y))
                # Generate points around the fence center to form a polygon
                for i in range(num_points):
                    angle = 2 * math.pi * i / num_points
                    px = center_x + radius * math.cos(angle)
                    py = center_y + radius * math.sin(angle)
                    fences.append([px, py])
    return fences, centers

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
        self.ob, self.centers = read_fences(fences_file)

def get_location_offset(location, dNorth, dEast):
    earth_radius = 6378137.0  # Radius of "spherical" earth
    dLat = dNorth / earth_radius
    dLon = dEast / (earth_radius * math.cos(math.pi * location.lat / 180))

    newlat = location.lat + (dLat * 180 / math.pi)
    newlon = location.lon + (dLon * 180 / math.pi)
    return LocationGlobalRelative(newlat, newlon, location.alt)

def plot_destinations_and_fences(ax, start_position, destinations, fences, current_goals, fence_centers):
    ax.plot(start_position[1], start_position[0], 'bo', label="Start Position")
    for i, dest in enumerate(destinations):
        color = 'ro' if dest in current_goals else 'go'
        ax.plot(dest[1], dest[0], color)
        ax.text(dest[1] + 0.001, dest[0] + 0.001, f'{i}', fontsize=8, color='red' if dest in current_goals else 'green')
    
    # Group fence points into polygons
    num_points = 20  # Assuming the number of points used to generate each fence
    for center in fence_centers:
        polygon = fences[:num_points]
        fences = fences[num_points:]
        fence_x, fence_y = zip(*polygon)
        ax.fill(fence_x, fence_y, 'r', alpha=0.5)  # Draw the fence area

def drone_movement(vehicle, config, vehicle_id, start_position, destinations, plot_queue, path, current_goal):
    x = np.array([start_position[0], start_position[1], 0.0, 0.0, 0.0])  # Initial state

    remaining_destinations = destinations.copy()

    while True:
        if not remaining_destinations:
            remaining_destinations = destinations.copy()

        destination = random.choice(remaining_destinations)
        current_goal[vehicle_id - 1] = destination
        print(f"Drone {vehicle_id} planning to go to destination: {destinations.index(destination)} at {destination}")
        goal = np.array([destination[0], destination[1]])
        ob = config.ob

        while True:
            print(f"Drone {vehicle_id} current position: {x[0]:.6f}, {x[1]:.6f}, yaw: {x[2]:.6f}")
            print(f"Obstacle positions: {ob}")

            # Call DWA control to get the control inputs and the trajectory
            u, trajectory = dwa_control(x, config, goal, ob)
            # Update the drone's state using the motion model
            x = motion(x, u, config.dt)

            path.append((x[0], x[1]))

            next_location = LocationGlobalRelative(x[0], x[1], vehicle.location.global_relative_frame.alt)
            print(f"Drone {vehicle_id} moving to: {next_location.lat:.6f}, {next_location.lon:.6f}")
            vehicle.simple_goto(next_location)

            # Add data to plot_queue
            plot_queue.put((x[0], x[1], vehicle_id))

            # Check if the drone has reached its goal
            dist_to_goal = math.hypot(x[0] - goal[0], x[1] - goal[1])
            if dist_to_goal <= config.robot_radius:
                print(f"Drone {vehicle_id} reached destination at {destination}")
                remaining_destinations.remove(destination)
                break

        print(f"Drone {vehicle_id} selecting new destination...")

def plot_update(frame, plot_queue, ax, start_position, destinations, fences, config1, config2, paths, current_goals, fence_centers):
    ax.cla()  # Clear the axis to update the plot
    plot_destinations_and_fences(ax, start_position, destinations, fences, current_goals, fence_centers)
    for path in paths:
        if path:
            xs, ys = zip(*path)
            ax.plot(ys, xs, '-k', label='DWA Path')  # Use black color for the path
            ax.quiver(ys[:-1], xs[:-1], np.diff(ys), np.diff(xs), scale_units='xy', angles='xy', scale=1, color='blue')

    while not plot_queue.empty():
        x, y, vehicle_id = plot_queue.get()
        config = config1 if vehicle_id == 1 else config2
        plot_robot(x, y, 0, config)  # Assuming yaw=0 for simplicity

    ax.set_aspect('equal')
    ax.grid(True)
    ax.legend(loc='best')
    ax.set_xlim(0, 1)  # Set larger axis limits
    ax.set_ylim(51, 52)  # Set larger axis limits

def main():
    parser = argparse.ArgumentParser(description='Control Copter and send commands in GUIDED mode')
    parser.add_argument('--connect1', help="Vehicle 1 connection target string.")
    parser.add_argument('--connect2', help="Vehicle 2 connection target string.")
    args = parser.parse_args()
    connection_string1 = args.connect1 if args.connect1 else 'udpin:127.0.0.1:14552'
    connection_string2 = args.connect2 if args.connect2 else 'udpin:127.0.0.1:14553'
    
   
    print('Connecting to vehicle on:', connection_string1)
    vehicle1 = connect(connection_string1, wait_ready=True, timeout=60)
    print('Connected to vehicle 1')

    print('Connecting to vehicle on:', connection_string2)
    vehicle2 = connect(connection_string2, wait_ready=True, timeout=60)
    print('Connected to vehicle 2')

    start_position = [51.73, 0.483, 56]  # Starting point for visualization

    # Read destinations and fences from files
    destinations = read_positions(DATA_FILE)
    fences, fence_centers = read_fences(FENCES_FILE)

    # Convert fences to obstacles for the DWA config
    obstacles = np.array(fences)

    config1 = Config(FENCES_FILE)
    config1.robot_type = RobotType.circle
    config1.ob = obstacles

    config2 = Config(FENCES_FILE)
    config2.robot_type = RobotType.circle
    config2.ob = obstacles

    plot_queue = queue.Queue()

    paths = [[], []]  # Separate paths for each drone
    current_goals = [None, None]  # Current goal for each drone

    vehicle1_thread = threading.Thread(target=lambda: drone_movement(vehicle1, config1, 1, start_position, destinations, plot_queue, paths[0], current_goals))
    vehicle1_thread.daemon = True
    vehicle1_thread.start()

    vehicle2_thread = threading.Thread(target=lambda: drone_movement(vehicle2, config2, 2, start_position, destinations, plot_queue, paths[1], current_goals))
    vehicle2_thread.daemon = True
    vehicle2_thread.start()

    fig, ax = plt.subplots(figsize=(15, 15))  # Increase the figure size to make the plot larger
    ani = animation.FuncAnimation(fig, plot_update, fargs=(plot_queue, ax, start_position, destinations, fences, config1, config2, paths, current_goals, fence_centers), interval=100, cache_frame_data=False)

    plt.show()

if __name__ == '__main__':
    main()
