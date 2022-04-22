import sys
import time
import matplotlib.pyplot as plt
import math
import numpy as np
import pandas as pd
sys.path.append("../../hopper_plant/")
from hopper_plant import HopperPlant

# Create Hopper multi-body plant
L1 = 0.205
L2 = 0.25
g = 9.81
plant = HopperPlant(link_length=[L1, L2], gravity=g)

# Task space trajectory
h = 0.1    # vertical height
A = h/2.    # Amplitude
offset = L1 + L2 - A    # offset
T = 5.  # time period
omega = 2*np.pi/T

# Joint space gains
Kp_shoulder = 25.
Kd_shoulder = 2.
Kp_elbow = 25.
Kd_elbow = 2.

print("Start")
time_vec_des = np.linspace(0,15,4500)
numSteps = len(time_vec_des)

foot_xd = np.zeros(numSteps)
foot_yd = np.zeros(numSteps)

shoulder_position_des = np.zeros(numSteps)
elbow_position_des = np.zeros(numSteps)

shoulder_velocity_des = np.zeros(numSteps)
elbow_velocity_des = np.zeros(numSteps)

# Desired Cartesian Coordinates

foot_x = np.zeros(numSteps)
foot_y = np.zeros(numSteps)
foot_xd = np.zeros(numSteps)
foot_yd = np.zeros(numSteps)

for i in range(numSteps):

    # Compute IK
    x_des = A*np.cos(omega*time_vec_des[i]) + offset
    y_des = 0.0
    shoulder_pos_des, elbow_pos_des = plant.inverse_kinematics(x_des, y_des, knee_direction=1)
    xd_des = -omega*A*np.sin(omega*time_vec_des[i])
    yd_des = 0.0
    shoulder_vel_des, elbow_vel_des = plant.inverse_velocity(shoulder_pos_des, elbow_pos_des, xd_des, yd_des)    
    shoulder_position_des[i] = shoulder_pos_des
    elbow_position_des[i] = elbow_pos_des
    shoulder_velocity_des[i] = shoulder_vel_des
    elbow_velocity_des[i] = elbow_vel_des

    foot_x[i] = x_des
    foot_y[i] = y_des
    foot_xd[i] = xd_des
    foot_yd[i] = yd_des

print("End")
plt.figure
plt.plot(time_vec_des, shoulder_position_des)
plt.plot(time_vec_des, elbow_position_des)
plt.xlabel("Time (s)")
plt.ylabel("Position (rad)")
plt.title("Position (rad) vs Time (s)")
plt.legend(['Shoulder Desired', 'Elbow Desired'])
plt.show()

plt.figure
plt.plot(time_vec_des, foot_x)
plt.plot(time_vec_des, foot_y)
plt.xlabel("Time (s)")
plt.ylabel("Cartesian Position (m)")
plt.title("Cartesian Position (m) vs Time (s)")
plt.legend(['Foot X Desired', 'Foot Y Desired'])
plt.show()

plt.figure
plt.plot(foot_x, foot_y)
plt.xlabel("Cartesian Position X (m)")
plt.ylabel("Cartesian Position Y (m)")
plt.title("Cartesian Position X (m) vs Y (m)")
plt.show()

plt.figure
plt.plot(time_vec_des, shoulder_velocity_des)
plt.plot(time_vec_des, elbow_velocity_des)
plt.xlabel("Time (s)")
plt.ylabel("Velocity (rad/s)")
plt.legend(['Shoulder Desired', 'Elbow Desired'])
plt.title("Velocity (rad/s) vs Time (s)")
plt.show()

plt.figure
plt.plot(time_vec_des, foot_xd)
plt.plot(time_vec_des, foot_yd)
plt.xlabel("Time (s)")
plt.ylabel("Cartesian Velocity (m/s)")
plt.title("Cartesian Velocity (m/s) vs Time (s)")
plt.legend(['Foot Xdot Desired', 'Foot Ydot Desired'])
plt.show()

desired_csv_data = np.array([np.array(time_vec_des),
                    np.array(shoulder_position_des),
                    np.array(shoulder_velocity_des),
                    np.zeros(len(time_vec_des)),
                    np.array(elbow_position_des),
                    np.array(elbow_velocity_des),
                    np.zeros(len(time_vec_des))]).T
np.savetxt("desired_data.csv", desired_csv_data, delimiter=',', header="time,shoulder_pos,shoulder_vel,shoulder_torque,elbow_pos,elbow_vel,elbow_torque", comments="")
