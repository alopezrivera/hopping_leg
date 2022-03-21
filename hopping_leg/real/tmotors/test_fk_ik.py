import sys
import time
from canmotorlib import CanMotorController
import matplotlib.pyplot as plt
import math
import numpy as np
import pandas as pd
sys.path.append("../../hopper_plant/")
from hopper_plant import HopperPlant

def setZeroPosition(motor, initPos):

    pos = initPos

    while abs(np.rad2deg(pos)) > 0.5:
        pos, vel, curr = motor.set_zero_position()
        print("Position: {}, Velocity: {}, Torque: {}".format(np.rad2deg(pos), np.rad2deg(vel),
                                                                curr))


# Motor ID
motor_shoulder_id = 0x03
motor_elbow_id = 0x04

# CAN port
can_port = 'can0'

if len(sys.argv) != 2:
    print('Provide CAN device name (can0, slcan0 etc.)')
    sys.exit(0)

print("Using Socket {} for can communucation".format(sys.argv[1],))

# Create Hopper multi-body plant
L1 = 0.205
L2 = 0.25
g=9.81
plant = HopperPlant(link_length=[L1, L2], gravity=g)

# Create motor controller objects
motor_shoulder_controller = CanMotorController(sys.argv[1], motor_shoulder_id)
motor_elbow_controller = CanMotorController(sys.argv[1], motor_elbow_id)

print("Enabling Motors..")

shoulder_pos, shoulder_vel, shoulder_torque = motor_shoulder_controller.enable_motor()

print("Shoulder Motor Status: Pos: {}, Vel: {}, Torque: {}".format(shoulder_pos, shoulder_vel, shoulder_torque))

elbow_pos, elbow_vel, elbow_torque = motor_elbow_controller.enable_motor()

print("Elbow Motor Status: Pos: {}, Vel: {}, Torque: {}".format(elbow_pos, elbow_vel, elbow_torque))

print("Setting Shoulder Motor to Zero Position...")

setZeroPosition(motor_shoulder_controller, shoulder_pos)

print("Setting Elbow Motor to Zero Position...")

setZeroPosition(motor_elbow_controller, elbow_pos)

print("Start")
numSteps = 10000

time_vec = np.zeros(numSteps)

shoulder_position = np.zeros(numSteps)
elbow_position = np.zeros(numSteps)

shoulder_velocity = np.zeros(numSteps)
elbow_velocity = np.zeros(numSteps)

shoulder_torque = np.zeros(numSteps)
elbow_torque = np.zeros(numSteps)

foot_x = np.zeros(numSteps)
foot_y = np.zeros(numSteps)

foot_xd = np.zeros(numSteps)
foot_yd = np.zeros(numSteps)

shoulder_position_ik = np.zeros(numSteps)
elbow_position_ik = np.zeros(numSteps)

shoulder_velocity_ik = np.zeros(numSteps)
elbow_velocity_ik = np.zeros(numSteps)

start_time = time.time()

for i in range(numSteps):

    dt = time.time()
    traj_time = dt - start_time
    
    time_vec[i] = traj_time
    # Send pos, vel and tau_ff command and use the in-built low level controller
    shoulder_pos, shoulder_vel, shoulder_tau = motor_shoulder_controller.send_rad_command(0.0,0.0,0.0,0.0,0.0)
    elbow_pos, elbow_vel, elbow_tau = motor_elbow_controller.send_rad_command(0.0,0.0,0.0,0.0,0.0)

    # Store data in lists
    shoulder_position[i] = shoulder_pos
    shoulder_velocity[i] = shoulder_vel
    shoulder_torque[i] = shoulder_tau

    elbow_position[i] = elbow_pos
    elbow_velocity[i] = elbow_vel
    elbow_torque[i] = elbow_tau

    # Compute FK
    x, y = plant.forward_kinematics(shoulder_pos, elbow_pos)
    xd, yd = plant.forward_velocity(shoulder_pos, elbow_pos, shoulder_vel, elbow_vel)    
    foot_x[i] = x
    foot_y[i] = y
    foot_xd[i] = xd
    foot_yd[i] = yd

    # Compute IK
    shoulder_pos_ik, elbow_pos_ik = plant.inverse_kinematics(x, y)
    shoulder_vel_ik, elbow_vel_ik = plant.inverse_velocity(shoulder_pos, elbow_pos, xd, yd)    
    shoulder_position_ik[i] = shoulder_pos_ik
    elbow_position_ik[i] = elbow_pos_ik
    shoulder_velocity_ik[i] = shoulder_vel_ik
    elbow_velocity_ik[i] = elbow_vel_ik

print("Disabling Motors...")

shoulder_pos, shoulder_vel, shoulder_tau = motor_shoulder_controller.disable_motor()

print("Shoulder Motor Status: Pos: {}, Vel: {}, Torque: {}".format(shoulder_pos, shoulder_vel, shoulder_tau))

elbow_pos, elbow_vel, elbow_tau = motor_elbow_controller.disable_motor()

print("Elbow Motor Status: Pos: {}, Vel: {}, Torque: {}".format(elbow_pos, elbow_vel, elbow_tau))

plt.figure
plt.plot(time_vec, shoulder_position)
plt.plot(time_vec, elbow_position)
plt.plot(time_vec, shoulder_position_ik)
plt.plot(time_vec, elbow_position_ik)
plt.xlabel("Time (s)")
plt.ylabel("Position (rad)")
plt.title("Position (rad) vs Time (s)")
plt.legend(['Shoulder', 'Elbow', 'Shoulder IK', 'Elbow IK'])
plt.show()

plt.figure
plt.plot(time_vec, foot_x)
plt.plot(time_vec, foot_y)
plt.xlabel("Time (s)")
plt.ylabel("Cartesian Position (m)")
plt.title("Cartesian Position (m) vs Time (s)")
plt.legend(['Foot X', 'Foot Y'])
plt.show()

plt.figure
plt.plot(foot_x, foot_y)
plt.xlabel("Cartesian Position X (m)")
plt.ylabel("Cartesian Position Y (m)")
plt.title("Cartesian Position X (m) vs Y (m)")
plt.show()

plt.figure
plt.plot(time_vec, shoulder_velocity)
plt.plot(time_vec, elbow_velocity)
plt.plot(time_vec, shoulder_velocity_ik)
plt.plot(time_vec, elbow_velocity_ik)
plt.xlabel("Time (s)")
plt.ylabel("Velocity (rad/s)")
plt.legend(['Shoulder', 'Elbow', 'Shoulder IK', 'Elbow IK'])
plt.title("Velocity (rad/s) vs Time (s)")
plt.show()

plt.figure
plt.plot(time_vec, foot_xd)
plt.plot(time_vec, foot_yd)
plt.xlabel("Time (s)")
plt.ylabel("Cartesian Velocity (m/s)")
plt.title("Cartesian Velocity (m/s) vs Time (s)")
plt.legend(['Foot Xdot', 'Foot Ydot'])
plt.show()

plt.figure
plt.plot(time_vec, shoulder_torque)
plt.plot(time_vec, elbow_torque)
plt.xlabel("Time (s)")
plt.ylabel("Torque (Nm)")
plt.title("Torque (Nm) vs Time (s)")
plt.legend(['Shoulder', 'Elbow'])
plt.show()

def running_mean(x, N):
    cumsum = np.cumsum(np.insert(x, 0, 0))
    return (cumsum[N:] - cumsum[:-N]) / float(N)


filtered_shoulder_torque = running_mean(np.array(shoulder_torque), 10)
filtered_elbow_torque = running_mean(np.array(elbow_torque), 10)
time_vec_filtered = running_mean(np.array(time_vec), 10)

plt.figure
plt.plot(time_vec_filtered, filtered_shoulder_torque)
plt.plot(time_vec_filtered, filtered_elbow_torque)
plt.xlabel("Time (s)")
plt.ylabel("Torque (Nm)")
plt.title("Filtered Torque (Nm) vs Time (s) with moving average filter (window = 100)")
plt.legend(['Shoulder', 'Elbow'])
plt.show()


measured_csv_data = np.array([np.array(time_vec),
                    np.array(shoulder_position),
                    np.array(shoulder_velocity),
                    np.array(shoulder_torque),
                    np.array(elbow_position),
                    np.array(elbow_velocity),
                    np.array(elbow_torque)]).T
np.savetxt("measured_data.csv", measured_csv_data, delimiter=',', header="time,shoulder_pos,shoulder_vel,shoulder_torque,elbow_pos,elbow_vel,elbow_torque", comments="")
