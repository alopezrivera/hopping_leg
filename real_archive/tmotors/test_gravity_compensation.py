import sys
import time
#from motor_driver.canmotorlib import CanMotorController
from canmotorlib import CanMotorController
import matplotlib.pyplot as plt
import math
import numpy as np
import pandas as pd
sys.path.append("../../hopper_plant/")
from hopper_plant import HopperPlant
#import hyrodyn 
import os

def setZeroPosition(motor, initPos):

    pos = initPos

    while abs(np.rad2deg(pos)) > 0.5:
        pos, vel, curr = motor.set_zero_position()
        print("Position: {}, Velocity: {}, Torque: {}".format(np.rad2deg(pos), np.rad2deg(vel),
                                                                curr))
'''
# Path to URDF and Submechanism.yml files 
path_to_urdf =  os.environ["AUTOPROJ_CURRENT_ROOT"] + "/../underactuated-robotics/hopping_leg/model/with_rails/urdf/v7_railfixed.urdf" 
path_to_submechamisms = os.environ["AUTOPROJ_CURRENT_ROOT"] + "/../underactuated-robotics/hopping_leg/model/with_rails/urdf/submechanisms.yml"
print("path_to_urdf", path_to_urdf)
print("path_to_submechamisms", path_to_submechamisms)
'''
# Load the robot model in HyRoDyn
#robot = hyrodyn.RobotModel(path_to_urdf, path_to_submechamisms)

# Create Hopper multi-body plant
L1 = 0.205
L2 = 0.2
g = -9.81
plant = HopperPlant(link_length=[L1, L2], gravity=g, torque_limits = [18,12])

# Motor ID
motor_shoulder_id = 0x05
motor_elbow_id = 0x01

# CAN port
can_port = 'can0'

if len(sys.argv) != 2:
    print('Provide CAN device name (can0, slcan0 etc.)')
    sys.exit(0)

print("Using Socket {} for can communucation".format(sys.argv[1],))

# Create motor controller objects
#motor_shoulder_controller = CanMotorController(sys.argv[1], motor_shoulder_id)
#motor_elbow_controller = CanMotorController(sys.argv[1], motor_elbow_id)
motor_shoulder_controller = CanMotorController(can_socket='can0', motor_id=motor_shoulder_id, motor_type='AK80_9_V2', socket_timeout=0.5)
motor_elbow_controller = CanMotorController(can_socket='can0', motor_id=motor_elbow_id, motor_type='AK80_6_V2', socket_timeout=0.5)

print("Enabling Motors..")

shoulder_pos, shoulder_vel, shoulder_torque = motor_shoulder_controller.enable_motor()

print("Shoulder Motor Status: Pos: {}, Vel: {}, Torque: {}".format(shoulder_pos, shoulder_vel, shoulder_torque))

elbow_pos, elbow_vel, elbow_torque = motor_elbow_controller.enable_motor()

print("Elbow Motor Status: Pos: {}, Vel: {}, Torque: {}".format(elbow_pos, elbow_vel, elbow_torque))

#print("Setting Shoulder Motor to Zero Position...")

#setZeroPosition(motor_shoulder_controller, shoulder_pos)

#print("Setting Elbow Motor to Zero Position...")

#setZeroPosition(motor_elbow_controller, elbow_pos)

print("Start")
numSteps = 10000

time_vec = np.zeros(numSteps)

shoulder_position = np.zeros(numSteps)
elbow_position = np.zeros(numSteps)

shoulder_velocity = np.zeros(numSteps)
elbow_velocity = np.zeros(numSteps)

shoulder_torque = np.zeros(numSteps)
elbow_torque = np.zeros(numSteps)

desired_shoulder_torque = np.zeros(numSteps)
desired_elbow_torque = np.zeros(numSteps)
tau = np.zeros(2)

start_time = time.time()

shoulder_pos = 0.0
elbow_pos = 0.0
shoulder_vel = 0.0
elbow_vel = 0.0

for i in range(numSteps):

    dt = time.time()
    traj_time = dt - start_time
    
    time_vec[i] = traj_time
    # Send pos, vel and tau_ff command and use the in-built low level controller (improves a bit)
    '''
    shoulder_pos, shoulder_vel, shoulder_tau = motor_shoulder_controller.send_rad_command(shoulder_pos,shoulder_vel,5.0,1.0,tau[0])
    elbow_pos, elbow_vel, elbow_tau = motor_elbow_controller.send_rad_command(elbow_pos,elbow_vel,5.0,1.0,tau[1])
    '''
    # Send only the tau_ff command and use the in-built low level controller
    shoulder_pos, shoulder_vel, shoulder_tau = motor_shoulder_controller.send_rad_command(0.0,0.0,0.0,0.0,tau[0])
    elbow_pos, elbow_vel, elbow_tau = motor_elbow_controller.send_rad_command(0.0,0.0,0.0,0.0,tau[1])
    
    # Compute gravity torques using HyRoDyn
    #robot.y = np.array([shoulder_pos, elbow_pos])
    #robot.calculate_inverse_dynamics()
    #tau = robot.Tau_actuated.flatten()
    #print("Inverse dynamics output: ", tau)
    tau_including_passive_dof = plant.gravity_vector(shoulder_pos, elbow_pos)
    tau = tau_including_passive_dof.flatten()[1:]

    # Store data in lists
    shoulder_position[i] = shoulder_pos
    shoulder_velocity[i] = shoulder_vel
    shoulder_torque[i] = shoulder_tau

    elbow_position[i] = elbow_pos
    elbow_velocity[i] = elbow_vel
    elbow_torque[i] = elbow_tau
    
    desired_shoulder_torque[i] = tau[0]
    desired_elbow_torque[i] = tau[1]

print("Disabling Motors...")

shoulder_pos, shoulder_vel, shoulder_tau = motor_shoulder_controller.disable_motor()

print("Shoulder Motor Status: Pos: {}, Vel: {}, Torque: {}".format(shoulder_pos, shoulder_vel, shoulder_tau))

elbow_pos, elbow_vel, elbow_tau = motor_elbow_controller.disable_motor()

print("Elbow Motor Status: Pos: {}, Vel: {}, Torque: {}".format(elbow_pos, elbow_vel, elbow_tau))

plt.figure
plt.plot(time_vec, shoulder_position)
plt.plot(time_vec, elbow_position)
plt.xlabel("Time (s)")
plt.ylabel("Position (rad)")
plt.title("Position (rad) vs Time (s)")
plt.legend(['Shoulder', 'Elbow'])
plt.show()

plt.figure
plt.plot(time_vec, shoulder_velocity)
plt.plot(time_vec, elbow_velocity)
plt.xlabel("Time (s)")
plt.ylabel("Velocity (rad/s)")
plt.legend(['Shoulder', 'Elbow'])
plt.title("Velocity (rad/s) vs Time (s)")
plt.show()

plt.figure
plt.plot(time_vec, shoulder_torque)
plt.plot(time_vec, elbow_torque)
plt.plot(time_vec, desired_shoulder_torque)
plt.plot(time_vec, desired_elbow_torque)
plt.xlabel("Time (s)")
plt.ylabel("Torque (Nm)")
plt.title("Torque (Nm) vs Time (s)")
plt.legend(['Measured Shoulder', 'Measured Elbow', 'Desired Shoulder', 'Desired Elbow'])
plt.show()

def running_mean(x, N):
    cumsum = np.cumsum(np.insert(x, 0, 0))
    return (cumsum[N:] - cumsum[:-N]) / float(N)


filtered_shoulder_torque = running_mean(np.array(shoulder_torque), 10)
filtered_elbow_torque = running_mean(np.array(elbow_torque), 10)
time_vec_filtered = running_mean(np.array(time_vec), 10)

filtered_desired_shoulder_torque = running_mean(np.array(desired_shoulder_torque), 10)
filtered_desired_elbow_torque = running_mean(np.array(desired_elbow_torque), 10)

plt.figure
plt.plot(time_vec_filtered, filtered_shoulder_torque)
plt.plot(time_vec_filtered, filtered_elbow_torque)
plt.plot(time_vec_filtered, filtered_desired_shoulder_torque)
plt.plot(time_vec_filtered, filtered_desired_elbow_torque)
plt.xlabel("Time (s)")
plt.ylabel("Torque (Nm)")
plt.title("Filtered Torque (Nm) vs Time (s) with moving average filter (window = 100)")
plt.legend(['Measured Shoulder', 'Measured Elbow', 'Desired Shoulder', 'Desired Elbow'])
plt.show()


measured_csv_data = np.array([np.array(time_vec),
                    np.array(shoulder_position),
                    np.array(shoulder_velocity),
                    np.array(shoulder_torque),
                    np.array(elbow_position),
                    np.array(elbow_velocity),
                    np.array(elbow_torque)]).T
np.savetxt("measured_data.csv", measured_csv_data, delimiter=',', header="time,shoulder_pos,shoulder_vel,shoulder_torque,elbow_pos,elbow_vel,elbow_torque", comments="")
