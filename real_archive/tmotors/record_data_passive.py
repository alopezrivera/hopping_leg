import sys
import time
#from motor_driver.canmotorlib import CanMotorController
from canmotorlib import CanMotorController
import matplotlib.pyplot as plt
import math
import numpy as np
import pandas as pd

def setZeroPosition(motor, initPos):

    pos = initPos

    while abs(np.rad2deg(pos)) > 0.5:
        pos, vel, curr = motor.set_zero_position()
        print("Position: {}, Velocity: {}, Torque: {}".format(np.rad2deg(pos), np.rad2deg(vel),
                                                                curr))


# Motor ID
motor_shoulder_id = 0x05
motor_elbow_id = 0x01

# CAN port
can_port = 'can0'

# Create motor controller objects
#motor_shoulder_controller = CanMotorController(can_port, motor_shoulder_id)
#motor_elbow_controller = CanMotorController(can_port, motor_elbow_id)

motor_shoulder_controller = CanMotorController(can_socket='can0', motor_id=motor_shoulder_id, motor_type='AK80_9_V2', socket_timeout=0.5)
motor_elbow_controller = CanMotorController(can_socket='can0', motor_id=motor_elbow_id, motor_type='AK80_6_V2', socket_timeout=0.5)

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

    #time.sleep(0.002)
    


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
