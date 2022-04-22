import sys
import time
from motor_driver.canmotorlib import CanMotorController
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

# Motor gains
Kp_shoulder = 50.0
Kd_shoulder = 2.0
Kp_elbow = 50.0
Kd_elbow = 2.0

# Motor ID
motor_shoulder_id = 0x02
motor_elbow_id = 0x04

# CAN port
can_port = 'can0'

if len(sys.argv) != 2:
    print('Provide CAN device name (can0, slcan0 etc.)')
    sys.exit(0)

print("Using Socket {} for can communucation".format(sys.argv[1],))

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


# Read the trajectory file for swing up control
data = pd.read_csv("measured_data.csv")

numSteps = len(data)

total_num_steps = numSteps 

time_vec = np.zeros(total_num_steps)

shoulder_position = np.zeros(total_num_steps)
elbow_position = np.zeros(total_num_steps)

shoulder_velocity = np.zeros(total_num_steps)
elbow_velocity = np.zeros(total_num_steps)

shoulder_torque = np.zeros(total_num_steps)
elbow_torque = np.zeros(total_num_steps)

# New better method: Precompute Trajectory
print("Reading Trajectory from File...")
des_time_vec = data["time"] 
shoulder_pos_traj = data["shoulder_pos"]
shoulder_vel_traj = data["shoulder_vel"]
shoulder_tau_traj = data["shoulder_torque"]
elbow_pos_traj = data["elbow_pos"]
elbow_vel_traj = data["elbow_vel"]
elbow_tau_traj = data["elbow_torque"]

shoulder_vel_traj = np.clip(shoulder_vel_traj, -40.0,40.0)
elbow_vel_traj = np.clip(elbow_vel_traj, -40.0,40.0)

shoulder_tau_traj = np.clip(shoulder_tau_traj, -12.0,12.0)
elbow_tau_traj = np.clip(elbow_tau_traj, -12.0,12.0)

# Initial setpoint
shoulder_pos, shoulder_vel, shoulder_tau = motor_shoulder_controller.send_rad_command(shoulder_pos_traj[0], shoulder_vel_traj[0], Kp_shoulder, Kd_shoulder, 0.0)
elbow_pos, elbow_vel, elbow_tau = motor_elbow_controller.send_rad_command(elbow_pos_traj[0], elbow_vel_traj[0], Kp_elbow, Kd_elbow, 0.0)
input("Enter any key to continue (Ctrl+D to exit): ")

dt = data["time"][1] - data["time"][0]
print("Sending Trajectories to Motors... ")

try:
    print("Start")
    t = 0.0
    realStartT = time.time()
    for i in range(numSteps):
        stepStartT = time.time()
        time_vec[i] = t
        # Send pos, vel and tau_ff command and use the in-built low level controller
        #shoulder_pos, shoulder_vel, shoulder_tau = motor_shoulder_controller.send_rad_command(shoulder_pos_traj[i], shoulder_vel_traj[i], Kp_shoulder, Kd_shoulder, 0.0)
        #elbow_pos, elbow_vel, elbow_tau = motor_elbow_controller.send_rad_command(elbow_pos_traj[i], elbow_vel_traj[i], Kp_elbow, Kd_elbow, 0.0)
        shoulder_pos, shoulder_vel, shoulder_tau = motor_shoulder_controller.send_rad_command(shoulder_pos_traj[i], shoulder_vel_traj[i], Kp_shoulder, Kd_shoulder, shoulder_tau_traj[i])
        elbow_pos, elbow_vel, elbow_tau = motor_elbow_controller.send_rad_command(elbow_pos_traj[i], elbow_vel_traj[i], Kp_elbow, Kd_elbow, elbow_tau_traj[i])

        # Store data in lists
        shoulder_position[i] = shoulder_pos
        shoulder_velocity[i] = shoulder_vel
        shoulder_torque[i] = shoulder_tau

        elbow_position[i] = elbow_pos
        elbow_velocity[i] = elbow_vel
        elbow_torque[i] = elbow_tau

        t = t + dt

        elapsedTime = time.time() - stepStartT

        if (elapsedTime > dt):
            print("Loop Index {} takes longer than expected dt of {}.".format(i, dt))

        while (time.time() - stepStartT < dt):
            pass

    realEndT = time.time()
    realdT = (realEndT - realStartT) / numSteps

    print("End. New dt: {}".format(realdT))
    print("Disabling Motors...")

    shoulder_pos, shoulder_vel, shoulder_tau = motor_shoulder_controller.disable_motor()

    print("Shoulder Motor Status: Pos: {}, Vel: {}, Torque: {}".format(shoulder_pos, shoulder_vel, shoulder_tau))

    elbow_pos, elbow_vel, elbow_tau = motor_elbow_controller.disable_motor()

    print("Elbow Motor Status: Pos: {}, Vel: {}, Torque: {}".format(elbow_pos, elbow_vel, elbow_tau))

except Exception as e:
    print(e)

    print("Disabling Motors...")
    shoulder_pos, shoulder_vel, shoulder_tau = motor_shoulder_controller.disable_motor()
    print("Shoulder Motor Status: Pos: {}, Vel: {}, Torque: {}".format(shoulder_pos, shoulder_vel, shoulder_tau))

    elbow_pos, elbow_vel, elbow_tau = motor_elbow_controller.disable_motor()
    print("Elbow Motor Status: Pos: {}, Vel: {}, Torque: {}".format(elbow_pos, elbow_vel, elbow_tau))

plt.figure
plt.plot(time_vec, shoulder_position)
plt.plot(time_vec, elbow_position)
plt.plot(des_time_vec, shoulder_pos_traj)
plt.plot(des_time_vec, elbow_pos_traj)
plt.xlabel("Time (s)")
plt.ylabel("Position (rad)")
plt.title("Position (rad) vs Time (s)")
plt.legend(['Shoulder Measured', 'Elbow Measured', 'Shoulder Desired', 'Elbow Desired'])
plt.show()

plt.figure
plt.plot(time_vec, shoulder_velocity)
plt.plot(time_vec, elbow_velocity)
plt.plot(des_time_vec, shoulder_vel_traj)
plt.plot(des_time_vec, elbow_vel_traj)
plt.xlabel("Time (s)")
plt.ylabel("Velocity (rad/s)")
plt.legend(['Shoulder Measured', 'Elbow Measured', 'Shoulder Desired', 'Elbow Desired'])
plt.title("Velocity (rad/s) vs Time (s)")
plt.show()

plt.figure
plt.plot(time_vec, shoulder_torque)
plt.plot(time_vec, elbow_torque)
plt.plot(des_time_vec, shoulder_tau_traj)
plt.plot(des_time_vec, elbow_tau_traj)
plt.xlabel("Time (s)")
plt.ylabel("Torque (Nm)")
plt.title("Torque (Nm) vs Time (s)")
plt.legend(['Shoulder Measured', 'Elbow Measured', 'Shoulder Desired', 'Elbow Desired'])
plt.show()


def running_mean(x, N):
    cumsum = np.cumsum(np.insert(x, 0, 0))
    return (cumsum[N:] - cumsum[:-N]) / float(N)


filtered_shoulder_torque = running_mean(np.array(shoulder_torque), 10)
filtered_elbow_torque = running_mean(np.array(elbow_torque), 10)
filtered_shoulder_torque_traj = running_mean(np.array(shoulder_tau_traj), 10)
filtered_elbow_torque_traj = running_mean(np.array(elbow_tau_traj), 10)

time_vec_filtered = running_mean(np.array(time_vec), 10)
des_time_vec_filtered = running_mean(np.array(des_time_vec), 10)

plt.figure
plt.plot(time_vec_filtered, filtered_shoulder_torque)
plt.plot(time_vec_filtered, filtered_elbow_torque)
plt.plot(des_time_vec_filtered, filtered_shoulder_torque_traj)
plt.plot(des_time_vec_filtered, filtered_elbow_torque_traj)

plt.xlabel("Time (s)")
plt.ylabel("Torque (Nm)")
plt.title("Filtered Torque (Nm) vs Time (s) with moving average filter (window = 100)")
plt.legend(['Shoulder Measured', 'Elbow Measured', 'Shoulder Desired', 'Elbow Desired'])
plt.show()

'''
measured_csv_data = np.array([np.array(time_vec),
                            np.array(shoulder_position),
                            np.array(shoulder_velocity),
                            np.array(shoulder_torque),
                            np.array(elbow_position),
                            np.array(elbow_velocity),
                            np.array(elbow_torque)]).T
np.savetxt("measured_data.csv", measured_csv_data, delimiter=',', header="time,shoulder_pos,shoulder_vel,shoulder_torque,elbow_pos,elbow_vel,elbow_torque", comments="")
'''
