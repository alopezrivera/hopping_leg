import sys
import time
import numpy as np
import matplotlib.pyplot as plt
from motor_driver.canmotorlib import CanMotorController

sys.path.append("../hoppping_leg")

from hopping_leg.utilities.cli import printu
from hopping_leg.utilities.real import motors, enable, zero


# Motor objects
motor_IDs = 0x02, 0x04

motor_shoulder, motor_elbow = motors(motor_IDs, sys.argv)

# Enable and Zero
printu("Enabling Motors")

shoulder_pos, shoulder_vel, shoulder_tau = enable(motor_shoulder, "Shoulder")

elbow_pos, elbow_vel, elbow_tau = enable(motor_elbow, "Elbow")

printu("Zeroing Motors")

zero(motor_shoulder, shoulder_pos)

zero(motor_elbow, elbow_pos)

# Actuation
printu("Actuation")

print("Seting joint angles and stiffness...")

shoulder_pos, shoulder_vel, shoulder_tau = motor_shoulder.send_rad_command(0.72425, 0, 15, 2, 0)

elbow_pos, elbow_vel, elbow_tau = motor_elbow.send_rad_command(-1.438, 0, 15, 2, 0)

print("Sleeping for 30 Seconds....")
time.sleep(30)

print("Starting recording data")
numSteps = 20000

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
    shoulder_pos, shoulder_vel, shoulder_tau = motor_shoulder.send_rad_command(0.72425, 0, 50,
                                                                                    2, 0)

    elbow_pos, elbow_vel, elbow_tau = motor_elbow.send_rad_command(-1.438, 0, 50, 2, 0)

    # Store data in lists
    shoulder_position[i] = shoulder_pos
    shoulder_velocity[i] = shoulder_vel
    shoulder_torque[i] = shoulder_tau

    elbow_position[i] = elbow_pos
    elbow_velocity[i] = elbow_vel
    elbow_torque[i] = elbow_tau


print("Go back to clamp!!!")

time.sleep(30)


print("Disabling Motors...")

shoulder_pos, shoulder_vel, shoulder_tau = motor_shoulder.disable_motor()

print("Shoulder Motor Status: Pos: {}, Vel: {}, Torque: {}".format(shoulder_pos, shoulder_vel, shoulder_tau))

elbow_pos, elbow_vel, elbow_tau = motor_elbow.disable_motor()

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
