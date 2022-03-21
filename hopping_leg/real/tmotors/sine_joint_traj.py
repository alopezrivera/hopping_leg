import time
from motor_driver.canmotorlib import CanMotorController
import matplotlib.pyplot as plt
import numpy as np

def setZeroPosition(motor, initPos):

    pos = initPos

    while abs(np.rad2deg(pos)) > 0.5:
        pos, vel, curr = motor.set_zero_position()
        print("Position: {}, Velocity: {}, Torque: {}".format(np.rad2deg(pos), np.rad2deg(vel),
                                                                curr))


# Motor ID
motor_shoulder_id = 0x01
motor_elbow_id = 0x04

# CAN port
can_port = 'can0'


# Task space trajectory
A = np.pi / 3.              # Amplitude
T = 5.                     # time period
omega = 2 * np.pi / T       # Angular Velocity

# Joint space gains
Kp_shoulder = 15.
Kd_shoulder = 0.5
Kp_elbow = 50.
Kd_elbow = 2.

# Create motor controller objects
motor_shoulder_controller = CanMotorController(can_port, motor_shoulder_id)
# motor_elbow_controller = CanMotorController(can_port, motor_elbow_id)

print("Enabling Motors..")

shoulder_pos, shoulder_vel, shoulder_torque = motor_shoulder_controller.enable_motor()

print("Shoulder Motor Status: Pos: {}, Vel: {}, Torque: {}".format(shoulder_pos, shoulder_vel,
                                                                    shoulder_torque))

# elbow_pos, elbow_vel, elbow_torque = motor_elbow_controller.enable_motor()

# print("Elbow Motor Status: Pos: {}, Vel: {}, Torque: {}".format(elbow_pos, elbow_vel, elbow_torque))

print("Setting Shoulder Motor to Zero Position...")

setZeroPosition(motor_shoulder_controller, shoulder_pos)

# print("Setting Elbow Motor to Zero Position...")

# setZeroPosition(motor_elbow_controller, elbow_pos)

# Initialize Arrays

time_vec_des = np.linspace(0, 5, 5000)
numSteps = len(time_vec_des)

time_vec = np.zeros(numSteps)

shoulder_position = np.zeros(numSteps)
elbow_position = np.zeros(numSteps)

shoulder_velocity = np.zeros(numSteps)
elbow_velocity = np.zeros(numSteps)

shoulder_torque = np.zeros(numSteps)
elbow_torque = np.zeros(numSteps)

shoulder_position_des = np.zeros(numSteps)
elbow_position_des = np.zeros(numSteps)

shoulder_velocity_des = np.zeros(numSteps)
elbow_velocity_des = np.zeros(numSteps)

start_time = time.time()

for i in range(numSteps):

    dt = time.time()
    traj_time = dt - start_time

    # Compute IK
    desired_joint_pos = A * np.sin(omega * traj_time)
    desired_joint_vel = omega * A * np.cos(omega * traj_time)
    shoulder_position_des[i] = desired_joint_pos
    # elbow_position_des[i] = elbow_pos_des
    shoulder_velocity_des[i] = desired_joint_vel
    # elbow_velocity_des[i] = elbow_vel_des

    time_vec[i] = traj_time
    # Send pos, vel and tau_ff command and use the in-built low level controller
    # shoulder_pos, shoulder_vel, shoulder_tau = motor_shoulder_controller.send_rad_command(0,0,0,0,0.0)
    # elbow_pos, elbow_vel, elbow_tau = motor_elbow_controller.send_rad_command(0,0,0,0,0.0)
    shoulder_pos, shoulder_vel, shoulder_tau = motor_shoulder_controller.send_rad_command(desired_joint_pos, desired_joint_vel, Kp_shoulder, Kd_shoulder, 0.0)


    # shoulder_pos, shoulder_vel, shoulder_tau = motor_elbow_controller.send_rad_command(desired_joint_pos, desired_joint_vel, Kp_shoulder, Kd_shoulder, 0.0)


    # elbow_pos, elbow_vel, elbow_tau = motor_elbow_controller.send_rad_command(elbow_pos_des,elbow_vel_des,Kp_elbow,Kd_elbow,0.0)

    # Store data in lists
    shoulder_position[i] = shoulder_pos
    shoulder_velocity[i] = shoulder_vel
    shoulder_torque[i] = shoulder_tau

    # elbow_position[i] = elbow_pos
    # elbow_velocity[i] = elbow_vel
    # elbow_torque[i] = elbow_tau

    # while(time.time() - dt < 0.002):
    #     pass


print("Disabling Motors...")

shoulder_pos, shoulder_vel, shoulder_tau = motor_shoulder_controller.disable_motor()

print("Shoulder Motor Status: Pos: {}, Vel: {}, Torque: {}".format(shoulder_pos, shoulder_vel,
                                                                    shoulder_tau))

# elbow_pos, elbow_vel, elbow_tau = motor_elbow_controller.disable_motor()

# print("Elbow Motor Status: Pos: {}, Vel: {}, Torque: {}".format(elbow_pos, elbow_vel, elbow_tau))

plt.figure
plt.plot(time_vec, shoulder_position)
plt.plot(time_vec, shoulder_position_des)
plt.xlabel("Time (s)")
plt.ylabel("Position (rad)")
plt.title("Position (rad) vs Time (s)")
plt.legend(['Shoulder Measured', 'Shoulder Desired'])
plt.show()

plt.figure
plt.plot(time_vec, shoulder_velocity)
plt.plot(time_vec, shoulder_velocity_des)
plt.xlabel("Time (s)")
plt.ylabel("Velocity (rad/s)")
plt.legend(['Shoulder Measured', 'Shoulder Desired'])
plt.title("Velocity (rad/s) vs Time (s)")
plt.show()

plt.figure
plt.plot(time_vec, shoulder_torque)
plt.xlabel("Time (s)")
plt.ylabel("Torque (Nm)")
plt.title("Torque (Nm) vs Time (s)")
plt.legend(['Shoulder'])
plt.show()


def running_mean(x, N):
    cumsum = np.cumsum(np.insert(x, 0, 0))
    return (cumsum[N:] - cumsum[:-N]) / float(N)


filtered_shoulder_torque = running_mean(np.array(shoulder_torque), 10)
time_vec_filtered = running_mean(np.array(time_vec), 10)

plt.figure
plt.plot(time_vec_filtered, filtered_shoulder_torque)
plt.xlabel("Time (s)")
plt.ylabel("Torque (Nm)")
plt.title("Filtered Torque (Nm) vs Time (s) with moving average filter (window = 100)")
plt.legend(['Shoulder'])
plt.show()
