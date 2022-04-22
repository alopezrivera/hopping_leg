import sys
import time
from canmotorlib import CanMotorController
#from motor_driver.canmotorlib import CanMotorController
import matplotlib.pyplot as plt
import math
import numpy as np
import pandas as pd
sys.path.append("../../")
from hopper_plant.hopper_plant import HopperPlant

def estimate_acceleration(meas_vel, t):
    from filters import butterworth as butter
    # get acceleration from filtered velocities
    vel_butter = butter.data_filter(meas_vel, order=3, cutoff=0.2)
    acc_vel_grad_butter = np.gradient(vel_butter, t)
    acc_grad_2butter = butter.data_filter(acc_vel_grad_butter, order=3, # 3
                                          cutoff=0.1)
    return acc_grad_2butter

def setZeroPosition(motor, initPos):

    pos = initPos

    while abs(np.rad2deg(pos)) > 0.5:
        pos, vel, curr = motor.set_zero_position()
        print("Position: {}, Velocity: {}, Torque: {}".format(np.rad2deg(pos), np.rad2deg(vel),
                                                                curr))
def mean_absolute_error(x,y):
    return np.average(np.absolute(x - y))

# Motor ID
motor_shoulder_id = 0x07
motor_elbow_id = 0x06

# CAN port
can_port = 'can0'

if len(sys.argv) != 2:
    print('Provide CAN device name (can0, slcan0 etc.)')
    sys.exit(0)

print("Using Socket {} for can communucation".format(sys.argv[1],))

# Create Hopper multi-body plant
L1 = 0.205
L2 = 0.2
g = -9.81
plant = HopperPlant(link_length=[L1, L2], gravity=g)

# Task space trajectory
h = 0.2    # vertical height
A = h / 2.    # Amplitude
offset = L1 + L2 - A - 0.05   # offset, 0.1 is the safety margin from the singularity
T = 0.25  # time period
omega = 2 * np.pi / T

# Joint space gains
Kp_shoulder = 50.0
Kd_shoulder = 2.0
Kp_elbow = 50.0
Kd_elbow = 2.0

# Create motor controller objects
#motor_shoulder_controller = CanMotorController(sys.argv[1], motor_shoulder_id)
#motor_elbow_controller = CanMotorController(sys.argv[1], motor_elbow_id)

motor_shoulder_controller = CanMotorController(can_socket='can0', motor_id=motor_shoulder_id, motor_type='AK80_6_V2', socket_timeout=0.5)
motor_elbow_controller = CanMotorController(can_socket='can0', motor_id=motor_elbow_id, motor_type='AK80_6_V2', socket_timeout=0.5)

print("Enabling Motors..")

shoulder_pos, shoulder_vel, shoulder_torque = motor_shoulder_controller.enable_motor()

print("Shoulder Motor Status: Pos: {}, Vel: {}, Torque: {}".format(shoulder_pos, shoulder_vel,
                                                                    shoulder_torque))

elbow_pos, elbow_vel, elbow_torque = motor_elbow_controller.enable_motor()

print("Elbow Motor Status: Pos: {}, Vel: {}, Torque: {}".format(elbow_pos, elbow_vel, elbow_torque))

print("Setting Shoulder Motor to Zero Position...")

setZeroPosition(motor_shoulder_controller, shoulder_pos)

print("Setting Elbow Motor to Zero Position...")

setZeroPosition(motor_elbow_controller, elbow_pos)

print("Start")


timeToInit = 1
numStepsToInit = 800
numSteps = 4000 + numStepsToInit

# time_vec_des = np.linspace(0, 15, 10000)
# numSteps = len(time_vec_des) + numStepsToInit

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

shoulder_position_des = np.zeros(numSteps)
elbow_position_des = np.zeros(numSteps)

shoulder_velocity_des = np.zeros(numSteps)
elbow_velocity_des = np.zeros(numSteps)

shoulder_acceleration_des = np.zeros(numSteps)
elbow_acceleration_des = np.zeros(numSteps)

shoulder_torque_des = np.zeros(numSteps)
elbow_torque_des = np.zeros(numSteps)

# Desired Cartesian Coordinates

foot_x = np.zeros(numSteps)
foot_y = np.zeros(numSteps)
foot_xd = np.zeros(numSteps)
foot_yd = np.zeros(numSteps)
foot_xdd = np.zeros(numSteps)
foot_ydd = np.zeros(numSteps)

# Measured Cartesian coordinates
foot_x_meas = np.zeros(numSteps)
foot_y_meas = np.zeros(numSteps)
foot_xd_meas = np.zeros(numSteps)
foot_yd_meas = np.zeros(numSteps)
foot_xdd_meas = np.zeros(numSteps)
foot_ydd_meas = np.zeros(numSteps)

# Set up initial position
x_des = offset + A
shoulder_pos_des_init, elbow_pos_des_init = plant.inverse_kinematics(x_des, 0, knee_direction=1)
print("shoulder_pos_des_init: ", shoulder_pos_des_init, " elbow_pos_des_init:", elbow_pos_des_init)

def cycloidal_trajectory(t, theta_final, T, theta_init=0.0):
    # Cycloidal trajectory
    omega = 2*math.pi/T
    ratio = (theta_final - theta_init)/T
    pos = theta_init + ratio * ( t - math.sin( omega * t ) / omega)
    vel = ratio * (1.0 - math.cos(omega * t))
    acc = ratio * omega * math.sin(omega * t)
    return pos, vel, acc


start_time = time.time()

for i in range(numStepsToInit):
    dt = time.time()
    traj_time = dt - start_time

    shoulder_pos_des, shoulder_vel_des, shoulder_acc_des = cycloidal_trajectory(traj_time, shoulder_pos_des_init,
                                                                timeToInit)
    elbow_pos_des, elbow_vel_des, elbow_acc_des = cycloidal_trajectory(traj_time, elbow_pos_des_init, timeToInit)

    # Compute Inverse Dynamics 
    tau = plant.inverse_dynamics(shoulder_pos_des, elbow_pos_des, shoulder_vel_des, elbow_vel_des, 0.0, shoulder_acc_des, elbow_acc_des)
    tau = plant.gravity_vector(shoulder_pos_des, elbow_pos_des)
    shoulder_tau_des = tau[1]
    elbow_tau_des = tau[2]

    shoulder_pos, shoulder_vel, shoulder_tau = motor_shoulder_controller.send_rad_command(shoulder_pos_des, shoulder_vel_des, Kp_shoulder, Kd_shoulder, shoulder_tau_des)
    elbow_pos, elbow_vel, elbow_tau = motor_elbow_controller.send_rad_command(elbow_pos_des, elbow_vel_des, Kp_elbow, Kd_elbow, elbow_tau_des)

    shoulder_position_des[i] = shoulder_pos_des
    elbow_position_des[i] = elbow_pos_des
    shoulder_velocity_des[i] = shoulder_vel_des
    elbow_velocity_des[i] = elbow_vel_des
    shoulder_acceleration_des[i] = shoulder_acc_des
    elbow_acceleration_des[i] = elbow_acc_des
    shoulder_torque_des[i] = shoulder_tau_des
    elbow_torque_des[i] = elbow_tau_des

    time_vec[i] = traj_time

    # Store data in lists
    shoulder_position[i] = shoulder_pos
    shoulder_velocity[i] = shoulder_vel
    shoulder_torque[i] = shoulder_tau
    elbow_position[i] = elbow_pos
    elbow_velocity[i] = elbow_vel
    elbow_torque[i] = elbow_tau

    foot_x[i] = 0
    foot_y[i] = 0
    foot_xd[i] = 0
    foot_yd[i] = 0
    foot_xdd[i] = 0
    foot_ydd[i] = 0

print("Calculating FK offline for init phase...")
for i in range(numStepsToInit):
    # Compute desired FK
    x_des, y_des = plant.forward_kinematics(shoulder_position_des[i], elbow_position_des[i])
    xd_des, yd_des = plant.forward_velocity(shoulder_position_des[i], elbow_position_des[i], shoulder_velocity_des[i], elbow_velocity_des[i])
    xdd_des, ydd_des = plant.forward_acceleration(shoulder_position_des[i], elbow_position_des[i], shoulder_velocity_des[i], elbow_velocity_des[i], shoulder_acceleration_des[i], elbow_acceleration_des[i])

    foot_x[i] = x_des
    foot_y[i] = y_des
    foot_xd[i] = xd_des
    foot_yd[i] = yd_des    
    foot_xdd[i] = xdd_des
    foot_ydd[i] = ydd_des    

    # Compute measured FK
    x_meas, y_meas = plant.forward_kinematics(shoulder_position[i], elbow_position[i])
    xd_meas, yd_meas = plant.forward_velocity(shoulder_position[i], elbow_position[i], shoulder_velocity[i], elbow_velocity[i])
    foot_x_meas[i] = x_meas
    foot_y_meas[i] = y_meas
    foot_xd_meas[i] = xd_meas
    foot_yd_meas[i] = yd_meas    

print("Done initializing...")

val = input("Enter y or Y to start vertical trajectory: ")

# time.sleep(0.5)
if val == 'y' or val == 'Y':
    for i in range(numStepsToInit, numSteps):

        dt = time.time()
        traj_time = dt - start_time

        # Compute position IK
        x_des = A * np.cos(omega * (traj_time - time_vec[numStepsToInit])) + offset
        y_des = 0.0
        shoulder_pos_des, elbow_pos_des = plant.inverse_kinematics(x_des, y_des, knee_direction=1)
        # Compute velocity IK 
        xd_des = -omega * A * np.sin(omega * (traj_time - time_vec[numStepsToInit]))
        yd_des = 0.0
        shoulder_vel_des, elbow_vel_des = plant.inverse_velocity(shoulder_pos_des, elbow_pos_des,
                                                                xd_des, yd_des)
        # Compute acceleration IK 
        xdd_des = -omega * omega * A * np.cos(omega * (traj_time - time_vec[numStepsToInit]))
        ydd_des = 0.0
        shoulder_acc_des, elbow_acc_des = plant.inverse_acceleration(shoulder_pos_des, elbow_pos_des, shoulder_vel_des, elbow_vel_des, xdd_des, ydd_des)

        # Compute Inverse Dynamics 
        tau = plant.inverse_dynamics(shoulder_pos_des, elbow_pos_des, shoulder_vel_des, elbow_vel_des, 0.0, shoulder_acc_des, elbow_acc_des)
        tau = plant.gravity_vector(shoulder_pos_des, elbow_pos_des)
        shoulder_tau_des = tau[1]
        elbow_tau_des = tau[2]

        shoulder_position_des[i] = shoulder_pos_des
        elbow_position_des[i] = elbow_pos_des
        shoulder_velocity_des[i] = shoulder_vel_des
        elbow_velocity_des[i] = elbow_vel_des
        shoulder_acceleration_des[i] = shoulder_acc_des
        elbow_acceleration_des[i] = elbow_acc_des
        shoulder_torque_des[i] = shoulder_tau_des
        elbow_torque_des[i] = elbow_tau_des

        time_vec[i] = traj_time
        # Send pos, vel and tau_ff command and use the in-built low level controller
        shoulder_pos, shoulder_vel, shoulder_tau = motor_shoulder_controller.send_rad_command(shoulder_pos_des, shoulder_vel_des, Kp_shoulder, Kd_shoulder, shoulder_tau_des)
        elbow_pos, elbow_vel, elbow_tau = motor_elbow_controller.send_rad_command(elbow_pos_des, elbow_vel_des, Kp_elbow, Kd_elbow, elbow_tau_des)

        # Store data in lists
        shoulder_position[i] = shoulder_pos
        shoulder_velocity[i] = shoulder_vel
        shoulder_torque[i] = shoulder_tau

        elbow_position[i] = elbow_pos
        elbow_velocity[i] = elbow_vel
        elbow_torque[i] = elbow_tau

        foot_x[i] = x_des
        foot_y[i] = y_des
        foot_xd[i] = xd_des
        foot_yd[i] = yd_des
        foot_xdd[i] = xdd_des
        foot_ydd[i] = ydd_des

print("Disabling Motors...")

shoulder_pos, shoulder_vel, shoulder_tau = motor_shoulder_controller.disable_motor()

print("Shoulder Motor Status: Pos: {}, Vel: {}, Torque: {}".format(shoulder_pos, shoulder_vel, shoulder_tau))

elbow_pos, elbow_vel, elbow_tau = motor_elbow_controller.disable_motor()

print("Elbow Motor Status: Pos: {}, Vel: {}, Torque: {}".format(elbow_pos, elbow_vel, elbow_tau))

avg_dt = np.average(np.diff(time_vec))
print("Avg. dt", avg_dt)
# Compute measured acceleration numerically
'''
shoulder_velocity_num = np.gradient(shoulder_position, time_vec)
elbow_velocity_num = np.gradient(elbow_position, time_vec)
shoulder_acceleration = np.gradient(shoulder_velocity_num, time_vec)
elbow_acceleration = np.gradient(elbow_velocity_num, time_vec)
'''
shoulder_acceleration = estimate_acceleration(shoulder_velocity, time_vec)
elbow_acceleration = estimate_acceleration(elbow_velocity, time_vec)

print("Calculating FK offline...")
for i in range(numSteps):
    # Compute FK
    x_meas, y_meas = plant.forward_kinematics(shoulder_position[i], elbow_position[i])
    xd_meas, yd_meas = plant.forward_velocity(shoulder_position[i], elbow_position[i], shoulder_velocity[i], elbow_velocity[i])
    xdd_meas, ydd_meas = plant.forward_acceleration(shoulder_position[i], elbow_position[i], shoulder_velocity[i], elbow_velocity[i], shoulder_acceleration[i], elbow_acceleration[i])

    foot_x_meas[i] = x_meas
    foot_y_meas[i] = y_meas
    foot_xd_meas[i] = xd_meas
    foot_yd_meas[i] = yd_meas    
    foot_xdd_meas[i] = xdd_meas
    foot_ydd_meas[i] = ydd_meas    

plt.figure
plt.plot(time_vec, shoulder_position)
plt.plot(time_vec, elbow_position)
plt.plot(time_vec, shoulder_position_des)
plt.plot(time_vec, elbow_position_des)
plt.xlabel("Time (s)")
plt.ylabel("Position (rad)")
plt.title("Position (rad) vs Time (s)")
plt.legend(['Shoulder Measured', 'Elbow Measured', 'Shoulder Desired', 'Elbow Desired'])
plt.show()

plt.figure
plt.plot(time_vec, foot_x)
plt.plot(time_vec, foot_y)
plt.plot(time_vec, foot_x_meas)
plt.plot(time_vec, foot_y_meas)
plt.xlabel("Time (s)")
plt.ylabel("Cartesian Position (m)")
plt.title("Cartesian Position (m) vs Time (s)")
plt.legend(['Foot X Desired', 'Foot Y Desired', 'Foot X Measured', 'Foot Y Measured'])
plt.show()

plt.figure
plt.plot(foot_y, foot_x)
plt.plot(foot_y_meas, foot_x_meas)
plt.xlabel("Cartesian Position Y (m)")
plt.ylabel("Cartesian Position X (m)")
plt.legend(['Desired', 'Measured'])
plt.title("Cartesian Position X (m) vs Y (m)")
plt.axis("equal")
plt.gca().invert_yaxis()
plt.show()

plt.figure
plt.plot(time_vec, shoulder_velocity)
plt.plot(time_vec, elbow_velocity)
plt.plot(time_vec, shoulder_velocity_des)
plt.plot(time_vec, elbow_velocity_des)
plt.xlabel("Time (s)")
plt.ylabel("Velocity (rad/s)")
plt.legend(['Shoulder Measured', 'Elbow Measured', 'Shoulder Desired', 'Elbow Desired'])
plt.title("Velocity (rad/s) vs Time (s)")
plt.show()

plt.figure
plt.plot(time_vec, foot_xd)
plt.plot(time_vec, foot_yd)
plt.plot(time_vec, foot_xd_meas)
plt.plot(time_vec, foot_yd_meas)
plt.xlabel("Time (s)")
plt.ylabel("Cartesian Velocity (m/s)")
plt.title("Cartesian Velocity (m/s) vs Time (s)")
plt.legend(['Foot Xdot Desired', 'Foot Ydot Desired', 'Foot Xdot Measured', 'Foot Ydot Measured'])
plt.show()

plt.figure
plt.plot(time_vec, shoulder_acceleration)
plt.plot(time_vec, elbow_acceleration)
plt.plot(time_vec, shoulder_acceleration_des)
plt.plot(time_vec, elbow_acceleration_des)
plt.xlabel("Time (s)")
plt.ylabel("Acceleration (rad/s^2)")
plt.legend(['Shoulder Measured', 'Elbow Measured', 'Shoulder Desired', 'Elbow Desired'])
plt.title("Acceleration (rad/s^2) vs Time (s)")
plt.show()

plt.figure
plt.plot(time_vec, foot_xdd)
plt.plot(time_vec, foot_ydd)
plt.plot(time_vec, foot_xdd_meas)
plt.plot(time_vec, foot_ydd_meas)
plt.xlabel("Time (s)")
plt.ylabel("Cartesian Acceleration (m/s^2)")
plt.title("Cartesian Acceleration (m/s^2) vs Time (s)")
plt.legend(['Foot Xdot Desired', 'Foot Ydot Desired', 'Foot Xdot Measured', 'Foot Ydot Measured'])
plt.show()

plt.figure
plt.plot(time_vec, shoulder_torque)
plt.plot(time_vec, elbow_torque)
plt.plot(time_vec, shoulder_torque_des)
plt.plot(time_vec, elbow_torque_des)
plt.xlabel("Time (s)")
plt.ylabel("Torque (Nm)")
plt.title("Torque (Nm) vs Time (s)")
plt.legend(['Shoulder', 'Elbow', 'Desired Shoulder', 'Desired Elbow'])
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

desired_csv_data = np.array([np.array(time_vec),
                    np.array(shoulder_position_des),
                    np.array(shoulder_velocity_des),
                    np.array(shoulder_torque_des),
                    np.array(elbow_position_des),
                    np.array(elbow_velocity_des),
                    np.array(elbow_torque_des)]).T
np.savetxt("desired_data.csv", desired_csv_data, delimiter=',', header="time,shoulder_pos,shoulder_vel,shoulder_torque,elbow_pos,elbow_vel,elbow_torque", comments="")

print("MAE for Shoulder Position = ", mean_absolute_error(shoulder_position, shoulder_position_des), " rad")
print("MAE for Elbow Position = ", mean_absolute_error(elbow_position, elbow_position_des), " rad")
print("MAE for Shoulder Velocity = ", mean_absolute_error(shoulder_velocity, shoulder_velocity_des), " rad/s")
print("MAE for Elbow Velocity = ", mean_absolute_error(elbow_velocity, elbow_velocity_des), " rad/s")

