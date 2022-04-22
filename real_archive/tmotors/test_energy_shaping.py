"""
Energy Shaping
==============
"""

import sys
import time
import numpy as np
import matplotlib.pyplot as plt
from canmotorlib import CanMotorController

sys.path.append("../hopping_leg")
from hopping_leg.plant.hopper import HopperPlant
from hopping_leg.controllers.PD import PD_controller

from hopping_leg.utilities.cli import printu
from hopping_leg.utilities.real import motors, enable, zero, disable

from real.heuristics import contact_detection, estimate_state, get_control

tstart = time.time()
t0 = 0
v0 = 0
height_factor = 6.0
maxh = 0

flight_counter = 0

# Create Hopper multi-body plant
L1 = 0.205
L2 = 0.2
g = -9.81
plant = HopperPlant(link_length=[L1, L2], gravity=g, torque_limits = [6,6])

# Initialize Controller
controller = PD_controller(plant)
Kp_shoulder = 5.0
Kd_shoulder = 0.1
Kp_elbow = 5.0
Kd_elbow = 0.1
contact_force_threshold = 10

# Desired EE configuration
x_des = 0.2
y_des = 0.0

# Motor objects
motor_IDs = 0x07, 0x06

motor_shoulder, motor_elbow = motors(motor_IDs, sys.argv)

# Enable and Zero
printu("Enabling Motors")

_shoulder_position, _shoulder_velocity, _shoulder_torque = enable(motor_shoulder, "shoulder")

_elbow_position, _elbow_velocity, _elbow_torque = enable(motor_elbow, "elbow")

printu("Zeroing Motors")

zero(motor_shoulder, _shoulder_position)

zero(motor_elbow, _elbow_position)

# Set Joint Angles and Stiffness
print("Seting Joint Angles and Stiffness")

qdes                                     = plant.inverse_kinematics(x_des, y_des, knee_direction=1)
_shoulder_position, _shoulder_velocity, _shoulder_torque = motor_shoulder.send_rad_command(qdes[0], 0, 50, 2, 0)
_elbow_position, _elbow_velocity, _elbow_torque          = motor_elbow.send_rad_command(qdes[1], 0, 50, 2, 0)

#################################################
#                                               #
#              Array Initialization             #
#                                               #
#################################################
numSteps = 5000
initArr  = np.zeros(numSteps)
init     = lambda var: locals().update({var: initArr})

# Time
time_vec = initArr

# Phase
phase_vec = initArr

# Initialize motor state arrays
for motor in ['shoulder', 'elbow']:
    for var in ['position', 'velocity', 'torque', 'position_des', 'velocity_des', 'torque_des']:
        init(f"{motor}_{var}")

# Initialize foot Cartesian cordinate arrays
for var in ['x', 'y', 'xd', 'yd', 'x_mes', 'y_mes', 'xd_mes', 'yd_mes']:
    init(f"foot_{var}")

#################################################
#                                               #
#              Initial Conditions               #
#                                               #
#################################################

current_phase = "LIFTOFF"
desired_height = 0.4
liftoff_extension = 0.16

phase_dict = {'FLIGHT': 0, 'TOUCHDOWN': 1, 'LIFTOFF': 2, 'FLIGHT_ASCEND': 3}

# For the initial phase
_shoulder_position_des = qdes[0]
_elbow_position_des    = qdes[1]
_shoulder_velocity_des = 0.0
_elbow_velocity_des    = 0.0
_shoulder_torque_des   = 0.0
_elbow_torque_des      = 0.0

#################################################
#################################################
#################################################
#                                               #
#                  Control Loop                 #
#                                               #
#################################################
#################################################
#################################################

# Confirm start
val = input("Enter y or Y to start energy shaping test: ")

# Start time
start_time = time.time()

if val == 'y' or val == 'Y':
    print("Starting recording data")
    for i in range(numSteps):

        # Record elapsed time
        time_vec[i] = time.time() - start_time
        
        # Gain scheduling
        if current_phase == "LIFTOFF":
            Kp_shoulder = 1.0
            Kd_shoulder = 1.0
            Kp_elbow = 1.0
            Kd_elbow = 1.0
        elif current_phase == "TOUCHDOWN":
            Kp_shoulder = 5.0
            Kd_shoulder = 0.1
            Kp_elbow = 5.0
            Kd_elbow = 0.1
        else:
            Kp_shoulder = 50.0
            Kd_shoulder = 2.0
            Kp_elbow = 50.0
            Kd_elbow = 2.0

        # Motor controller commands --------------------------------------------------------------------------------------------------------------------------------------------------------------
        _shoulder_position, _shoulder_velocity, _shoulder_torque = motor_shoulder.send_rad_command(_shoulder_position_des, _shoulder_velocity_des, Kp_shoulder, Kd_shoulder, _shoulder_torque_des)
        _elbow_position, _elbow_velocity, _elbow_torque          = motor_elbow.send_rad_command(_elbow_position_des, _elbow_velocity_des, Kp_elbow, Kd_elbow, _elbow_torque_des)
        # ----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
        
        # Update phase
        phase_vec[i] = phase_dict[current_phase]

        # Update state
        state = [_shoulder_position, _elbow_position, _shoulder_velocity, _elbow_velocity]
        control = get_control(controller, current_phase, state, np.array([_shoulder_torque, _elbow_torque]), desired_height)
        _shoulder_position_des, _elbow_position_des, _shoulder_velocity_des, _elbow_velocity_des, _shoulder_torque_des, _elbow_torque_des, current_phase, flight_counter, maxh, height_factor, v0, t0 = control

        # Update height estimation
        height_est[i] = estimate_state(state,current_phase, t0,v0)
        height_factor_vec[i] = height_factor

        # Compute Inverse Dynamics for feed-forward torques in joint space
        #tau = plant.gravity_vector(_shoulder_position_des, _elbow_position_des)
        #_shoulder_torque_des = _shoulder_torque_des + tau[1]
        #_elbow_torque_des = _elbow_torque_des + tau[2]

        # Record state
        for motor in ['shoulder', 'elbow']:
            for var in ['position', 'velocity', 'torque', 'position_des', 'velocity_des', 'torque_des']:
                locals()[f"{motor}_{var}"][i] = locals()[f"_{motor}_{var}"]

#################################################
#################################################
#################################################
#                                               #
#              End of Control Loop              #
#                                               #
#################################################
#################################################
#################################################

printu("Motion finished")
qdes                                                     = plant.inverse_kinematics(x_des, y_des, knee_direction=1)
_shoulder_position, _shoulder_velocity, _shoulder_torque = motor_shoulder.send_rad_command(qdes[0], 0, 50, 2, 0)
_elbow_position, _elbow_velocity, _elbow_torque          = motor_elbow.send_rad_command(qdes[1], 0, 50, 2, 0)

time.sleep(5)

printu("Disabling Motors")

_shoulder_position, _shoulder_velocity, _shoulder_torque = disable(motor_shoulder, "shoulder")

_elbow_position, _elbow_velocity, _elbow_torque          = disable(motor_elbow, "elbow")

print("Number of Jumps:", flight_counter)

#################################################
#                                               #
#                Postprocessing                 #
#                                               #
#################################################
print("Calculating FK offline...")
for i in range(numSteps):
    # Compute FK for measured data
    x_meas, y_meas = plant.forward_kinematics(shoulder_position[i], elbow_position[i])
    xd_meas, yd_meas = plant.forward_velocity(shoulder_position[i], elbow_position[i], shoulder_velocity[i], elbow_velocity[i])

    foot_x_meas[i] = x_meas
    foot_y_meas[i] = y_meas
    foot_xd_meas[i] = xd_meas
    foot_yd_meas[i] = yd_meas 

    # Compute FK for desired data
    x_des, y_des = plant.forward_kinematics(shoulder_position_des[i], elbow_position_des[i])
    xd_des, yd_des = plant.forward_velocity(shoulder_position_des[i], elbow_position_des[i], shoulder_velocity_des[i], elbow_velocity_des[i])

    foot_x[i] = x_des
    foot_y[i] = y_des
    foot_xd[i] = xd_des
    foot_yd[i] = yd_des

print("Calculating filtered states...")
def running_mean(x, N):
        cumsum = np.cumsum(np.insert(x, 0, 0))
        return (cumsum[N:] - cumsum[:-N]) / float(N)

filtered_shoulder_torque     = running_mean(np.array(shoulder_torque), 10)
filtered_elbow_torque        = running_mean(np.array(elbow_torque), 10)
filtered_shoulder_torque_des = running_mean(np.array(shoulder_torque_des), 10)
filtered_elbow_torque_des    = running_mean(np.array(elbow_torque_des), 10)
time_vec_filtered            = running_mean(np.array(time_vec), 10)

#################################################
#                                               #
#                Visualization                  #
#                                               #
#################################################
get = lambda var: locals()[var]

# State -----------------------------------------
state_plot = []
for stp in ['position', 'velocity', 'torque']:
    state_plot += [get(stp), get(f"{stp}_des")]
# Cartesian parameters
for crp in ['x', 'y', 'xd', 'yd']:
    state_plot += [get(f"foot_{stp}"), get(f"foot_{crp}_mes")]

# Filtered torque -------------------------------
torque_plot = []
for motor in ['shoulder', 'elbow']:
    torque_plot += [get(f"filtered_{motor}_torque"), get(f"filtered_{motor}_torque_des")]

from mpl_plotter.two_d import panes
panes(time_vec,
      [shoulder_position, shoulder_position_des])
comparison(time_vec,
           [shoulder_position, elbow_position, shoulder_position_des, elbow_position_des])

# plt.figure
# plt.plot(time_vec, shoulder_position)
# plt.plot(time_vec, elbow_position)
# plt.plot(time_vec, shoulder_position_des)
# plt.plot(time_vec, elbow_position_des)
# #plt.plot(time_vec, phase_vec)
# plt.xlabel("Time (s)")
# plt.ylabel("Position (rad)")
# plt.title("Position (rad) vs Time (s)")
# plt.legend(['Shoulder', 'Elbow', 'Shoulder Desired', 'Elbow Desired'])
# ax = plt.gca()
# ax.pcolorfast((time_vec[0],time_vec[-1]),ax.get_ylim(),np.array(phase_vec)[np.newaxis],cmap="Greens", alpha=.3)
# plt.show()

plt.figure()
fig, ax = plt.subplots()
ax.plot(time_vec, height_est)
ax.plot(time_vec, desired_height*np.ones(len(time_vec)))
ax2 = ax.twinx()
ax2.plot(time_vec, height_factor_vec)
ax.set_xlabel("Time (s)")
ax.set_ylabel("Position x (m)")
ax2.set_ylabel("Height factor")
plt.legend(["Estimated height", "Desired height", "Height Factor"])
ax = plt.gca()
ax.pcolorfast((time_vec[0],time_vec[-1]),ax.get_ylim(),np.array(phase_vec)[np.newaxis],cmap="Greens", alpha=.3)
plt.show()


# plt.figure
# plt.plot(time_vec, foot_x)
# plt.plot(time_vec, foot_y)
# plt.plot(time_vec, foot_x_meas)
# plt.plot(time_vec, foot_y_meas)
# ax = plt.gca()
# ax.pcolorfast((time_vec[0],time_vec[-1]),ax.get_ylim(),np.array(phase_vec)[np.newaxis],cmap="Greens", alpha=.3)
# plt.xlabel("Time (s)")
# plt.ylabel("Cartesian Position (m)")
# plt.title("Cartesian Position (m) vs Time (s)")
# plt.legend(['Foot X Desired', 'Foot Y Desired', 'Foot X Measured', 'Foot Y Measured'])
# plt.show()

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

# plt.figure
# plt.plot(time_vec, foot_xd)
# plt.plot(time_vec, foot_yd)
# plt.plot(time_vec, foot_xd_meas)
# plt.plot(time_vec, foot_yd_meas)
# #plt.plot(time_vec, phase_vec)
# plt.xlabel("Time (s)")
# plt.ylabel("Cartesian Velocity (m/s)")
# plt.title("Cartesian Velocity (m/s) vs Time (s)")
# plt.legend(['Foot Xdot Desired', 'Foot Ydot Desired', 'Foot Xdot Measured', 'Foot Ydot Measured'])
# ax = plt.gca()
# ax.pcolorfast((time_vec[0],time_vec[-1]),ax.get_ylim(),np.array(phase_vec)[np.newaxis],cmap="Greens", alpha=.3)
# plt.show()

# plt.figure
# plt.plot(time_vec, shoulder_velocity)
# plt.plot(time_vec, elbow_velocity)
# #plt.plot(time_vec, phase_vec)
# plt.xlabel("Time (s)")
# plt.ylabel("Velocity (rad/s)")
# plt.legend(['Shoulder', 'Elbow'])
# plt.title("Velocity (rad/s) vs Time (s)")
# ax = plt.gca()
# ax.pcolorfast((time_vec[0],time_vec[-1]),ax.get_ylim(),np.array(phase_vec)[np.newaxis],cmap="Greens", alpha=.3)
# plt.show()

# plt.figure
# plt.plot(time_vec, shoulder_torque)
# plt.plot(time_vec, elbow_torque)
# plt.plot(time_vec, shoulder_torque_des)
# plt.plot(time_vec, elbow_torque_des)
# #plt.plot(time_vec, phase_vec)
# plt.xlabel("Time (s)")
# plt.ylabel("Torque (Nm)")
# plt.title("Torque (Nm) vs Time (s)")
# plt.legend(['Shoulder', 'Elbow', 'Shoulder Desired', 'Elbow Desired'])
# ax = plt.gca()
# ax.pcolorfast((time_vec[0],time_vec[-1]),ax.get_ylim(),np.array(phase_vec)[np.newaxis],cmap="Greens", alpha=.3)
# plt.show()

# plt.figure
# plt.plot(time_vec_filtered, filtered_shoulder_torque)
# plt.plot(time_vec_filtered, filtered_elbow_torque)
# plt.plot(time_vec_filtered, filtered_shoulder_torque_des)
# plt.plot(time_vec_filtered, filtered_elbow_torque_des)
# plt.xlabel("Time (s)")
# plt.ylabel("Torque (Nm)")
# plt.title("Filtered Torque (Nm) vs Time (s) with moving average filter (window = 100)")
# plt.legend(['Shoulder', 'Elbow', 'Shoulder Desired', 'Elbow Desired'])
# ax = plt.gca()
# ax.pcolorfast((time_vec_filtered[0],time_vec_filtered[-1]),ax.get_ylim(),np.array(phase_vec)[np.newaxis],cmap="Greens", alpha=.3)
# plt.show()

measured_csv_data = np.array([np.array(time_vec),
                    np.array(shoulder_position),
                    np.array(shoulder_velocity),
                    np.array(shoulder_torque),
                    np.array(elbow_position),
                    np.array(elbow_velocity),
                    np.array(elbow_torque)]).T
np.savetxt("measured_data.csv", measured_csv_data, delimiter=',', header="time,_shoulder_position,_shoulder_velocity,shoulder_torque,_elbow_position,_elbow_velocity,elbow_torque", comments="")

desired_csv_data = np.array([np.array(time_vec),
                    np.array(shoulder_position_des),
                    np.array(shoulder_velocity_des),
                    np.array(shoulder_torque_des),
                    np.array(elbow_position_des),
                    np.array(elbow_velocity_des),
                    np.array(elbow_torque_des)]).T
np.savetxt("desired_data.csv", desired_csv_data, delimiter=',', header="time,_shoulder_position,_shoulder_velocity,shoulder_torque,_elbow_position,_elbow_velocity,elbow_torque", comments="")
