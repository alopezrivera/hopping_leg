import sys
import time
import numpy as np
import matplotlib.pyplot as plt
from canmotorlib import CanMotorController

sys.path.append("../hopping_leg")
from hopping_leg.plant.hopper import HopperPlant
from hopping_leg.controllers.PD import PD_controller

from hopping_leg.utilities.cli import printu
from hopping_leg.utilities.file import csv
from hopping_leg.utilities.stats import running_mean
from hopping_leg.utilities.real import motors, enable, zero, disable

from real.heuristics import contact_detection, estimate_state, get_control

gains = {'LIFTOFF':   {'Kp_shoulder': 1.0,
                       'Kd_shoulder': 1.0,
                       'Kp_elbow':    1.0,
                       'Kd_elbow':    1.0},
         'TOUCHDOWN': {'Kp_shoulder': 5.0,
                       'Kd_shoulder': 0.1,
                       'Kp_elbow':    5.0,
                       'Kd_elbow':    0.1},
         'ELSE':      {'Kp_shoulder': 50.0,
                       'Kd_shoulder': 2.0,
                       'Kp_elbow':    50.0,
                       'Kd_elbow':    2.0}}

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
#       Control Loop Variable Declaration       #
#                                               #
#################################################
init     = lambda var: locals().update({var: initArr})
desired  = lambda par: f"{par}_des"
measured = lambda par: f"{par}_mes"

# Current namespace variables
nmspc0   = locals()

# Timesteps
numSteps = 5000
initArr  = np.zeros(numSteps)

# Motors
motors = ['shoulder', 'elbow']
# Parameters
state_params = ['position', 'velocity', 'torque']
cart_params  = ['x', 'y', 'xd', 'yd']

#################################################
#             Array Initialization              #
#################################################
# Time
time_vec = initArr

# Phase
phase_vec = initArr

# Initialize motor state arrays
for motor in motors:
    for var in state_params + [desired(param) for param in state_params]:
        init(f"{motor}_{var}")

# Initialize foot Cartesian cordinate arrays
for var in cart_params + [measured(param) for param in state_params]:
    init(f"foot_{var}")

#################################################
#              Initial Conditions               #
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
#       Control Loop Variable Dictionary        #
#################################################
nmspc1 = locals()
inputs = {k: nmspc1[k] for k in nmspc1.keys() - nmspc0.keys()}

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

if val == 'y' or val == 'Y':

    main(**inputs)

def main(**kwargs):
    """
    Control Loop
    """

    # Retrieve local variables by name string
    get = lambda var: locals()[var]

    # Start time
    start_time = time.time()

    for i in range(numSteps):

        # Record elapsed time
        time_vec[i] = time.time() - start_time
        
        # Gain scheduling
        locals().update(gains[current_phase] if current_phase in gains.keys() else gains['ELSE'])

        # Motor controller commands --------------------------------------------------------------------------------------------------------------------------------------------------------------
        _shoulder_position, _shoulder_velocity, _shoulder_torque = motor_shoulder.send_rad_command(_shoulder_position_des, 
                                                                                                    _shoulder_velocity_des, 
                                                                                                    Kp_shoulder, 
                                                                                                    Kd_shoulder, 
                                                                                                    _shoulder_torque_des)
        _elbow_position, _elbow_velocity, _elbow_torque          = motor_elbow.send_rad_command(_elbow_position_des, 
                                                                                                _elbow_velocity_des, 
                                                                                                Kp_elbow, 
                                                                                                Kd_elbow,
                                                                                                _elbow_torque_des)
        # ----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
        
        # Update phase
        phase_vec[i]         = phase_dict[current_phase]

        # Update state
        state                = [_shoulder_position, _elbow_position, _shoulder_velocity, _elbow_velocity]
        control              = get_control(controller, current_phase, state, np.array([_shoulder_torque, _elbow_torque]), desired_height)
        
        _shoulder_position_des, _elbow_position_des, _shoulder_velocity_des, _elbow_velocity_des, _shoulder_torque_des, _elbow_torque_des, current_phase, flight_counter, maxh, height_factor, v0, t0 = control

        # Update height estimation
        height_est[i]        = estimate_state(state,current_phase, t0,v0)
        height_factor_vec[i] = height_factor

        # Compute Inverse Dynamics for feed-forward torques in joint space
        #tau = plant.gravity_vector(_shoulder_position_des, _elbow_position_des)
        #_shoulder_torque_des = _shoulder_torque_des + tau[1]
        #_elbow_torque_des = _elbow_torque_des + tau[2]

        # Record state
        for motor in motors:
            for var in state_params + [desired(param) for param in state_params]:
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
        foot_x_meas[i], foot_y_meas[i]   = plant.forward_kinematics(shoulder_position[i], elbow_position[i])
        foot_xd_meas[i], foot_yd_meas[i] = plant.forward_velocity(shoulder_position[i], elbow_position[i], shoulder_velocity[i], elbow_velocity[i])

        # Compute FK for desired data
        foot_x[i], foot_y[i]   = plant.forward_kinematics(shoulder_position_des[i], elbow_position_des[i])
        foot_xd[i], foot_yd[i] = plant.forward_velocity(shoulder_position_des[i], elbow_position_des[i], shoulder_velocity_des[i], elbow_velocity_des[i])

    print("Calculating filtered states...")
    filtered_shoulder_torque     = running_mean(np.array(shoulder_torque),     10)
    filtered_elbow_torque        = running_mean(np.array(elbow_torque),        10)
    filtered_shoulder_torque_des = running_mean(np.array(shoulder_torque_des), 10)
    filtered_elbow_torque_des    = running_mean(np.array(elbow_torque_des),    10)
    time_vec_filtered            = running_mean(np.array(time_vec),            10)

    #################################################
    #                                               #
    #                Visualization                  #
    #                                               #
    #################################################

    from mpl_plotter.two_d import panes

    # State -----------------------------------------
    arguments = []
    for stp in state_params:
        arguments += [get(stp), get(f"{stp}_des")]
    # Cartesian parameters
    for crp in cart_params:
        arguments += [get(f"foot_{stp}"), get(f"foot_{crp}_mes")]
    panes(time_vec,
        [shoulder_position, shoulder_position_des])
    comparison(time_vec,
            [shoulder_position, elbow_position, shoulder_position_des, elbow_position_des])

    # Filtered torque -------------------------------
    arguments = []
    for motor in motors:
        arguments += [get(f"filtered_{motor}_torque"), get(f"filtered_{motor}_torque_des")]
    
    # Estimated height ------------------------------
    # X vs Y ----------------------------------------

    #################################################
    #                                               #
    #                  Data Export                  #
    #                                               #
    #################################################

    # Record states through experiment
    csv([get(param) for param in state_params], motors,
        'measured_states.csv')

    # Record desired states through experiment
    csv([get(desired(param)) for param in state_params], motors,
        'desired_states.csv')
