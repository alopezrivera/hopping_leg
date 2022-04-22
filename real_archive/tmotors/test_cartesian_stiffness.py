import sys
import time
import numpy as np
import matplotlib.pyplot as plt
from motor_driver.canmotorlib import CanMotorController

sys.path.append("../../hopper_plant/")
from hopper_plant import HopperPlant
sys.path.append("../../Controllers/low_level_control/")
from low_level_control import LowLevelControl

def setZeroPosition(motor, initPos):

    pos = initPos

    while abs(np.rad2deg(pos)) > 0.5:
        pos, vel, curr = motor.set_zero_position()
        print("Position: {}, Velocity: {}, Torque: {}".format(np.rad2deg(pos), np.rad2deg(vel),
                                                                curr))
def contact_detection(effort):
    if np.linalg.norm(effort) > contact_force_threshold:
        print("Contact")
        return True
    else:
        print("No Contact")
        return False

# Create Hopper multi-body plant
L1 = 0.205
L2 = 0.25
g = -9.81
plant = HopperPlant(link_length=[L1, L2], gravity=g)
# Initialize Controller
controller = LowLevelControl(plant)

Kp_shoulder = 5.0
Kd_shoulder = 0.1
Kp_elbow = 5.0
Kd_elbow = 0.1
contact_force_threshold = 1.5

# Desired EE configuration
x_des = 0.35
y_des = 0.0

# Motor ID
motor_shoulder_id = 0x02
motor_elbow_id = 0x04

if len(sys.argv) != 2:
    print('Provide CAN device name (can0, slcan0 etc.)')
    sys.exit(0)

print("Using Socket {} for can communucation".format(sys.argv[1],))

motor_shoulder = CanMotorController(sys.argv[1], motor_shoulder_id)
motor_elbow = CanMotorController(sys.argv[1], motor_elbow_id)

print("Enabling Motors..")

shoulder_pos, shoulder_vel, shoulder_tau = motor_shoulder.enable_motor()

print("Shoulder Motor Status: Pos: {}, Vel: {}, Torque: {}".format(shoulder_pos, shoulder_vel, shoulder_tau))

elbow_pos, elbow_vel, elbow_tau = motor_elbow.enable_motor()

print("Elbow Motor Status: Pos: {}, Vel: {}, Torque: {}".format(elbow_pos, elbow_vel, elbow_tau))

print("Setting Shoulder Motor to Zero Position...")

setZeroPosition(motor_shoulder, shoulder_pos)

print("Setting Elbow Motor to Zero Position...")

setZeroPosition(motor_elbow, elbow_pos)


# Set Joint Angles and Stiffness

print("Seting Joint Angles and Stiffness")

qdes = plant.inverse_kinematics(x_des, y_des, knee_direction=1)
shoulder_pos, shoulder_vel, shoulder_tau = motor_shoulder.send_rad_command(qdes[0], 0, 50, 2, 0)
elbow_pos, elbow_vel, elbow_tau = motor_elbow.send_rad_command(qdes[1], 0, 50, 2, 0)


numSteps = 20000

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

shoulder_torque_des = np.zeros(numSteps)
elbow_torque_des = np.zeros(numSteps)

# Desired Cartesian Coordinates
foot_x = np.zeros(numSteps)
foot_y = np.zeros(numSteps)
foot_xd = np.zeros(numSteps)
foot_yd = np.zeros(numSteps)

# Measured Cartesian coordinates
foot_x_meas = np.zeros(numSteps)
foot_y_meas = np.zeros(numSteps)
foot_xd_meas = np.zeros(numSteps)
foot_yd_meas = np.zeros(numSteps)

start_time = time.time()

shoulder_tau_des = 0.0
elbow_tau_des = 0.0

val = input("Enter y or Y to start cartesian stiffness test: ")

if val == 'y' or val == 'Y':
    print("Starting recording data")
    for i in range(numSteps):

        dt = time.time()
        traj_time = dt - start_time

        time_vec[i] = traj_time
        
        qdes = plant.inverse_kinematics(x_des,y_des, knee_direction=1)
        shoulder_pos_des = qdes[0]
        elbow_pos_des = qdes[1]

        # Send pos, vel and tau_ff command and use the in-built low level controller
        shoulder_pos, shoulder_vel, shoulder_tau = motor_shoulder.send_rad_command(shoulder_pos_des, 0, Kp_shoulder,
                                                                                        Kd_shoulder, shoulder_tau_des)

        elbow_pos, elbow_vel, elbow_tau = motor_elbow.send_rad_command(elbow_pos_des, 0, Kp_elbow, Kd_elbow, elbow_tau_des)


        state = [shoulder_pos, elbow_pos, shoulder_vel, elbow_vel]

        '''
        if i < 20:
            shoulder_tau_des, elbow_tau_des = controller.cartesian_stiffness(state, p_d = [x_des,y_des], pd_d = [0.0,0.0], Kpc = (100,100), Kdc=(10,10), f_ff = [10.0,0.0], knee_direction=1)
        else:
            shoulder_tau_des, elbow_tau_des = controller.cartesian_stiffness(state, p_d = [x_des-0.1,y_des], pd_d = [0.0,0.0], Kpc = (100,100), Kdc=(10,10), knee_direction=1)
        '''
        shoulder_pos_des, elbow_pos_des, shoulder_tau_des, elbow_tau_des = controller.cartesian_stiffness(state, p_d = [x_des,y_des], pd_d = [0.0,0.0], Kpc = (1000,1000), Kdc=(10,10), knee_direction=1)

        # Monitor contacts
        contact_detection(np.array([shoulder_tau, elbow_tau]))

        # Store data in lists
        shoulder_position[i] = shoulder_pos
        shoulder_velocity[i] = shoulder_vel
        shoulder_torque[i] = shoulder_tau

        elbow_position[i] = elbow_pos
        elbow_velocity[i] = elbow_vel
        elbow_torque[i] = elbow_tau

        # Store data in lists
        shoulder_position_des[i] = shoulder_pos_des
        shoulder_velocity_des[i] = 0.0
        shoulder_torque_des[i] = shoulder_tau_des

        elbow_position_des[i] = elbow_pos_des
        elbow_velocity_des[i] = 0.0
        elbow_torque_des[i] = elbow_tau_des

        foot_x[i] = x_des
        foot_y[i] = y_des
        foot_xd[i] = 0.0
        foot_yd[i] = 0.0


print("Disabling Motors...")

shoulder_pos, shoulder_vel, shoulder_tau = motor_shoulder.disable_motor()

print("Shoulder Motor Status: Pos: {}, Vel: {}, Torque: {}".format(shoulder_pos, shoulder_vel, shoulder_tau))

elbow_pos, elbow_vel, elbow_tau = motor_elbow.disable_motor()

print("Elbow Motor Status: Pos: {}, Vel: {}, Torque: {}".format(elbow_pos, elbow_vel, elbow_tau))

print("Calculating FK offline...")
for i in range(numSteps):
    # Compute FK
    x_meas, y_meas = plant.forward_kinematics(shoulder_position[i], elbow_position[i])
    xd_meas, yd_meas = plant.forward_velocity(shoulder_position[i], elbow_position[i], shoulder_velocity[i], elbow_velocity[i])

    foot_x_meas[i] = x_meas
    foot_y_meas[i] = y_meas
    foot_xd_meas[i] = xd_meas
    foot_yd_meas[i] = yd_meas 


plt.figure
plt.plot(time_vec, shoulder_position)
plt.plot(time_vec, elbow_position)
plt.plot(time_vec, shoulder_position_des)
plt.plot(time_vec, elbow_position_des)
plt.xlabel("Time (s)")
plt.ylabel("Position (rad)")
plt.title("Position (rad) vs Time (s)")
plt.legend(['Shoulder', 'Elbow', 'Shoulder Desired', 'Elbow Desired'])
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
plt.plot(time_vec, shoulder_torque_des)
plt.plot(time_vec, elbow_torque_des)
plt.xlabel("Time (s)")
plt.ylabel("Torque (Nm)")
plt.title("Torque (Nm) vs Time (s)")
plt.legend(['Shoulder', 'Elbow', 'Shoulder Desired', 'Elbow Desired'])
plt.show()


def running_mean(x, N):
    cumsum = np.cumsum(np.insert(x, 0, 0))
    return (cumsum[N:] - cumsum[:-N]) / float(N)


filtered_shoulder_torque = running_mean(np.array(shoulder_torque), 10)
filtered_elbow_torque = running_mean(np.array(elbow_torque), 10)
filtered_shoulder_torque_des = running_mean(np.array(shoulder_torque_des), 10)
filtered_elbow_torque_des = running_mean(np.array(elbow_torque_des), 10)
time_vec_filtered = running_mean(np.array(time_vec), 10)

plt.figure
plt.plot(time_vec_filtered, filtered_shoulder_torque)
plt.plot(time_vec_filtered, filtered_elbow_torque)
plt.plot(time_vec_filtered, filtered_shoulder_torque_des)
plt.plot(time_vec_filtered, filtered_elbow_torque_des)
plt.xlabel("Time (s)")
plt.ylabel("Torque (Nm)")
plt.title("Filtered Torque (Nm) vs Time (s) with moving average filter (window = 100)")
plt.legend(['Shoulder', 'Elbow', 'Shoulder Desired', 'Elbow Desired'])
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

