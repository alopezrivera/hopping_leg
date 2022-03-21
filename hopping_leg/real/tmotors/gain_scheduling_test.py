import sys
import time
import numpy as np
import matplotlib.pyplot as plt
#from motor_driver.canmotorlib import CanMotorController
from canmotorlib import CanMotorController
sys.path.append("../../hopper_plant/")
from hopper_plant import HopperPlant
sys.path.append("../../Controllers/low_level_control/")
from low_level_control import LowLevelControl

tstart = time.time()
t0 = 0
v0 = 0
def setZeroPosition(motor, initPos):

    pos = initPos

    while abs(np.rad2deg(pos)) > 0.5:
        pos, vel, curr = motor.set_zero_position()
        print("Position: {}, Velocity: {}, Torque: {}".format(np.rad2deg(pos), np.rad2deg(vel),
                                                                curr))
def contact_detection(effort):
    if abs(effort[0]) >= contact_force_threshold or abs(effort[1]) >= contact_force_threshold:
        return True
    else:
        return False

def estimate_state(state, phase, t0, v0, g = -9.81):
        
        
        if phase not in ("FLIGHT", "FLIGHT_ASCEND"):
            return plant.forward_kinematics(*state[:2])[0]
        t = time.time() - tstart
        # t0 = self.time_at_liftoff
        x0 = x_des+0.12
        # v0 = self.liftoff_vel
        # g = -self.plant.g
        x = g/2 *(t**2-t0**2) - g*t0*(t-t0) + v0*(t-t0) + x0
        return x
    
flight_counter = 0

def get_control(controller, phase, state, effort):
    global flight_counter
    if phase == "FLIGHT":
        shoulder_pos_des, elbow_pos_des, shoulder_vel_des, elbow_vel_des, tau1, tau2 = controller.cartesian_stiffness(state, p_d = [x_des,y_des], pd_d = [0.0,0.0], Kpc = (1000,2000), Kdc=(10,10), f_ff = [0.0,0.0], knee_direction=1)
        
        if contact_detection(effort) and plant.forward_kinematics(*state[:2])[0]<=x_des-0.005:
            phase = "TOUCHDOWN"         

    if phase == "TOUCHDOWN":
        #change
        shoulder_pos_des, elbow_pos_des, shoulder_vel_des, elbow_vel_des, tau1, tau2 = controller.cartesian_stiffness(state, p_d = [x_des,y_des], pd_d = [0.0,0.0], Kpc = (10.0,2000), Kdc=(10,10), f_ff = [0.0,0.0], knee_direction=1)

        #change
        if np.linalg.norm(plant.forward_velocity(*state))<0.1 and plant.forward_kinematics(*state[:2])[0]<=x_des-0.07:
            phase = "LIFTOFF"

    elif current_phase == "LIFTOFF":
        # change
        shoulder_pos_des, elbow_pos_des, shoulder_vel_des, elbow_vel_des, tau1, tau2 = controller.cartesian_stiffness(state, p_d = [x_des+0.2,y_des], pd_d = [0.0,0.0], Kpc = (1000,2000), Kdc=(10,10), f_ff = [100.0,0.0], knee_direction=1)

        if plant.forward_kinematics(*state[:2])[0]>=x_des+0.12: 
            global v0
            global t0
            v0 = plant.forward_velocity(*state)[0]
            t0 = time.time()-tstart
            phase = "FLIGHT_ASCEND"
        
        #if not contact_detection(effort):
        #    phase = "FLIGHT"

    elif current_phase == "FLIGHT_ASCEND":
        shoulder_pos_des, elbow_pos_des, shoulder_vel_des, elbow_vel_des, tau1, tau2 = controller.cartesian_stiffness(state, p_d = [x_des,y_des], pd_d = [0.0,0.0], Kpc = (1000,2000), Kdc=(10,10), f_ff = [0.0,0.0], knee_direction=1)
        
        if not contact_detection(effort):
            flight_counter = flight_counter + 1
            phase = "FLIGHT"

    return shoulder_pos_des, elbow_pos_des, tau1,tau2, phase
        

# Create Hopper multi-body plant
L1 = 0.205
L2 = 0.2
g = -9.81
plant = HopperPlant(link_length=[L1, L2], gravity=g, torque_limits = [18,12])
# Initialize Controller
controller = LowLevelControl(plant)

Kp_shoulder = 5.0
Kd_shoulder = 0.1
Kp_elbow = 5.0
Kd_elbow = 0.1
contact_force_threshold = 10

# Desired EE configuration
x_des = 0.2
y_des = 0.0

# Motor ID
motor_shoulder_id = 0x05
motor_elbow_id = 0x01

if len(sys.argv) != 2:
    print('Provide CAN device name (can0, slcan0 etc.)')
    sys.exit(0)

print("Using Socket {} for can communucation".format(sys.argv[1],))

#motor_shoulder = CanMotorController(sys.argv[1], motor_shoulder_id)
#motor_elbow = CanMotorController(sys.argv[1], motor_elbow_id)

motor_shoulder = CanMotorController(can_socket='can0', motor_id=motor_shoulder_id, motor_type='AK80_9_V2', socket_timeout=0.5)
motor_elbow = CanMotorController(can_socket='can0', motor_id=motor_elbow_id, motor_type='AK80_6_V2', socket_timeout=0.5)

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


numSteps = 5000

time_vec = np.zeros(numSteps)

phase_vec = np.zeros(numSteps)

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
height_est = np.zeros(numSteps)
start_time = time.time()

shoulder_tau_des = 0.0
elbow_tau_des = 0.0
current_phase = "LIFTOFF"

phase_dict = {'FLIGHT': 0, 'TOUCHDOWN': 1, 'LIFTOFF': 2, 'FLIGHT_ASCEND': 3}

# For the initial phase
shoulder_pos_des = qdes[0]
elbow_pos_des = qdes[1]

val = input("Enter y or Y to start gain scheduling test: ")

if val == 'y' or val == 'Y':
    print("Starting recording data")
    for i in range(numSteps):

        dt = time.time()
        traj_time = dt - start_time

        time_vec[i] = traj_time
        
        # Send pos, vel and tau_ff command and use the in-built low level controller
        if current_phase in ["LIFTOFF", "TOUCHDOWN"]:
            Kp_shoulder = 5.0
            Kd_shoulder = 0.1
            Kp_elbow = 5.0
            Kd_elbow = 0.1
        else:
            Kp_shoulder = 50.0
            Kd_shoulder = 2.0
            Kp_elbow = 50.0
            Kd_elbow = 2.0

        shoulder_pos, shoulder_vel, shoulder_tau = motor_shoulder.send_rad_command(shoulder_pos_des, 0, Kp_shoulder, Kd_shoulder, shoulder_tau_des)
        elbow_pos, elbow_vel, elbow_tau = motor_elbow.send_rad_command(elbow_pos_des, 0, Kp_elbow, Kd_elbow, elbow_tau_des)

        state = [shoulder_pos, elbow_pos, shoulder_vel, elbow_vel]

        phase_vec[i] = phase_dict[current_phase]

        shoulder_pos_des, elbow_pos_des, shoulder_tau_des, elbow_tau_des, current_phase = get_control(controller, current_phase, state, np.array([shoulder_tau, elbow_tau]))
        #shoulder_tau_des, elbow_tau_des = controller.cartesian_stiffness(state, p_d = [x_des,y_des], pd_d = [0.0,0.0], Kpc = (1000,1000), Kdc=(10,10), f_ff = [0.0,0.0], knee_direction=1)
        height_est[i] = estimate_state(state,current_phase, t0,v0)

        # Compute Inverse Dynamics for feed-forward torques in joint space
        tau = plant.gravity_vector(shoulder_pos_des, elbow_pos_des)
        shoulder_tau_des = shoulder_tau_des + tau[1]
        elbow_tau_des = elbow_tau_des + tau[2]

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


print("Motion finished...")
qdes = plant.inverse_kinematics(x_des, y_des, knee_direction=1)
shoulder_pos, shoulder_vel, shoulder_tau = motor_shoulder.send_rad_command(qdes[0], 0, 50, 2, 0)
elbow_pos, elbow_vel, elbow_tau = motor_elbow.send_rad_command(qdes[1], 0, 50, 2, 0)

time.sleep(5)

print("Disabling Motors...")

shoulder_pos, shoulder_vel, shoulder_tau = motor_shoulder.disable_motor()

print("Shoulder Motor Status: Pos: {}, Vel: {}, Torque: {}".format(shoulder_pos, shoulder_vel, shoulder_tau))

elbow_pos, elbow_vel, elbow_tau = motor_elbow.disable_motor()

print("Elbow Motor Status: Pos: {}, Vel: {}, Torque: {}".format(elbow_pos, elbow_vel, elbow_tau))

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

print("Number of Jumps:", flight_counter)

plt.figure
plt.plot(time_vec, shoulder_position)
plt.plot(time_vec, elbow_position)
plt.plot(time_vec, shoulder_position_des)
plt.plot(time_vec, elbow_position_des)
#plt.plot(time_vec, phase_vec)
plt.xlabel("Time (s)")
plt.ylabel("Position (rad)")
plt.title("Position (rad) vs Time (s)")
plt.legend(['Shoulder', 'Elbow', 'Shoulder Desired', 'Elbow Desired'])
ax = plt.gca()
ax.pcolorfast((time_vec[0],time_vec[-1]),ax.get_ylim(),np.array(phase_vec)[np.newaxis],cmap="Greens", alpha=.3)
plt.show()

plt.figure()
plt.plot(time_vec, height_est)
plt.xlabel("Time (s)")
plt.ylabel("Position x (m)")
plt.legend(["Estimated height"])
ax = plt.gca()
ax.pcolorfast((time_vec[0],time_vec[-1]),ax.get_ylim(),np.array(phase_vec)[np.newaxis],cmap="Greens", alpha=.3)
plt.show()


plt.figure
plt.plot(time_vec, foot_x)
plt.plot(time_vec, foot_y)
plt.plot(time_vec, foot_x_meas)
plt.plot(time_vec, foot_y_meas)
ax = plt.gca()
ax.pcolorfast((time_vec[0],time_vec[-1]),ax.get_ylim(),np.array(phase_vec)[np.newaxis],cmap="Greens", alpha=.3)
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
#plt.plot(time_vec, phase_vec)
plt.xlabel("Time (s)")
plt.ylabel("Cartesian Velocity (m/s)")
plt.title("Cartesian Velocity (m/s) vs Time (s)")
plt.legend(['Foot Xdot Desired', 'Foot Ydot Desired', 'Foot Xdot Measured', 'Foot Ydot Measured'])
ax = plt.gca()
ax.pcolorfast((time_vec[0],time_vec[-1]),ax.get_ylim(),np.array(phase_vec)[np.newaxis],cmap="Greens", alpha=.3)
plt.show()

plt.figure
plt.plot(time_vec, shoulder_velocity)
plt.plot(time_vec, elbow_velocity)
#plt.plot(time_vec, phase_vec)
plt.xlabel("Time (s)")
plt.ylabel("Velocity (rad/s)")
plt.legend(['Shoulder', 'Elbow'])
plt.title("Velocity (rad/s) vs Time (s)")
ax = plt.gca()
ax.pcolorfast((time_vec[0],time_vec[-1]),ax.get_ylim(),np.array(phase_vec)[np.newaxis],cmap="Greens", alpha=.3)
plt.show()

plt.figure
plt.plot(time_vec, shoulder_torque)
plt.plot(time_vec, elbow_torque)
plt.plot(time_vec, shoulder_torque_des)
plt.plot(time_vec, elbow_torque_des)
#plt.plot(time_vec, phase_vec)
plt.xlabel("Time (s)")
plt.ylabel("Torque (Nm)")
plt.title("Torque (Nm) vs Time (s)")
plt.legend(['Shoulder', 'Elbow', 'Shoulder Desired', 'Elbow Desired'])
ax = plt.gca()
ax.pcolorfast((time_vec[0],time_vec[-1]),ax.get_ylim(),np.array(phase_vec)[np.newaxis],cmap="Greens", alpha=.3)
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
ax = plt.gca()
ax.pcolorfast((time_vec_filtered[0],time_vec_filtered[-1]),ax.get_ylim(),np.array(phase_vec)[np.newaxis],cmap="Greens", alpha=.3)
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

