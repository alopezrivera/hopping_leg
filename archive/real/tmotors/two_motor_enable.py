import sys
import time
import numpy as np
from motor_driver.canmotorlib import CanMotorController


def setZeroPosition(motor, initPos):

    pos = initPos

    while abs(np.rad2deg(pos)) > 0.5:
        pos, vel, curr = motor.set_zero_position()
        print("Position: {}, Velocity: {}, Torque: {}".format(np.rad2deg(pos), np.rad2deg(vel),
                                                                curr))


# Motor ID
motor_shoulder_id = 0x02
motor_elbow_id = 0x01

if len(sys.argv) != 2:
    print('Provide CAN device name (can0, slcan0 etc.)')
    sys.exit(0)

print("Using Socket {} for can communucation".format(sys.argv[1],))

motor_shoulder = CanMotorController(sys.argv[1], motor_shoulder_id)
motor_elbow = CanMotorController(sys.argv[1], motor_elbow_id)

print("Enabling Motors..")

shoulder_pos, shoulder_vel, shoulder_torque = motor_shoulder.enable_motor()

print("Shoulder Motor Status: Pos: {}, Vel: {}, Torque: {}".format(shoulder_pos, shoulder_vel, shoulder_torque))

elbow_pos, elbow_vel, elbow_torque = motor_elbow.enable_motor()

print("Elbow Motor Status: Pos: {}, Vel: {}, Torque: {}".format(elbow_pos, elbow_vel, elbow_torque))

print("Setting Shoulder Motor to Zero Position...")

setZeroPosition(motor_shoulder, shoulder_pos)

print("Setting Elbow Motor to Zero Position...")

setZeroPosition(motor_elbow, elbow_pos)
'''
startT = time.time()
for i in range(1000):
    pos, vel, tau = motor_shoulder.send_rad_command(0, 0, 0, 0, 0)
    pos, vel, tau = motor_elbow.send_rad_command(0, 0, 0, 0, 0)

print("Freq: {}".format(1000/(time.time() - startT)))
'''
print("Disabling Motors...")

shoulder_pos, shoulder_vel, shoulder_torque = motor_shoulder.disable_motor()

print("Shoulder Motor Status: Pos: {}, Vel: {}, Torque: {}".format(shoulder_pos, shoulder_vel, shoulder_torque))

elbow_pos, elbow_vel, elbow_torque = motor_elbow.disable_motor()

print("Elbow Motor Status: Pos: {}, Vel: {}, Torque: {}".format(elbow_pos, elbow_vel, elbow_torque))
