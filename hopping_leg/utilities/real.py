"""
Real System
===========
"""


def motors(motor_ids, command_line_args):

    if len(args) != 2:
        print('Provide CAN device name (can0, slcan0 etc.)')
        sys.exit(0)

    can = command_line_args[1]

    print(f"Using Socket {can} for can communucation")

    return [CanMotorController(can, id) for id in motor_ids]


def enable(motor, name=""):
    """
    Enable motor
    """

    print(f"Enabling {name.title() + ' ' if name else name}Motor...")

    pos, vel, tau = motor.enable_motor()

    print(f"{name + ' ' if name else name}Motor Status: Pos: {pos}, Vel: {vel}, Torque: {tau}")

    return pos, vel, tau


def zero(motor, initPos, name=""):
    """
    Set zero position of motor
    """

    print(f"Setting {name.title() + ' ' if name else name}Motor to Zero Position...")

    pos = initPos

    while abs(np.rad2deg(pos)) > 0.5:
        pos, vel, curr = motor.set_zero_position()
        print("Position: {}, Velocity: {}, Torque: {}".format(np.rad2deg(pos), np.rad2deg(vel), curr))


def disable(motor, name=""):
    """
    Disable motor
    """

    print(f"Disabling {name.title() + ' ' if name else name}Motor...")

    pos, vel, tau = motor.disable_motor()

    print(f"{name + ' ' if name else name}Motor Status: Pos: {pos}, Vel: {vel}, Torque: {tau}")

    return pos, vel, tau