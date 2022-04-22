from motors import mjbot
from control import FDDP

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

motors = mjbot

input('confirm run')

if input:
    run_control_loop(FDDP)
    postprocess()
    visualize()