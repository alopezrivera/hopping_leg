# Dependencies

* pybullet (pip3 install pybullet)
* numpy
* matplotlib
* pandas (just for replay trajectory script)

# Using the programs

## kinematics simulation
Main method contains easy controller which follow a pre defined sin-like shape.

```
python3 kinematics_simulation.py --help
usage: kinematics_simulation.py [-h] [--position_control] [--velocity_control]
                                [--torque_control] [--save_results] [--rail]

Simulation of hopping leg based on quasi-direct drives

optional arguments:
  -h, --help              show this help message and exit
  --position_control, -p  Use position control mode
  --velocity_control, -v  Use velocity control mode
  --torque_control, -t    Use torque control mode
  --save_results, -s      Save results to the results folder (../results/control_mode)
  --rail, -r              Use URDF with rail
```

To use the position control mode, use:
`python3 kinematics_simulation.py --position_control`

To use the velocity control mode, use:
`python3 kinematics_simulation.py --velocity_control`

To use the torque control mode, use:
`python3 kinematics_simulation.py --torque_control`

To save the results, provide the `--save_results` flag with any of the control options. For e.g.:

`python3 kinematics_simulation.py --position_control --save_results`

To use the URDF file with rail, provide the `--rail` flag with any of the control options. For e.g.:

`python3 kinematics_simulation.py --position_control --rail`



## energy shaping backflip
Script contains gain scheduling and energy shaping controller. Optional, the energy shaping controller can perform sequences of back- and forward flips.


```
python3 energy_shaping_backflip.py --help
usage: energy_shaping_backflip.py [-h] [--gain_scheduling] [--energy_shaping]
                                  [--backflip] [--save_results] [--rail]
                                  [--long_rail]

Simulation of hopping leg based on quasi-direct drives

optional arguments:
  -h, --help            show this help message and exit
  --gain_scheduling, -g
                        Use gain scheduling control mode
  --energy_shaping, -e  Use energy shaping control mode
  --backflip, -b        Use backflip control mode
  --save_results, -s    Save results to the results folder (../results/control_mode)
  --rail, -r            Use URDF with rail
  --long_rail, -l       Use URDF with long rail
```

To use the gain scheduling control mode, use:
`python3 energy_shaping_backflip.py --gain_scheduling`


To use the energy shaping control mode, use:
`python3 energy_shaping_backflip.py --energy_shaping`

To use the energy shaping control mode with backflips, use:
`python3 energy_shaping_backflip.py --backflip`

To save the results, provide the `--save_results` flag with any of the control options. For e.g.:

`python3 energy_shaping_backflip.py --energy_shaping --save_results`

To use the URDF file with rail, provide the `--rail` flag with any of the control options. For e.g.:

`python3 energy_shaping_backflip.py --energy_shaping --rail`

To use the URDF file with rail and long leg, provide the `--long_rail` flag with any of the control options. For e.g.:

`python3 energy_shaping_backflip.py --energy_shaping --long_rail`


### Changing the flip sequence
The flip sequence can be changed in the script by changing the variable `modes` in the follorwing lines:
```
 elif args.backflip:
        results_folder = "backflip_energy_shaping_control"
        # existing modes: 
        # "j": jump, "b": backflip, "bc": backflip + change knee direction, "bd": double backflip, 
        # "f": forwardflip, "fc": forwardflip + change knee direction, "fd": double forwardflip
        modes = ["j", "j", "b", "j", "fc", "b"]
```
The variable `modes` has to be a list of strings, which can be `"j"` for a normal jump, `"b"` for a backflip, `"bc"` for a  backflip with changing the knee direction at the same time, `"bd"` for performing a double backflip, 
        `"f"` for a forward flip, `"fc"` to perform a forwardflip with changing the  knee direction or `"fd"` for a  double forward flip.


## Replay Trajectory
The replay trajectory script command a sequence of joint positions, velocities or torques from a csv file in simulation.

```
python replay_trajectory.py --help
usage: replay_trajectory.py [-h] [--position_control] [--velocity_control]
                            [--torque_control] [--save_results] [--rail]
                            [--long_rail]

Simulation of hopping leg based on quasi-direct drives

optional arguments:
  -h, --help            show this help message and exit
  --position_control, -p
                        Use position control mode
  --velocity_control, -v
                        Use velocity control mode
  --torque_control, -t  Use torque control mode
  --save_results, -s    Save results to the results folder (../results/control_mode)
  --rail, -r            Use URDF with rail
  --long_rail, -l       Use URDF with long rail
```

To use the position control mode, use:
`python3 replay_trajectory.py --position_control`

To use the velocity control mode, use:
`python3 replay_trajectory.py --velocity_control`

To use the torque control mode, use:
`python3 replay_trajectory.py --torque_control`

To save the results, provide the `--save_results` flag with any of the control options. For e.g.:

`python3 replay_trajectory.py --position_control --save_results`

To use the URDF file with rail, provide the `--rail` flag with any of the control options. For e.g.:

`python3 replay_trajectory.py --position_control --rail`

To use the URDF file with rail and long leg, provide the `--long_rail` flag with any of the control options. For e.g.:

`python3 replay_trajectory.py --position_control --long_rail`


### Seting up the path for trajecotry replay
The path of the csv file to replay can be changed in the script by changing the path in the `data` variable in the following lines:
```
############################################# include csv path for replaying here #######################################
data = pd.read_csv("../results/energy_shaping_control/trajectoryexport.csv")
#########################################################################################################################
```

Currently the script requires the csv file to have following columns:
`time`, `shoulder_pos`, `elbow_pos`, `shoulder_vel`, `elbow_vel`, `shoulder_torque` and `elbow_torque`.

To change that, you have to change the strings in the following lines:
```
for i in range(len(data["time"])):
        q1= data["shoulder_pos"][i]
        q2= data["elbow_pos"][i]
        dq1= data["shoulder_vel"][i]
        dq2= data["elbow_vel"][i]
        tau1= data["shoulder_torque"][i]
        tau2= data["elbow_vel"][i]
``` 


# Using the HopperSimulation class in other scripts
The hopper simulation class has been written to setup a pybullet simulation in few lines of code.
To import it to other scripts use:
```
import pybullet as p
import sys
sys.path.append("/path/to/pybullet_simulation")
from kinematics_simulation import HopperSimulation
```

## Usage:
To setup the simulation and load the urdf use:
`simulation = HopperSimulation(control_mode, rail, drop_height, gravity, torque_limits,long_leg, frequency)`
The contructor has following doc string, which explains the available options:
```
"""
Hopper simulation class which setups the environment and the correct physical parameters.

Args:
    control_mode (int, optinal): Prepare simulation for 
            0: Position control.
            1: Velocity control.
            2: Torque control.
        Defaults to 2
    rail (bool, optional): Use urdf with rail. Defaults to True.
    drop_height (float, optional): Drop height at beginning of the simulation. Defaults to 0.5.
    gravity (float, optional): Gravity. Defaults to 9.81.
    torque_limits (tuple, optional): Motor torque limits. Defaults to (6,6).
    long_leg (bool, optional): Use urdf with long leg (requires rail to be True). Defaults to False.
    frequency ([type], optional): Control frequency in Hz. 
        By default 240 for position control, 500 for velocity control and 1000 for torque control.
"""
```


A minimal control loop can look like:
```
while True:
    # step the current time
    simulation.step_time()

    # send motor commands with pybullet
    p.setJointMotorControlArray(
                    simulation.robot, simulation.jointIndices, p.TORQUE_CONTROL, forces=[tau1, tau2])

    # step the simulation
    p.stepSimulation()
    # sleep for one timestep
    simulation.sleep()

# disconnect pybullet
p.disconnect()

```


# Using the SimulationTracker class
The SimulationTracker class has been written to save all data (positions, velocities etc) during a simulation. It can be used to save the data afterwards and/or create some plots.

To import the class in other scripts use:
```
import sys
sys.path.append("/path/to/pybullet_simulation")
from kinematics_simulation import SimulationTracker
```

## Usage
Initialize the Simulation tracker use:
`tracker = SimulationTracker(simulation)`
The SImulationTracker requires an instance of the HopperSimulation as argument.

During the control loop, use 
`tracker.update()` 
to save the current joint and end effector positions and velocities, the joint torques and the ground reaction forces. 

Note: During torque control mode the `tracker.update()` function requires the commandes motor torques as arguments:
`tracker.update(tau1, tau2)`.

For saving further values, like desired positions, there are several lists in the tracker, which can be appended:
eg: 
```
tracker.q1_desired.append(q1)
tracker.q2_desired.append(q2)
tracker.q1d_desired.append(dq1)
tracker.q2d_desired.append(dq2)
tracker.tau1_desired.append(tau1)
tracker.tau2_desired.append(tau2)
tracker.x_desired.append(xdes) 
tracker.y_desired.append(ydes) 
tracker.xd_desired.append(dxdes) 
tracker.yd_desired.append(dydes)
```
If you require other values look up in the `__init__()` mehtod, which lists are already defined.


To create some plots use:
`tracker.create_plots(save_figures=False, folder="../results/)`

The data can be saved as csv with:
`tracker.write_csv("../results/")`

Or formatted for the usage in the trajectory_replay script using:
`tracker.trajectory_export("../results/")`




# Issues with pybullet

On some computers importing matplotlib.pyplot before loading the urdf file of the hopper can lead 
to strange behaviour in pybullet. This can be avoided by commenting out the import and import it somewere 
later in code just before use. 