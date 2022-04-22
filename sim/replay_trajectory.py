import numpy as np
import pybullet as p
# from matplotlib import pyplot as plt # import may lead to strange behaviour of pybullet
import argparse
from argparse import RawTextHelpFormatter
import pandas as pd

from simulation.kinematics_simulation import SimulationTracker, HopperSimulation


if __name__  == "__main__":
# Create the parser
    parser = argparse.ArgumentParser(
        description="Simulation of hopping leg based on quasi-direct drives",
        formatter_class=RawTextHelpFormatter)
    parser.add_argument("--position_control", "-p", action='store_true',
                        help="Use position control mode")
    parser.add_argument("--velocity_control", "-v", action='store_true',
                        help="Use velocity control mode")
    parser.add_argument("--torque_control", "-t", action='store_true',
                        help="Use torque control mode")
    parser.add_argument("--save_results", "-s", action='store_true',
                        help="Save results to the results folder (../results/control_mode)")
    parser.add_argument("--rail", "-r", action='store_true',
                        help="Use URDF with rail")
    parser.add_argument("--long_rail", "-l", action='store_true',
                        help="Use URDF with long rail")


    # Execute the parse_args method
    args, unknown = parser.parse_known_args()
    if not (args.position_control or args.velocity_control or args.torque_control):
        args.position_control = True


    results_folder = "../results/trajectory_replay" 
    
    # load data
    ############################################# include csv path for replaying here #######################################
    data = pd.read_csv("../results/energy_shaping_control/trajectoryexport.csv")
    #########################################################################################################################
    
    sim = HopperSimulation(2,(args.rail or args.long_rail),long_leg=args.long_rail,frequency=700)
    tracker = SimulationTracker(sim)
    
    for i in range(len(data["time"])):
        q1= data["shoulder_pos"][i]
        q2= data["elbow_pos"][i]
        dq1= data["shoulder_vel"][i]
        dq2= data["elbow_vel"][i]
        tau1= data["shoulder_torque"][i]
        tau2= data["elbow_vel"][i]
        
        sim.step_time()
        
        
        if args.position_control:
            p.setJointMotorControlArray(
                    sim.robot, sim.jointIndices, p.POSITION_CONTROL, [q1,q2], forces=[12,12])
        elif args.velocity_control:        
           p.setJointMotorControlArray(
                    sim.robot, sim.jointIndices, p.VELOCITY_CONTROL, [dq1,dq2], forces=[12, 12])
        elif args.torque_control:
            p.setJointMotorControlArray(
                    sim.robot, sim.jointIndices, p.TORQUE_CONTROL, forces=[tau1, tau2])
        
        
        # track data
        xdes,ydes = sim.plant.forward_kinematics(q1,q2) 
        dxdes,dydes = sim.plant.forward_velocity(q1,q2,dq1,dq2)
        
        tracker.update(tau1, tau2)
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
        # tracker.phase.append(phase_dict[ce.phase])
        # move leg up and down (position control)
        p.stepSimulation()
        sim.sleep()
    
    p.disconnect()
    save = save_figures=args.save_results
    tracker.create_plots(save_figures=save,
                         folder="../results/"+results_folder)
    if save:
        tracker.write_csv("../results/"+results_folder)
        tracker.trajectory_export("../results/"+results_folder)