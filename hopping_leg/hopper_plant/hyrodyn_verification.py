import hyrodyn
import numpy as np
import os
from hopper_plant import HopperPlant
import random
import matplotlib.pyplot as plt
def main():
    """Return the print-outs as a dictionary
    ----- Parameters -----
    active_joints, independent_joints, spanning_tree_joints : lists 
    lists of the robot's joints
    Returns
    dict_jointnames : dictionary
    dictionary of the joints of the robot
    """
    # Path to URDF and Submechanism.yml files
    path_to_urdf = "../model/with_rails/urdf/v7.urdf"
    path_to_submechamisms = "../model/with_rails/urdf/submechanisms.yml"

    # Load the robot model in HyRoDyn
    robot = hyrodyn.RobotModel(path_to_urdf, path_to_submechamisms)
    differences = []
    for i in range(100):
        q0 = random.random()
        q1 = random.uniform(0,np.pi)
        q2 = random.uniform(0,np.pi)
        dq0 = random.random()
        dq1 = random.random()
        dq2 = random.random()
        ddq0 = random.random()
        ddq1 = random.random()
        ddq2 = random.random()
        
        robot.y = np.array([q0,q1,q2])
        robot.yd = np.array([dq0,dq1,dq2])
        robot.ydd = np.array([ddq0,ddq1,ddq2])
        # Inverse dynamics
        robot.calculate_inverse_dynamics()
        
        p = HopperPlant(Izz=[0.0014078, 0.00012493], com1=[0.056105, 1.0785E-05], com2=[0.096585, 9.8958E-09], link_length=[0.205, 0.25])
        # p = HopperPlant(Izz=[0.0014078, 0.00012493], com1=[0.056105, 0.039497], com2=[0.096585, -6.9822E-05], link_length=[0.205, 0.25])
        idyn = p.inverse_dynamics(q1, q2, dq1, dq2, ddq0, ddq1, ddq2)
        print("Inverse dynamics hyrodyn: ", robot.Tau_actuated)
        
        print("Inverse dynamics plant:   ",idyn.T)
        differences.append(robot.Tau_actuated-idyn.T)
        print("Difference:               ", robot.Tau_actuated-idyn.T)
    
    print("Average Differences: \n",sum(differences)/len(differences))
    
if __name__ == '__main__':
    # Call the main function
    main()
