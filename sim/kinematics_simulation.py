import os
import sys
import time
import numpy as np

import argparse

import pybullet as p
import pybullet_data

sys.path.append("../hopping_leg")

from hopping_leg.plant.hopper import HopperPlant
from hopping_leg.controllers import PD_controller
from hopping_leg.state_estimation import StateEstimation


urdf = lambda model: f"../hopping_leg/model/leg/{model}"

result = lambda dest: f"../sim/results/{dest}"


class SimulationTracker():

    def __init__(self, simulation, se = None,  saveRecording=False, filename="pybulletRecording.mp4"):
        """Stores simulation data and creates plots and csv files.

        Args:
            simulation (class): HopperSimulation instance.
            se (class, optional): StateEstimation instance. Defaults to None.
            saveRecording (bool, optional): Save video of the pybullet simulation. Defaults to False.
            filename (str, optional): Filename and path for the simulation video. Has to end with ".mp4". Defaults to "pybulletRecording.mp4".
        """
        if saveRecording:
            if not os.path.exists(os.path.dirname(filename)):
                os.mkdir(os.path.dirname(filename))
            p.startStateLogging(p.STATE_LOGGING_VIDEO_MP4, filename)
        self.simulation = simulation
        self.bodyUniqueId = self.simulation.robot
        self.jointIndices = self.simulation.jointIndices
        self.groundUniqueId = self.simulation.planeID
        self.plant = simulation.plant
        self.se = se
        # endeffector position
        self.x = []
        self.y = []
        self.x_desired = []
        self.y_desired = []
        # endeffector velocity
        self.xd = []
        self.yd = []
        self.xd_desired = []
        self.yd_desired = []
        # base vertical acceleration
        self.zdd = []
        # base height
        self.h = []
        # time
        self.t = []
        # joint angles
        self.q1 = []
        self.q2 = []
        self.q1_desired = []
        self.q2_desired = []
        # joint angles limits
        self.q1min = -2.3
        self.q1max = 2.3
        # self.q2min = None
        # self.q2max = None
        # joint velocities
        self.q1d = []
        self.q2d = []
        self.q1d_desired = []
        self.q2d_desired = []
        # joint velocity limits
        self.q1dmin = -38.2
        self.q1dmax = 38.2
        self.q2dmin = -38.2
        self.q2dmax = 38.2
        # joint torques
        self.tau1 = []
        self.tau2 = []
        self.tau1_desired = []
        self.tau2_desired = []
        # joint torque limits (peak)
        self.tau1min_peak = -12
        self.tau1max_peak = 12
        self.tau2min_peak = -12
        self.tau2max_peak = 12
        # joint torque limits (continuous)
        self.tau1min_cont = -6
        self.tau1max_cont = 6
        self.tau2min_cont = -6
        self.tau2max_cont = 6
        # ground reaction forces
        self.Fx = []
        self.Fy = []
        self.Fz = []
        self.Fx_desired = []
        self.Fy_desired = []
        self.Fz_desired = []
        # Base position
        self.basepos = []
        self.basepos_desired = []
        self.basepos_estimated = []
        self.phase =  []
        self.height_factor = []

    def update(self, tau1=None, tau2=None):
        """Save measured data of the current timestep.

        Args:
            tau1 ([type], optional): For torque control, the commanded torque has to be added here. Defaults to None.
            tau2 ([type], optional): For torque control, the commanded torque has to be added here. Defaults to None.
        """
        jointpos = [j[0] for j in p.getJointStates(
            self.bodyUniqueId, self.jointIndices)]
        jointvel = [j[1] for j in p.getJointStates(
            self.bodyUniqueId, self.jointIndices)]
        jointtorque = [j[3] for j in p.getJointStates(
            self.bodyUniqueId, self.jointIndices)]
        self.q1.append(jointpos[0])
        self.q2.append(jointpos[1])
        self.q1d.append(jointvel[0])
        self.q2d.append(jointvel[1])
        if tau1 is None:
            self.tau1.append(jointtorque[0])
        else:  # for torque control
            self.tau1.append(tau1)
        if tau2 is None:
            self.tau2.append(jointtorque[1])
        else:
            self.tau2.append(tau2)
        x, y = self.plant.forward_kinematics(jointpos[0], jointpos[1])
        self.x.append(x)
        self.y.append(y)
        xd, yd = self.plant.forward_velocity(
            jointpos[0], jointpos[1], jointvel[0], jointvel[1])
        self.xd.append(xd)
        self.yd.append(yd)
        if self.se is not None:
            zdd = self.se.get_vertical_acceleration()
            self.zdd.append(zdd)
            h = self.se.get_height()
            self.h.append(h)
        self.t.append(self.simulation.curtime)
        Fn, Fy, Fz = self.get_reaction_forces()
        self.Fx.append(Fn)
        self.Fy.append(Fy)
        self.Fz.append(Fz)
        if self.simulation.rail:
            self.basepos.append(p.getJointState(self.bodyUniqueId, 0)[0])
        else:
            self.basepos.append(
                p.getBasePositionAndOrientation(self.bodyUniqueId)[0][2])

    def get_reaction_forces(self):
        """Read reaction forces from pybullet.

        Returns:
            tuple: Fx, Fy, Fz
        """
        cp = p.getContactPoints(
            self.bodyUniqueId, self.groundUniqueId, self.simulation.jointIndices[1], -1)
        Fn = 0
        Fy = 0
        Fz = 0
        for i in range(len(cp)):
            Fn += cp[i][9]
            Fy += cp[i][12]
            Fz += cp[i][10]
        return Fn, Fy, Fz

    def write_csv(self):
        """
        Write (almost?) all data in csv file
        """
        names = ["time","x","y","x_desired","y_desired","xd","yd","xd_desired","yd_desired","zdd","h","q1","q2","q1_desired","q2_desired","q1d","q2d","q1d_desired","q2d_desired","tau1","tau2","tau1_desired","tau2_desired","Fx","Fy","Fz","Fx_desired","Fy_desired","Fz_desired","basepos","basepos_desired"]
        vars = [self.t,self.x,self.y,self.x_desired,self.y_desired,self.xd,self.yd,self.xd_desired,self.yd_desired,self.zdd,self.h,self.q1,self.q2,self.q1_desired,self.q2_desired,self.q1d,self.q2d,self.q1d_desired,self.q2d_desired,self.tau1,self.tau2,self.tau1_desired,self.tau2_desired,self.Fx,self.Fy,self.Fz,self.Fx_desired,self.Fy_desired,self.Fz_desired,self.basepos,self.basepos_desired]
        write_names = []
        write_vars = []
        for i in range(len(vars)):
            if vars[i]:
                write_names.append(names[i])
                write_vars.append(vars[i])    
        with open(result("results.csv"), "w+") as f:
            # f.writelines("time,x,y,x_desired,y_desired,xd,yd,xd_desired,yd_desired,zdd,h,q1,q2,q1_desired,q2_desired,q1d,q2d,q1d_desired,q2d_desired,tau1,tau2,tau1_desired,tau2_desired,Fx,Fy,Fz,Fx_desired,Fy_desired,Fz_desired,basepos,basepos_desired\n")
            f.write(",".join(write_names)+"\n")
            for i in range(len(self.t)):
                # f.write(",".join([self.t[i],self.x[i],self.y[i],self.x_desired[i],self.y_desired[i],self.xd[i],self.yd[i],self.xd_desired[i],self.yd_desired[i],self.zdd[i],self.h[i],self.q1[i],self.q2[i],self.q1_desired[i],self.q2_desired[i],self.q1d[i],self.q2d[i],self.q1d_desired[i],self.q2d_desired[i],self.tau1,self.tau2[i],self.tau1_desired[i],self.tau2_desired[i],self.Fx[i],self.Fy[i],self.Fz[i],self.Fx_desired[i],self.Fy_desired[i],self.Fz_desired[i],self.basepos[i],self.basepos_desired[i]])+"\n")
                f.write(",".join([str(v[i]) for v in write_vars])+"\n") 
    
    def trajectory_export(self):
        """
        Export csv file of measured data, compatible with trajectory replay script.
        """
        names = ["time","shoulder_pos","elbow_pos","shoulder_pos_des","elbow_pos_des","shoulder_vel","elbow_vel","shoulder_vel_des","elbow_vel_des","shoulder_torque","elbow_torque","shoulder_torque_des","elbow_torque_des"]
        vars = [self.t,self.q1,self.q2,self.q1_desired,self.q2_desired,self.q1d,self.q2d,self.q1d_desired,self.q2d_desired,self.tau1,self.tau2,self.tau1_desired,self.tau2_desired]
        write_names = []
        write_vars = []
        for i in range(len(vars)):
            if vars[i]:
                write_names.append(names[i])
                write_vars.append(vars[i])    
        with open(result("trajectoryexport.csv"), "w+") as f:
            # f.writelines("time,x,y,x_desired,y_desired,xd,yd,xd_desired,yd_desired,zdd,h,q1,q2,q1_desired,q2_desired,q1d,q2d,q1d_desired,q2d_desired,tau1,tau2,tau1_desired,tau2_desired,Fx,Fy,Fz,Fx_desired,Fy_desired,Fz_desired,basepos,basepos_desired\n")
            f.write(",".join(write_names)+"\n")
            for i in range(len(self.t)):
                # f.write(",".join([self.t[i],self.x[i],self.y[i],self.x_desired[i],self.y_desired[i],self.xd[i],self.yd[i],self.xd_desired[i],self.yd_desired[i],self.zdd[i],self.h[i],self.q1[i],self.q2[i],self.q1_desired[i],self.q2_desired[i],self.q1d[i],self.q2d[i],self.q1d_desired[i],self.q2d_desired[i],self.tau1,self.tau2[i],self.tau1_desired[i],self.tau2_desired[i],self.Fx[i],self.Fy[i],self.Fz[i],self.Fx_desired[i],self.Fy_desired[i],self.Fz_desired[i],self.basepos[i],self.basepos_desired[i]])+"\n")
                f.write(",".join([str(v[i]) for v in write_vars])+"\n") 
    
    def create_plots(self, coordinates=True, trajectory=True,
                     ee_velocity=True, joint_angles=True,
                     acceleration=True,
                     joint_velocities=True, joint_efforts=True,
                     ground_reaction_forces=True, base_position=True,
                     save_figures=False):
        """Create plots of saved data.

        Args:
            coordinates (bool, optional): Create xy position plot. Defaults to True.
            trajectory (bool, optional): Create trajectory in xy cordinates. Defaults to True.
            ee_velocity (bool, optional): Plot cartesian ee velocities. Defaults to True.
            joint_angles (bool, optional): Plot joint angles. Defaults to True.
            acceleration (bool, optional): Plot accelerations from state estimation. Defaults to True.
            joint_velocities (bool, optional): Plot joint velocities. Defaults to True.
            joint_efforts (bool, optional): Plot joint efforts. Defaults to True.
            ground_reaction_forces (bool, optional): Plot ground reaction forces. Defaults to True.
            base_position (bool, optional): Plot base position height. Defaults to True.
            save_figures (bool, optional): Save figures instead of plotting them. Defaults to False.
        """

        from matplotlib import pyplot as plt
        if coordinates:
            plt.figure("Cartesian Position")
            plt.plot(self.t, self.x, "b-", label="x")
            plt.plot(self.t, self.y, "g-", label="y")
            if self.x_desired:
                plt.plot(self.t, self.x_desired, "b-.", label="x_desired")
                plt.plot(self.t, self.y_desired, "g-.", label="y_desired")
            plt.xlabel("Time")
            plt.ylabel("Cartesian Position (m)")
            plt.legend(loc='upper right')
            if self.phase:
                ax = plt.gca()
                ax.pcolorfast((self.t[0],self.t[-1]),ax.get_ylim(),np.array(self.phase)[np.newaxis],cmap="Greens", alpha=.3)
            if save_figures:
                plt.savefig(result("position.png"))
                plt.close()
        if trajectory:
            plt.figure("Endeffector trajectory")
            plt.plot(self.y, self.x, "b-", label="measured trajectory")
            if self.x_desired:
                plt.plot(self.y_desired, self.x_desired,
                         "g-", label="desired trajectory")
            plt.gca().invert_yaxis()
            plt.xlabel("y (m)")
            plt.ylabel("x (m)")
            plt.axis('equal')
            plt.legend(loc='upper right')
            if save_figures:
                plt.savefig(result("trajectory.png"))
                plt.close()
        if ee_velocity:
            plt.figure("Cartesian Velocity")
            plt.plot(self.t, self.xd, "b-", label="xd")
            plt.plot(self.t, self.yd, "g-", label="yd")
            if self.xd_desired:
                plt.plot(self.t, self.xd_desired, "b-.", label="xd_desired")
                plt.plot(self.t, self.yd_desired, "g-.", label="yd_desired")
            plt.xlabel("Time")
            plt.ylabel("Cartesian Velocity (m/s)")
            plt.legend(loc='upper right')
            if self.phase:
                ax = plt.gca()
                ax.pcolorfast((self.t[0],self.t[-1]),ax.get_ylim(),np.array(self.phase)[np.newaxis],cmap="Greens", alpha=.3)
            if save_figures:
                plt.savefig(result("velocity.png"))
                plt.close()
        if acceleration:
            if len(self.zdd) > 0:
                plt.figure("Vertical Acceleration")
                plt.plot(self.t, self.zdd, "b-", label="zdd")
                plt.xlabel("Time")
                plt.ylabel("Vertical Acceleration (m/s^2)")
                plt.legend(loc='upper right')
                if self.phase:
                    ax = plt.gca()
                    ax.pcolorfast((self.t[0],self.t[-1]),ax.get_ylim(),np.array(self.phase)[np.newaxis],cmap="Greens", alpha=.3)
                if save_figures:
                    plt.savefig(result("acceleration.png"))
                    plt.close()
                plt.figure("Estimated Base Height")
                plt.plot(self.t, self.h, "b-", label="estimated height")
                plt.plot(self.t, self.basepos, "b-.", label="ground truth height")
                plt.xlabel("Time")
                plt.ylabel("Base height (m)")
                plt.legend(loc='upper right')
                if self.phase:
                    ax = plt.gca()
                    ax.pcolorfast((self.t[0],self.t[-1]),ax.get_ylim(),np.array(self.phase)[np.newaxis],cmap="Greens", alpha=.3)
                if save_figures:
                    plt.savefig(result("estimated_height.png"))
                    plt.close()
        if joint_angles:
            plt.figure("Joint angles")
            plt.plot(self.t, self.q1, "b-", label="q_hip")
            plt.plot(self.t, self.q2, "g-", label="q_knee")
            if self.q1_desired:
                plt.plot(self.t, self.q1_desired, "b-.", label="q_hip_desired")
                plt.plot(self.t, self.q2_desired,
                         "g-.", label="q_knee_desired")
            plt.plot(self.t, self.q1min*np.ones(len(self.t)),
                     "b--", label="q_hip_min")
            plt.plot(self.t, self.q1max*np.ones(len(self.t)),
                     "b--", label="q_hip_max")
            # plt.plot(self.t, self.q2min*np.ones(len(self.t)),
            #          "g--", label="q_knee_min")
            # plt.plot(self.t, self.q2max*np.ones(len(self.t)),
            #          "g--", label="q_knee_max")
            plt.xlabel("Time (s)")
            plt.ylabel("Joint Position (rad)")
            plt.legend(loc='upper right')
            if self.phase:
                ax = plt.gca()
                ax.pcolorfast((self.t[0],self.t[-1]),ax.get_ylim(),np.array(self.phase)[np.newaxis],cmap="Greens", alpha=.3)
            if save_figures:
                plt.savefig(result("joint_angles.png"))
                plt.close()
        if joint_velocities:
            plt.figure("Joint velocities")
            plt.plot(self.t, self.q1d, "b-", label="dq_hip")
            plt.plot(self.t, self.q2d, "g-", label="dq_knee")
            if self.q1d_desired:
                plt.plot(self.t, self.q1d_desired,
                         "b-.", label="qd_hip_desired")
                plt.plot(self.t, self.q2d_desired,
                         "g-.", label="qd_knee_desired")
            plt.plot(self.t, self.q1dmin*np.ones(len(self.t)),
                     "b--", label="dq_hip_min")
            plt.plot(self.t, self.q1dmax*np.ones(len(self.t)),
                     "b--", label="dq_hip_max")
            plt.plot(self.t, self.q2dmin*np.ones(len(self.t)),
                     "g--", label="dq_knee_min")
            plt.plot(self.t, self.q2dmax*np.ones(len(self.t)),
                     "g--", label="dq_knee_max")
            plt.xlabel("Time (s)")
            plt.ylabel("Joint Velocity (rad/s)")
            plt.legend(loc='upper right')
            if self.phase:
                ax = plt.gca()
                ax.pcolorfast((self.t[0],self.t[-1]),ax.get_ylim(),np.array(self.phase)[np.newaxis],cmap="Greens", alpha=.3)
            if save_figures:
                plt.savefig(result("joint_velocities.png"))
                plt.close()
        if joint_efforts:
            plt.figure("Joint efforts")
            plt.plot(self.t, self.tau1, "b-", label="tau_hip")
            plt.plot(self.t, self.tau2, "g-", label="tau_knee")
            plt.plot(self.t, self.tau1min_peak*np.ones(len(self.t)),
                     "b-.", label="tau_hip_min_peak")
            plt.plot(self.t, self.tau1max_peak*np.ones(len(self.t)),
                     "b-.", label="tau_hip_max_peak")
            plt.plot(self.t, self.tau2min_peak*np.ones(len(self.t)),
                     "g-.", label="tau_knee_min_peak")
            plt.plot(self.t, self.tau2max_peak*np.ones(len(self.t)),
                     "g-.", label="tau_knee_max_peak")
            plt.plot(self.t, self.tau1min_cont*np.ones(len(self.t)),
                     "b--", label="tau_hip_min_cont")
            plt.plot(self.t, self.tau1max_cont*np.ones(len(self.t)),
                     "b--", label="tau_hip_max_cont")
            plt.plot(self.t, self.tau2min_cont*np.ones(len(self.t)),
                     "g--", label="tau_knee_min_cont")
            plt.plot(self.t, self.tau2max_cont*np.ones(len(self.t)),
                     "g--", label="tau_knee_max_cont")
            plt.xlabel("Time (s)")
            plt.ylabel("Joint Torque (Nm)")
            plt.legend(loc='upper right')
            if self.phase:
                ax = plt.gca()
                ax.pcolorfast((self.t[0],self.t[-1]),ax.get_ylim(),np.array(self.phase)[np.newaxis],cmap="Greens", alpha=.3)
            if save_figures:
                plt.savefig(result("joint_efforts.png"))
                plt.close()
        if ground_reaction_forces:
            plt.figure("Reaction Forces")
            plt.plot(self.t, self.Fx, "-", label="Normal Force Fx")
            plt.plot(self.t, self.Fy, "-", label="Lateral Friction Fy")
            plt.plot(self.t, self.Fz, "-", label="Lateral Friction Fz")
            plt.xlabel("Time (s)")
            plt.ylabel("Reaction Forces (N)")
            plt.legend(loc='upper right')
            if self.phase:
                ax = plt.gca()
                ax.pcolorfast((self.t[0],self.t[-1]),ax.get_ylim(),np.array(self.phase)[np.newaxis],cmap="Greens", alpha=.3)
            if save_figures:
                plt.savefig(result("reaction_forces.png"))
                plt.close()
        if base_position:
            
            plt.figure("Base Position")
            plt.plot(self.t, self.basepos, "-", label="Base Position")
            ax = plt.gca()
            if self.basepos_desired:
                plt.plot(self.t, self.basepos_desired, "-", label="Desired Base Position")    
            if self.basepos_estimated:
                plt.plot(self.t, self.basepos_estimated, "-", label="Estimated Base Position")    
            if self.phase:
                ax.pcolorfast((self.t[0],self.t[-1]),ax.get_ylim(),np.array(self.phase)[np.newaxis],cmap="Greens", alpha=.3)
            plt.xlabel("Time (s)")
            plt.ylabel("Base Position (m)")
            plt.legend(loc='upper right')
            if self.height_factor:
                ax2 = ax.twinx()
                ax2.plot(self.t, self.height_factor, color="g", label="Height factor")
                ax2.set_ylabel("Height factor")
                plt.legend(loc='lower right')
            
            if save_figures:
                plt.savefig(result("base_position.png"))
                plt.close()

        if not save_figures:
            plt.show()


class HopperSimulation():

    def __init__(self, control_mode = 2, rail=True, drop_height=0.5, gravity=9.81, torque_limits=(6,6),long_leg = False, frequency = None):
        """Hopper simulation class which setups the environment and the correct physical parameters.

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
        self.rail = rail
        self.g = gravity
        if frequency is not None:
            self.dt = 1/frequency
        else:
            self.dt = 1. / 240.
        self.curtime = 0.0

        if self.rail:
            if long_leg:
                self.jointIndices = [2, 3]
                self.L1 = 0.205
                self.L2 = 0.25
                self.mass = 2.27295
                
                # load plant
                self.plant = HopperPlant(
                    mass = [0.91482,1.2148,0.14334], Izz=[0.0014078,0.00012493], 
                    com1=[0.056105,1.0785E-05], com2=[0.096585,9.8958E-09],
                    link_length=[self.L1, self.L2], gravity=self.g, torque_limits=torque_limits)
            
            else: # new rail
                self.jointIndices = [2, 3]
                self.L1 = 0.2
                self.L2 = 0.2 # .187?
                self.mass = 2.32358
                
                # load plant
                self.plant = HopperPlant(
                    mass = [0.91281,1.2804,0.13037], Izz=[0.0015899,6.3388E-05], 
                    com1=[0.059331, 1.3564E-05], com2=[0.078298, 1.088E-08],
                    link_length=[self.L1, self.L2], gravity=self.g, torque_limits=torque_limits)
        else:
            self.jointIndices = [1, 2]
            self.L1 = 0.2
            self.L2 = 0.2
            self.mass = 2.258899

            # load plant
            self.plant = HopperPlant(
                mass = [0.91482,1.2086,0.13548], Izz=[0.0013698,6.3774E-05], 
                com1=[0.054312,6.3584E-06], com2=[0.075345,1.0469E-08],
                link_length=[self.L1, self.L2], gravity=self.g, torque_limits=torque_limits)

        # Setup environment
        self.physicsClient = p.connect(p.GUI)
        # p.setPhysicsEngineParameter(enableFileCaching=0) # uncomment after changing urdf
        p.configureDebugVisualizer(
            p.COV_ENABLE_GUI, 0, physicsClientId=self.physicsClient)

        # Camera position
        if self.rail:
            p.resetDebugVisualizerCamera(cameraDistance=1.0, cameraYaw=0, cameraPitch=0,
                                         cameraTargetPosition=[0, 0, 0.25], physicsClientId=self.physicsClient)
        else:
            p.resetDebugVisualizerCamera(cameraDistance=1.0, cameraYaw=180, cameraPitch=0,
                                         cameraTargetPosition=[0, 0, 0.25], physicsClientId=self.physicsClient)

        p.setRealTimeSimulation(0, physicsClientId=self.physicsClient)

        # to include ground plane
        p.setAdditionalSearchPath(pybullet_data.getDataPath())

        self.planeID = p.loadURDF(
            "plane.urdf", [0, 0, 0], useFixedBase=0, physicsClientId=self.physicsClient)
        p.changeDynamics(self.planeID, -1, lateralFriction=60,
                         physicsClientId=self.physicsClient)  # TODO: get real values

        self.drop_height = drop_height
        
        # Rail
        if self.rail:

            p.setGravity(0, 0, -self.g, physicsClientId=self.physicsClient)

            # Load model
            if long_leg: # old Rail
                model= urdf("with_rails/urdf/v7.urdf")
            else: # new Rail
                model = urdf("with_rails2/urdf/vertical_test.SLDASM.urdf")
            self.robot = p.loadURDF(
                model, [0, 0, 0], useFixedBase=1, physicsClientId=self.physicsClient)
            
            # Set up simulation
            p.setTimeStep(self.dt, physicsClientId=self.physicsClient)

            # Initial stabilization loop
            while self.curtime < 1:
                self.curtime = self.curtime + self.dt
                p.setJointMotorControlArray(self.robot, self.jointIndices, p.POSITION_CONTROL, self.plant.inverse_kinematics(
                    0.24, 0, knee_direction=0), positionGains=[1.0, 1.0], physicsClientId=self.physicsClient)
                p.stepSimulation(physicsClientId=self.physicsClient)
                time.sleep(self.dt)
            
            p.resetJointState(self.robot, 0, self.drop_height,
                              physicsClientId=self.physicsClient)
            p.setJointMotorControl2(
                self.robot, 0, p.VELOCITY_CONTROL, 0, force=0, physicsClientId=self.physicsClient)
            
        # No rail
        else:

            # Load model
            model = urdf("without_rails/urdf/v6_new_joint_limits.urdf")
            self.robot = p.loadURDF(model, [
                                    0, 0, self.drop_height], useFixedBase=0, physicsClientId=self.physicsClient)
            
            # Add prismatic constraint for the base
            cid = p.createConstraint(
                self.robot, -1, -1, -1, p.JOINT_PRISMATIC, [0, 0, 1], [0, 0, 0], [0, 0, 0], physicsClientId=self.physicsClient)

            p.setGravity(0, 0, -self.g, physicsClientId=self.physicsClient)

        # Set up variables for control mode:
        self.controlmode = control_mode
        if self.controlmode == 0:  # position control
            if frequency is not None:
                self.dt = 1./frequency
            else:
                self.dt = 1. / 240.
            p.setTimeStep(self.dt, physicsClientId=self.physicsClient)
        elif self.controlmode == 1:  # velocity control
            if frequency is not None:
                self.dt = 1./frequency
            else:
                self.dt = 1. / 500.
            p.setTimeStep(self.dt, physicsClientId=self.physicsClient)
        elif self.controlmode == 2:  # torque control
            if frequency is not None:
                self.dt = 1./frequency
            else:
                self.dt = 1. / 1000.
            p.setTimeStep(self.dt, physicsClientId=self.physicsClient)
            # deactivate joints to enable torque control
            p.setJointMotorControlArray(
                self.robot, self.jointIndices, p.VELOCITY_CONTROL, forces=[0, 0], physicsClientId=self.physicsClient)

    def ground_contact(self):
        """Check whether leg has ground contact.

        Returns:
            bool: Contact True/False
        """
        if len(p.getContactPoints(self.robot, self.planeID, self.jointIndices[1], -1, physicsClientId=self.physicsClient)) > 0:
            return 1
        else:
            return 0

    def get_current_state(self):
        """Create dictionary with current state

        Returns:
            dict: current state.
        """
        jointpos = [j[0] for j in p.getJointStates(
            self.robot, self.jointIndices, physicsClientId=self.physicsClient)]
        jointvel = [j[1] for j in p.getJointStates(
            self.robot, self.jointIndices, physicsClientId=self.physicsClient)]
        jointtorque = [j[3] for j in p.getJointStates(
            self.robot, self.jointIndices, physicsClientId=self.physicsClient)]
        cpos = self.plant.forward_kinematics(*jointpos)
        cvel = self.plant.forward_velocity(*jointpos, *jointvel)
        state = {"q1": jointpos[0], "q2": jointpos[1], "q1d": jointvel[0], "q2d": jointvel[1],
                 "tau1": jointtorque[0], "tau2": jointtorque[1], "x": cpos[0], "y": cpos[1],
                 "xd": cvel[0], "yd": cvel[1]}
        return state

    def get_state_vector(self):
        """Get state vector

        Returns:
            list: [q1,q2,sq1,dq2]
        """
        jointpos = [j[0] for j in p.getJointStates(
            self.robot, self.jointIndices, physicsClientId=self.physicsClient)]
        jointvel = [j[1] for j in p.getJointStates(
            self.robot, self.jointIndices, physicsClientId=self.physicsClient)]
        X = [jointpos[0], jointpos[1], jointvel[0], jointvel[1]]
        return X

    def step_time(self):
        """Step simulation

        Returns:
            float: Current simulaiton time
        """
        self.curtime = self.curtime + self.dt
        return self.curtime

    def sleep(self):
        """Sleep for the lenght of one timestep.
        """
        time.sleep(self.dt)


if __name__ == "__main__":
    # Create the parser
    parser = argparse.ArgumentParser(
        description="Simulation of hopping leg based on quasi-direct drives",
        formatter_class=RawTextHelpFormatter)
    parser.add_argument("--position_control", 
                        "-p", 
                        action='store_true',
                        help="Use position control mode")
    parser.add_argument("--velocity_control", 
                        "-v", 
                        action='store_true',
                        help="Use velocity control mode")
    parser.add_argument("--torque_control", 
                        "-t", 
                        action='store_true',
                        help="Use torque control mode")
    parser.add_argument("--save_results", 
                        "-s", 
                        action='store_true',
                        help="Save results to the results folder")
    parser.add_argument("--rail", 
                        "-r", 
                        action='store_true',
                        help="Use URDF with rail")

    # Execute the parse_args method
    args, unknown = parser.parse_known_args()

    results_folder = ""
    if args.position_control:
        results_folder = "position_control"
        control_mode = 0

    if args.velocity_control:
        results_folder = "velocity_control"
        control_mode = 1

    if args.torque_control:
        results_folder = "torque_control"
        control_mode = 2

    # setup simulation
    sim = HopperSimulation(control_mode=control_mode, rail=args.rail)
    plant = sim.plant
    
    # Initialize state estimation and start virtual IMU streaming
    se = StateEstimation(sim.robot, sim.planeID, plant, sim.dt,  imu_link=0)
    se.start_imu_scan()
    se.start_state_estimation()

    tracker = SimulationTracker(sim, 
                                se,     
                                saveRecording=args.save_results,
                                filename=result(results_folder + "/pybullet_recording.mp4"))
    
    # Initialize Controller
    c = PD_controller(plant)
    count = 0
    while sim.curtime < 7:
        count = count + 1
        sim.step_time()

        # with rail
        if 0:# sim.rail:
            # move leg up and down (position control)
            if args.position_control:
                offset = 0.3
                A = 0.10
                omega = 8
                # x_desired = offset+A*np.sin(omega*sim.curtime)
                x_desired = offset+A*(0.5*np.sin(omega*sim.curtime)**2+np.cos(omega*sim.curtime)+A*(np.sin(omega*sim.curtime)*np.cos(omega*sim.curtime)-np.sin(omega*sim.curtime))/3)
                y_desired = 0.0
                tracker.x_desired.append(x_desired)
                tracker.y_desired.append(y_desired)
                q1_desired, q2_desired = plant.inverse_kinematics(
                    x_desired, 0, knee_direction=0)
                tracker.q1_desired.append(q1_desired)
                tracker.q2_desired.append(q2_desired)
                # xd_desired = omega*A*np.cos(omega*sim.curtime)
                xd_desired = A*omega/3*(-3*np.sin(omega*sim.curtime)+5*np.cos(omega*sim.curtime)*np.sin(omega*sim.curtime)-2*np.cos(omega*sim.curtime)*np.sin(omega*sim.curtime)**3-4*np.cos(omega*sim.curtime)**2*np.sin(omega*sim.curtime)+2*np.cos(omega*sim.curtime)**3*np.sin(omega*sim.curtime)+2*np.sin(omega*sim.curtime)**3)
                yd_desired = 0.0
                tracker.xd_desired.append(xd_desired)
                tracker.yd_desired.append(yd_desired)
                q1d_desired, q2d_desired = plant.inverse_velocity(
                    q1_desired, q2_desired, xd_desired, yd_desired)
                tracker.q1d_desired.append(q1d_desired)
                tracker.q2d_desired.append(q2d_desired)
                tau1_desired = 6
                tau2_desired = 6
                p.setJointMotorControlArray(
                    sim.robot, sim.jointIndices, p.POSITION_CONTROL, [q1_desired, q2_desired], 
                    targetVelocities=[q1d_desired, q2d_desired],
                    positionGains=[3, 3], 
                    velocityGains=[10, 10], forces=[tau1_desired, tau2_desired])

                tracker.update()
                p.stepSimulation()
                sim.sleep()

            if args.velocity_control:  # heuristic
                state = sim.get_current_state()
                a = 20  # acceleration
                v = count * sim.dt * a  # cartesian velocity in x dir
                if sim.ground_contact():  # and state["x"] < .35:
                    # y position control
                    yd = -10 * np.sin(state["y"]*np.pi/.8)**3
                    velocity = plant.inverse_velocity(
                        state["q1"], state["q2"], -v, yd)
                    # avoid movements larger than velocity limits
                    while np.abs(velocity[0]) > 38.2 or np.abs(velocity[1]) > 38.2 and count > 0:
                        count -= 5
                        v = count * sim.dt * a
                        velocity = plant.inverse_velocity(
                            state["q1"], state["q2"], -v, yd)
                    forces = 6 + 6*np.sin(count*sim.dt*1000*np.pi/(2*100))**8
                    p.setJointMotorControlArray(
                        sim.robot, sim.jointIndices, p.VELOCITY_CONTROL, 
                        targetVelocities=velocity, forces=[forces, forces], 
                        velocityGains=[.2, .2])
                # position control during flight phase
                else:
                    count = -50
                    p.setJointMotorControlArray(
                        sim.robot, sim.jointIndices, p.POSITION_CONTROL, 
                        plant.inverse_kinematics(.3, -0.1, 1), positionGains=[.1, .4], 
                        forces=[6, 6], targetVelocities=[20, 20])
                tracker.update()
                p.stepSimulation()
                sim.sleep()

            # cartesian stiffness control
            if args.torque_control:
                pos = [0.24+0.15*np.sin(20*sim.curtime), 0]
                vel = [0.15*20*np.cos(20*sim.curtime), 0]
                if sim.ground_contact():
                    tau1, tau2 = c.combined_stiffness(
                        sim.get_state_vector(), Kpc=[1000, 1000], p_d=pos, Kdc=[100, 250], pd_d=vel)

                else:
                    tau1, tau2 = c.combined_stiffness(
                        sim.get_state_vector(), Kpc=[10000, 10000], p_d=pos, Kdc=[500, 500], pd_d=vel)
                # clip limits
                tau1_lim = 6
                tau2_lim = 6
                tau1 = tau1_lim if tau1 > tau1_lim else tau1
                tau2 = tau2_lim if tau2 > tau2_lim else tau2
                tau1 = -tau1_lim if tau1 < -tau1_lim else tau1
                tau2 = -tau2_lim if tau2 < -tau2_lim else tau2
                p.setJointMotorControlArray(
                    sim.robot, sim.jointIndices, p.TORQUE_CONTROL, forces=[tau1, tau2])
                # Step simulation
                tracker.update(tau1, tau2)
                p.stepSimulation()
                sim.sleep()

        # no rail
        else:
            # move leg up and down (position control)
            if args.position_control:
                offset = 0.24
                A = 0.1
                omega = 10
                # x_desired = offset+A*np.sin(omega*sim.curtime)
                x_desired = offset+A*(0.5*np.sin(omega*sim.curtime)**2+np.cos(omega*sim.curtime)+A*(np.sin(omega*sim.curtime)*np.cos(omega*sim.curtime)-np.sin(omega*sim.curtime))/3)
                y_desired = 0.0
                tracker.x_desired.append(x_desired)
                tracker.y_desired.append(y_desired)
                q1_desired, q2_desired = plant.inverse_kinematics(
                    x_desired, 0, knee_direction=0)
                tracker.q1_desired.append(q1_desired)
                tracker.q2_desired.append(q2_desired)
                # xd_desired = omega*A*np.cos(omega*sim.curtime)
                xd_desired = A*omega/3*(-3*np.sin(omega*sim.curtime)+5*np.cos(omega*sim.curtime)*np.sin(omega*sim.curtime)-2*np.cos(omega*sim.curtime)*np.sin(omega*sim.curtime)**3-4*np.cos(omega*sim.curtime)**2*np.sin(omega*sim.curtime)+2*np.cos(omega*sim.curtime)**3*np.sin(omega*sim.curtime)+2*np.sin(omega*sim.curtime)**3)
                yd_desired = 0.0
                tracker.xd_desired.append(xd_desired)
                tracker.yd_desired.append(yd_desired)
                q1d_desired, q2d_desired = plant.inverse_velocity(
                    q1_desired, q2_desired, xd_desired, yd_desired)
                tracker.q1d_desired.append(q1d_desired)
                tracker.q2d_desired.append(q2d_desired)
                tau1_desired = 6
                tau2_desired = 6
                p.setJointMotorControlArray(
                    sim.robot, sim.jointIndices, p.POSITION_CONTROL, [q1_desired, q2_desired], 
                    targetVelocities=[q1d_desired, q2d_desired], positionGains=[0.1, .1], 
                    velocityGains=[1, 1], forces=[tau1_desired, tau2_desired])

                tracker.update()
                p.stepSimulation()
                sim.sleep()

            if args.velocity_control:  # heuristic
                state = sim.get_current_state()
                a = 20  # acceleration
                v = count * sim.dt * a  # cartesian velocity in x dir
                if sim.ground_contact():  # and state["x"] < .35:
                    # y position control
                    yd = -10 * np.sin(state["y"]*np.pi/.8)**3
                    velocity = plant.inverse_velocity(
                        state["q1"], state["q2"], -v, yd)
                    # avoid movements larger than velocity limits
                    while np.abs(velocity[0]) > 38.2 or np.abs(velocity[1]) > 38.2 and count > 0:
                        count -= 5
                        v = count * sim.dt * a
                        velocity = plant.inverse_velocity(
                            state["q1"], state["q2"], -v, yd)
                    forces = 6 + 6*np.sin(count*sim.dt*1000*np.pi/(2*100))**8
                    p.setJointMotorControlArray(
                        sim.robot, sim.jointIndices, p.VELOCITY_CONTROL, 
                        targetVelocities=velocity, forces=[forces, forces], 
                        velocityGains=[.2, .2])
                # position control during flight phase
                else:
                    count = -50
                    p.setJointMotorControlArray(
                        sim.robot, sim.jointIndices, 
                        p.POSITION_CONTROL, plant.inverse_kinematics(.3, -0.1, 1), 
                        positionGains=[.1, .4], forces=[6, 6], targetVelocities=[20, 20])
                tracker.update()
                p.stepSimulation()
                sim.sleep()

            # cartesian stiffness control
            if args.torque_control:
                pos = [0.24+0.15*np.sin(20*sim.curtime), 0]
                vel = [0.15*20*np.cos(20*sim.curtime), 0]
                if sim.ground_contact():
                    tau1, tau2 = c.combined_stiffness(
                        sim.get_state_vector(), Kpc=[1000, 1000], p_d=pos, Kdc=[100, 250], pd_d=vel)

                else:
                    tau1, tau2 = c.combined_stiffness(
                        sim.get_state_vector(), Kpc=[10000, 10000], p_d=pos, Kdc=[500, 500], pd_d=vel)
                # clip limits
                tau1_lim = 12
                tau2_lim = 12
                tau1 = tau1_lim if tau1 > tau1_lim else tau1
                tau2 = tau2_lim if tau2 > tau2_lim else tau2
                tau1 = -tau1_lim if tau1 < -tau1_lim else tau1
                tau2 = -tau2_lim if tau2 < -tau2_lim else tau2
                p.setJointMotorControlArray(
                    sim.robot, sim.jointIndices, p.TORQUE_CONTROL, forces=[tau1, tau2])
                # Step simulation
                tracker.update(tau1, tau2)
                p.stepSimulation()
                sim.sleep()

    p.disconnect()
    tracker.create_plots(save_figures=args.save_results)
    if args.save_results:
        tracker.write_csv()
