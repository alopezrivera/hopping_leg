import os
import sys
import time
import numpy as np

import pybullet as p
import pybullet_data

sys.path.append("../hopping_leg")

from hopping_leg.plant.hopper import HopperPlant
from hopping_leg.controllers import PD_controller, GainScheduling
from hopping_leg.state_estimation import StateEstimation

from sim.kinematics_simulation import SimulationTracker, HopperSimulation

from hopping_leg.utilities.cli import parse


rail = 0
long_leg = 1


class GainSchedulingControl(PD_controller):
    TOUCHDOWN = "TOUCHDOWN"
    FLIGHT = "FLIGHT"
    LIFTOFF = "LIFTOFF"
    
    def __init__(self, plant, state_estimation) -> None:
        """Gain scheduling controler

        Args:
            plant (class): hopper plant. 
            state_estimation (class): State estimation.
        """
        self.plant = plant
        self.phase = self.FLIGHT
        self.se = state_estimation # state estimation
        
    def do_control(self, state):
        if self.phase == self.FLIGHT:
            tau1, tau2 = self.combined_stiffness(state,Kpj = (100,600), Kdj = (4,4), q_d = self.plant.inverse_kinematics(.30,0), Kdc=(100,20))
            
            if self.se.get_contact():
                self.phase = self.TOUCHDOWN          

        if self.phase == self.TOUCHDOWN:
            tau1, tau2 = self.combined_stiffness(state,Kpj = (100,600), q_d = self.plant.inverse_kinematics(.30,0), Kdc=(100,40))
            
            if self.plant.forward_velocity(*state)[0]>=0: 
                self.phase = self.LIFTOFF
            
        elif self.phase == self.LIFTOFF:
            tau1, tau2 = self.combined_stiffness(state,Kpj = (1000,1000), q_d = self.plant.inverse_kinematics(.33,0), Kdc=(20,20),pd_d = (0,0))
            
            if not self.se.get_contact():
                self.phase = self.FLIGHT
        
        return tau1,tau2



class EnergyShapingControl(PD_controller):
    TOUCHDOWN = "TOUCHDOWN"
    FLIGHT = "FLIGHT"
    LIFTOFF = "LIFTOFF"
    
    def __init__(self, plant, state_estimation, modes = ["j"]) -> None:
        """Energy shaping control with and without doing backflips.
        Shapes the energy inbetween jumps to reach a desired jumping height.

        Args:
            plant (class): hopper plant. 
            state_estimation (class): State estimation.
            modes (list, optional): List of strings with jumping modes which are done after each another.
                At the end of the list ist starts with the first element again.
                Posible modes are: 
                    "j": normal jump.
                    "b": backflip.
                    "f": forward_flip.
                    "bc": backflip with knee directionchange.
                    "fc": forwardflip with knee directionchange.
                    "bd": double backflip.
                    "fd": double forwardflip.
                Defaults to ["j"].
        """
        self.plant = plant
        self.phase = self.TOUCHDOWN
        self.se = state_estimation # state estimation
        self.last_jumping_height = 0
        self.start_liftoff_pos = 0
        self.end_liftoff_pos = 0
        self.last_liftoff_way = .1
        self.height_factor = 1
        self.nth_jump = 0
        self.flight_time = 0
        self.last_flight_time = None
        self.dt = 1/800
        self.time = 0
        self.pose_at_liftoff = None
        self.pose_at_touchdown = None
        self.flip_count = 0
        self.knee_direction = 0
        self.add = 0
        self.savety_time = .1
        self.mode_list = modes
        self.mode = 'j'
        self.liftoff_vel = 0
        self.time_at_liftoff = 0
        
        
    def estimate_last_height(self, x0, x1, t):
        """Calculate last jumping height from jumping time and liftoff and touchdown pose
        
        Args:
            x0 (flaot): liftoff pose
            x1 (float): touchdown pose
            t (float): jumping time

        Returns:
            float: estimated jumping height
        """
        g = -self.plant.g
        v0 = (x1 - x0)/t - g/2 * t
        t1 = -v0/g
        h = g/2 * t1**2 + v0 * t1 + x0
        return h
    
    def estimate_state(self,state):
        """Estimate current base height by simple integration during flight phase and forward kinematics otherwise.

        Args:
            state (list): State vector [q1,q2,dq1,dq2]

        Returns:
            float: Estimated base height.
        """
              
        if self.phase != self.FLIGHT:
            return self.plant.forward_kinematics(*state[:2])[0]
        t = self.time
        t0 = self.time_at_liftoff
        x0 = self.end_liftoff_pos
        v0 = self.liftoff_vel
        g = -self.plant.g
        x = +g/2 *(t**2-t0**2) - g*t0*(t-t0) + v0*(t-t0) + x0
        return x
    
    def calc_trajectory(self, pose, mode):
        """Calculated the trajectorys for all kinds of backflips and adds them to the current inverse kinematics solution.

        Args:
            pose (list): inverse kinematics solution
            mode (str): jumping mode to perform.
                Posible modes are: 
                    "j": normal jump.
                    "b": backflip.
                    "f": forward_flip.
                    "bc": backflip with knee directionchange.
                    "fc": forwardflip with knee directionchange.
                    "bd": double backflip.
                    "fd": double forwardflip.

        Returns:
            tuple: Desired state vector (q1,q2,dq1,dq2)
        """
        if mode == 'j': # jump
            return (*pose,0,0)
        elif mode == 'b': # backflip
            q1 = pose[0] - 2*np.pi + (- np.pi * np.cos(np.pi*self.flight_time/(self.last_flight_time-self.savety_time)) +  np.pi)
            q2 = pose[1]
            dq1 = -np.pi **2 / (self.last_flight_time-self.savety_time) * np.sin(np.pi*self.flight_time/(self.last_flight_time-self.savety_time))
            dq2 = 0
        elif mode == 'bd': # double backflip
            q1 = pose[0] - 4*np.pi + (- 2*np.pi * np.cos(np.pi*self.flight_time/(self.last_flight_time-self.savety_time)) +  2*np.pi)
            q2 = pose[1]
            dq1 = -2*np.pi **2 / (self.last_flight_time-self.savety_time) * np.sin(np.pi*self.flight_time/(self.last_flight_time-self.savety_time))
            dq2 = 0
        
        elif mode == 'bc': # backflip and change knee direction
            q1 = - pose[0] - 2*np.pi + (- (abs(pose[0]) + np.pi) * np.cos(np.pi*self.flight_time/(self.last_flight_time-self.savety_time)) +  np.pi)
            q2 = - pose[1]*np.cos(np.pi*self.flight_time/(self.last_flight_time-self.savety_time))
            dq1 = -(np.pi-abs(pose[0])) *np.pi / (self.last_flight_time-self.savety_time) * np.sin(np.pi*self.flight_time/(self.last_flight_time-self.savety_time))
            dq2 = pose[1]*np.sin(np.pi*self.flight_time/(self.last_flight_time-self.savety_time)) *(np.pi/(self.last_flight_time-self.savety_time))
        elif mode == 'f': # forwardflip
            q1 = pose[0] + 2*np.pi - (-np.pi * np.cos(np.pi*self.flight_time/(self.last_flight_time-self.savety_time)) +  np.pi)
            q2 = pose[1]
            dq1 = np.pi **2 / (self.last_flight_time-self.savety_time) * np.sin(np.pi*self.flight_time/(self.last_flight_time-self.savety_time))
            dq2 = 0
        elif mode == 'fc': # forwardflip and change knee direction
            q1 = - pose[0] - 2*np.pi - (- (abs(pose[0]) + np.pi) * np.cos(np.pi*self.flight_time/(self.last_flight_time-self.savety_time)) +  np.pi)
            q2 = - pose[1]*np.cos(np.pi*self.flight_time/(self.last_flight_time-self.savety_time))
            dq1 = (np.pi-abs(pose[0])) *np.pi / (self.last_flight_time-self.savety_time) * np.sin(np.pi*self.flight_time/(self.last_flight_time-self.savety_time))
            dq2 = pose[1]*np.sin(np.pi*self.flight_time/(self.last_flight_time-self.savety_time)) *(np.pi/(self.last_flight_time-self.savety_time))
        elif mode == 'fd': # double forward flip
            q1 = pose[0] + 4*np.pi - (-2*np.pi * np.cos(np.pi*self.flight_time/(self.last_flight_time-self.savety_time)) +  2*np.pi)
            q2 = pose[1]
            dq1 = 2*np.pi **2 / (self.last_flight_time-self.savety_time) * np.sin(np.pi*self.flight_time/(self.last_flight_time-self.savety_time))
            dq2 = 0
        
        return q1, q2, dq1, dq2
    
    def prepare_mode(self, mode):
        """Prepare self.add variable for the next jumping mode.
        This has to be used driectly before the next flight phase, where the calc trajecotry funtion is used to perform the jumping mode.
        
        

        Args:
            mode (str): jumping mode to perform next.
                Posible modes are: 
                    "j": normal jump.
                    "b": backflip.
                    "f": forward_flip.
                    "bc": backflip with knee directionchange.
                    "fc": forwardflip with knee directionchange.
                    "bd": double backflip.
                    "fd": double forwardflip.

        """
        if mode == 'j':
            return 
        elif mode == 'b':
            self.add += 2* np.pi
        elif mode == 'bd':
            self.add += 4* np.pi
        elif mode == 'bc':
            self.add += 2* np.pi
            self.knee_direction = int(not bool(self.knee_direction))
        elif mode == 'f':
            self.add -= 2* np.pi
        elif mode == 'fc':
            self.knee_direction = int(not bool(self.knee_direction))
            self.add -= 2* np.pi
            
    
    def energy_shaping_force(self, state, h_des, k = 100):
        """Second way to do energy shaping:
        Continuously control energy during liftoff a achieve the correct jumpoff energy.
        In Simulation, this hasn't worked yet.

        Args:
            state (list): Current state: [q1,q2,dq1,dq2]
            h_des (float): Desired jumping height.
            k (int, optional): hyperparameter to tune. Has to be larger 0. Defaults to 100.

        Returns:
            float: Feed forward force to apply
        """
        q0 = self.plant.forward_kinematics(*state[:2])[0]
        dq0 = self.plant.forward_velocity(*state)[0]
        m = self.plant.m
        g = self.plant.g
        # fac = 1 if dq0 >= 0 else -1
        E = 1/2 * m * dq0**2 + m*g*q0
        E_des = m*g*h_des 
        F_ff = -k*dq0*(E-E_des)
        return F_ff
    
    
    
    def calcff(self, state, Fx, Fy=0):
        """Experimental function  to calculate the feed forward torque form a given feed forward force.

        Args:
            state (tuple): current state vector
            Fx (float): Desired force in x direction
            Fy (float, optional): Desired force in y direction. Defaults to 0.

        Returns:
            tuple: tau1, tau2
        """
        q1 = state[0]
        q2 = state[1]
        L1 = 0.205
        L2 = 0.25
        cos = np.cos
        sin = np.sin
        
        tau2 = -Fx*L2*sin(q1+q2) + Fy*L2*cos(q1+q2)
        tau1 = -tau2 -Fx*(L1*sin(q1)+L2*sin(q1+q2)) + Fy*(L1*cos(q1)+L2*cos(q1+q2))
        return  tau1, tau2
    
    def do_control(self, state, desired_height, verbose = True):
        """Control fuction to calculate motor torques.

        Args:
            state (list): State vector [q1,q2,dq1,dq2]
            desired_height (float): Desired jumping height.
            verbose (bool, optional): Print out phase and jumping height. Defaults to True.

        Returns:
            tuple: tau1, tau2
        """
        self.time += self.dt
        if self.phase == self.FLIGHT:
            self.flight_time += self.dt
            pose = list(self.plant.inverse_kinematics(.25,0, self.knee_direction))
            dpose = [0,0]
            
            # Backflipp control
            if self.nth_jump > 0 and self.flight_time <= (self.last_flight_time-self.savety_time):
                pose[0], pose[1], dpose[0], dpose[1] = self.calc_trajectory(pose, self.mode)
            
            # Add x times 2 pi for the correct joint position
            pose[0] += self.add
            
            tau1, tau2 = self.combined_stiffness(state,
                            # tau_ff=np.zeros((2, 1)),
                            tau_ff=self.plant.gravity_vector(*state[0:2])[1:3], # cancel out gravity
                            Kpj=(50,100), q_d=pose,
                            Kdj=(4,4), qd_d=dpose,
                            f_ff=(0,0),
                            Kpc=(50,50), p_d=self.plant.forward_kinematics(*pose),
                            Kdc=(40,10), pd_d=self.plant.forward_velocity(*state[:2],*dpose),knee_direction=self.knee_direction)
            
            if self.se.get_contact():
                self.phase = self.TOUCHDOWN
                # save height for later state estimation
                self.pose_at_touchdown = self.plant.forward_kinematics(*state[:2])[0]
                if verbose: 
                    print("Touchdown")        

        if self.phase == self.TOUCHDOWN:
            pose = list(self.plant.inverse_kinematics(.25,0,self.knee_direction))
            pose[0] += self.add
            
            tau1, tau2 = self.combined_stiffness(state,
                            # tau_ff=np.zeros((2, 1)),
                            tau_ff=self.plant.gravity_vector(*state[0:2])[1:3],
                            Kpj=(100,600), q_d=pose,
                            Kdj=(130,130), qd_d=(0,0),
                            # f_ff=(-self.plant.m*self.plant.g,0),
                            Kpc=(10,2000), p_d=(.25,0),
                            Kdc=(20,20), pd_d=(0,0),knee_direction=self.knee_direction)
            
            #change
            if self.plant.forward_velocity(*state)[0]>=0: 
                self.phase = self.LIFTOFF
                self.start_liftoff_pos = self.plant.forward_kinematics(*state[:2])[0]
                try:
                    self.last_jumping_height = self.estimate_last_height(self.pose_at_liftoff, self.pose_at_touchdown, self.flight_time)
                except TypeError:
                    pass
                if verbose: 
                    print("Liftoff")
                try:
                    self.height_factor *= (desired_height/self.last_jumping_height)**2        
                except ZeroDivisionError:
                    pass
            
                    
        elif self.phase == self.LIFTOFF:
            pose = list(self.plant.inverse_kinematics(.33,0,self.knee_direction))
            pose[0] += self.add
            tau1, tau2 = self.combined_stiffness(state,
                                # tau_ff=self.plant.gravity_vector(*state[0:2])[1:3],
                                # tau_ff=self.calcff(state,self.height_factor * desired_height*(self.plant.m * self.plant.g)/self.last_liftoff_way+self.plant.g*self.plant.m,0),
                                # tau_ff=np.zeros((2, 1)),
                                # Kpj=(100,200), q_d=pose,
                                # Kdj=(0,0), qd_d=(0,0),
                                f_ff=(self.height_factor * desired_height*(self.plant.m * self.plant.g)/self.last_liftoff_way+self.plant.g*self.plant.m,0),
                                Kpc=(0,20), p_d=(.33,0),
                                Kdc=(0,1), pd_d=(0,0),knee_direction=self.knee_direction)
            
            if not self.se.get_contact():
            # if self.plant.forward_kinematics(*state[:2])[0] > .36:
                self.phase = self.FLIGHT
                self.end_liftoff_pos = self.plant.forward_kinematics(*state[:2])[0]
                self.time_at_liftoff = self.time
                self.last_liftoff_way = self.end_liftoff_pos - self.start_liftoff_pos
                self.nth_jump +=1 # count jumps
                self.last_flight_time = self.flight_time # save last flight time as estimator how fast backflips have to be
                if verbose: 
                    print("Flight")
                    print("last way ", self.last_liftoff_way )
                    print("last jumping height ", self.last_jumping_height)
                    print("time",self.last_flight_time-self.savety_time)
                self.last_jumping_height = 0
                self.flight_time = 0 # set timer to zero for the next jump
                self.pose_at_liftoff = self.plant.forward_kinematics(*state[:2])[0] # save pose at liftoff
                self.liftoff_vel = self.plant.forward_velocity(*state)[0]
                # prepare next jumping mode
                self.mode = self.mode_list[(self.nth_jump-1)%len(self.mode_list)]
                self.prepare_mode(self.mode)

        return tau1,tau2



if __name__ == "__main__":

    # CLI
    cli = [{'g|gain_scheduling': {'action': 'store_true', 'help': "Use gain scheduling control mode"},
            'e|energy_shaping':  {'action': 'store_true', 'help': "Use gain scheduling control mode"},
            'b|backflip':        {'action': 'store_true', 'help': "Use backflip control mode"},
            's|save_results':    {'action': 'store_true', 'help': "Save results"},
            'r|rail':            {'action': 'store_true', 'help': "Use URDF with rail"},
            'l|long_rail':       {'action': 'store_true', 'help': "Use URDF with long rail"}},
           "Simulation of hopping leg based on quasi-direct drives"]
    parser = parse(*cli)
    args, unknown = parser.parse_known_args()

    # Results folder
    results_folder = {""}
    results_folder = ""
    modes = ["j"]

    if args.gain_scheduling:
        results_folder = "gain_scheduling_control"
    elif args.energy_shaping:
        results_folder = "energy_shaping_control"        
    elif args.backflip:
        results_folder = "backflip_energy_shaping_control"
        # existing modes: 
        # "j": jump, "b": backflip, "bc": backflip + change knee direction, "bd": double backflip, 
        # "f": forwardflip, "fc": forwardflip + change knee direction, "fd": double forwardflip
        modes = ["j", "j", "b", "j", "fc", "b"]
    # setup simulation
    sim = HopperSimulation(control_mode=2, rail=(args.rail or args.long_rail), drop_height=1, torque_limits=(12,12), long_leg=args.long_rail, frequency=700)
    plant = sim.plant
    
    # Initialize state estimation and start virtual IMU streaming
    se = StateEstimation(sim.robot, sim.planeID, plant, sim.dt,  imu_link=0)
    se.start_imu_scan()
    se.start_state_estimation()
    
    tracker = SimulationTracker(sim, se, saveRecording=args.save_results,
                            filename="../results/" + results_folder + "/pybullet_recording.mp4")
    # Initialize Controller
    c = PD_controller(plant)
    ce = EnergyShapingControl(plant, se, modes)
    cg = GainSchedulingControl(plant, se)
    # cg = GainScheduling(plant, se) # Use energy shaping from state machine class instead.

    # initial cycle:
    for i in range(1000):
        p.setJointMotorControlArray(
                    sim.robot, sim.jointIndices, p.POSITION_CONTROL, plant.inverse_kinematics(.25,0,0))
        p.stepSimulation()
        sim.sleep()
    p.setJointMotorControlArray(
                sim.robot, sim.jointIndices, p.VELOCITY_CONTROL, (0,0), forces=(0,0))
   
    count = 0
    phase_dict = {"FLIGHT":0,"TOUCHDOWN":1,"LIFTOFF":2, "FLIGHT_ASCEND": 3}
    
    while sim.curtime < 5:
        count = count + 1
        sim.step_time()
        
        if args.gain_scheduling:
            tau1, tau2 = cg.do_control(sim.get_state_vector())
            tracker.phase.append(phase_dict[cg.phase])
        
        else:
            desired_height = .6
            tau1, tau2 = ce.do_control(sim.get_state_vector(),desired_height=desired_height)
            x = ce.estimate_state(sim.get_state_vector()) 
            # tracker.basepos_estimated.append(x)
            # tracker.basepos_estimated.append(ce.last_jumping_height)
            tracker.basepos_desired.append(desired_height)
            tracker.height_factor.append(ce.height_factor)
            tracker.phase.append(phase_dict[ce.phase])
            
        
        p.setJointMotorControlArray(
                    sim.robot, sim.jointIndices, p.TORQUE_CONTROL, forces=[tau1, tau2])
        tracker.update(tau1, tau2)
        # move leg up and down (position control)
        p.stepSimulation()
        sim.sleep()
        # time.sleep(0.002)
        
    p.disconnect()
    save = save_figures=args.save_results
    tracker.create_plots(save_figures=save,
                         folder="../results/"+results_folder)
    if save:
        tracker.write_csv("../results/"+results_folder)
        tracker.trajectory_export("../results/"+results_folder)
