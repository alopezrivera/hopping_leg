import numpy as np
from numpy.core.fromnumeric import clip


class PD_controller():
    def __init__(self, hopper_plant) -> None:
        self.plant = hopper_plant
        
    def combined_stiffness(self, state_vector,
                            tau_ff=np.zeros((2, 1)),
                            Kpj=np.zeros((2, 2)), q_d=np.zeros((2, 1)),
                            Kdj=np.zeros((2, 2)), qd_d=np.zeros((2, 1)),
                            f_ff=np.zeros((2, 1)),
                            Kpc=np.zeros((2, 2)), p_d=np.zeros((2, 1)),
                            Kdc=np.zeros((2, 2)), pd_d=np.zeros((2, 1)),
                            knee_direction=0, clip_torques=True):
        """Low level pd controler

        Parameters
        ----------
        state_vector : list or array_like
            current state in following order: [q1,q2,dq1,dq2]
        tau_ff : list or arraylike (2,1), optional
            feed forward torque, by default np.zeros((2, 1))
        Kpj : list or arraylike (2,1) or (2,2), optional
            Kp value joint space, by default np.zeros((2, 2))
        q_d : list or arraylike (2,1), optional
            desired joint positions, by default np.zeros((2, 1))
        Kdj : list or arraylike (2,1) or (2,2), optional
            Kd value joint space, by default np.zeros((2, 2))
        qd_d : list or arraylike (2,1), optional
            desired joint velocities, by default np.zeros((2, 1))
        f_ff : list or arraylike (2,1), optional
            cartesian feed forward force at the ee, by default np.zeros((2, 1))
        Kpc : list or arraylike (2,1) or (2,2), optional
            Kp value cartesian space, by default np.zeros((2, 2))
        p_d : list or arraylike (2,1), optional
            end effector position cartesian space, by default np.zeros((2, 1))
        Kdc : list or arraylike (2,1) or (2,2), optional
            Kd value cartesian space, by default np.zeros((2, 2))
        pd_d : list or arraylike (2,1), optional
            end effector velocity cartesian space, by default np.zeros((2, 1))
        knee_direction: int, optional 
                0: Knee to the left.
                1: Knee to the right.
                2: Knee always to the outer side.
                3: Knee always to the inner side.
                    by default 0.
        clip_torques: bool, optional
            clip torque to values set in plant, by default True

        Returns
        -------
        tuple
            tau1, tau2
        """
        q = np.array(state_vector[:2]).reshape((2, 1))  # joint angles
        qd = np.array(state_vector[2:4]).reshape((2, 1))  # joint velocities
        p = np.array(self.plant.forward_kinematics(
            *q)).reshape((2, 1))  # Cartesian position of EE
        # Cartesian velocity of EE
        pd = np.array(self.plant.forward_velocity(*q, *qd)).reshape(2, 1)
        try:
            qdes = self.plant.inverse_kinematics(p_d[0],p_d[1],knee_direction)
        except:
            qdes = self.plant.inverse_kinematics(p_d[0]-0.01,p_d[1]-0.01,knee_direction)
            # print (p_d)
            # raise
        J = self.plant.jacobian(qdes[0], qdes[1])  # Jacobian

        # format gain matrices
        Kpj = np.array(Kpj)
        if Kpj.size == 2:
            Kpj = np.diag(Kpj)

        Kdj = np.array(Kdj)
        if Kdj.size == 2:
            Kdj = np.diag(Kdj)

        Kpc = np.array(Kpc)
        if Kpc.size == 2:
            Kpc = np.diag(Kpc)

        Kdc = np.array(Kdc)
        if Kdc.size == 2:
            Kdc = np.diag(Kdc)

        # reshape input variables
        q_d = np.matrix(q_d).reshape((2, 1))
        qd_d = np.matrix(qd_d).reshape((2, 1))
        p_d = np.matrix(p_d).reshape((2, 1))
        pd_d = np.matrix(pd_d).reshape((2, 1))
        f_ff = np.matrix(f_ff).reshape((2, 1))

        tau = (tau_ff  # torque control
               + np.matmul(Kpj, (q_d - q))  # joint position control
               + np.matmul(Kdj, (qd_d - qd))  # joint velocity control
               + np.matmul(J.T, (f_ff  # force control
                                 + (np.matmul(Kpc, (p_d - p))  # cartesian position control
                                    + np.matmul(Kdc, (pd_d - pd)))  # cartesian velocity control
                                 ))
               )
        
        
        
        # print("tau joint space",tau_ff  # torque control
        #        + np.matmul(Kpj, (q_d - q))  # joint position control
        #        + np.matmul(Kdj, (qd_d - qd))  )
        # print("tau f_ff", np.matmul(J.T, (f_ff)))
        # print("tau pos", np.matmul(J.T, np.matmul(Kpc, (p_d - p)) ))
        # print("tau vel",  np.matmul(J.T, np.matmul(Kdc, (pd_d - pd)) ))
        # print("tau_cart", np.matmul(J.T, (f_ff  # force control
        #                          + (np.matmul(Kpc, (p_d - p))  # cartesian position control
        #                             + np.matmul(Kdc, (pd_d - pd)))  # cartesian velocity control
        #                          )))
        # print("tau", tau)
        if clip_torques:
            tau[0,0] = np.clip(tau[0,0],-self.plant.torque_limits[0],self.plant.torque_limits[0])
            tau[1,0] = np.clip(tau[1,0],-self.plant.torque_limits[1],self.plant.torque_limits[1])
        return tau[0, 0], tau[1, 0]


    def cartesian_stiffness(self, state_vector,
                            f_ff=np.zeros((2, 1)),
                            Kpc=np.zeros((2, 2)), p_d=np.zeros((2, 1)),
                            Kdc=np.zeros((2, 2)), pd_d=np.zeros((2, 1)),
                            knee_direction=0, clip_torques=True):
        """Low level pd controler

        Parameters
        ----------
        state_vector : list or array_like
            current state in following order: [q1,q2,dq1,dq2]
        f_ff : list or arraylike (2,1), optional
            cartesian feed forward force at the ee, by default np.zeros((2, 1))
        Kpc : list or arraylike (2,1) or (2,2), optional
            Kp value cartesian space, by default np.zeros((2, 2))
        p_d : list or arraylike (2,1), optional
            end effector position cartesian space, by default np.zeros((2, 1))
        Kdc : list or arraylike (2,1) or (2,2), optional
            Kd value cartesian space, by default np.zeros((2, 2))
        pd_d : list or arraylike (2,1), optional
            end effector velocity cartesian space, by default np.zeros((2, 1))
        knee_direction: int, optional 
                0: Knee to the left.
                1: Knee to the right.
                2: Knee always to the outer side.
                3: Knee always to the inner side.
                    by default 0.
        clip_torques: bool, optional
            clip torque to values set in plant, by default True

        Returns
        -------
        tuple
            tau1, tau2
        """
        q = np.array(state_vector[:2]).reshape((2, 1))  # joint angles
        qd = np.array(state_vector[2:4]).reshape((2, 1))  # joint velocities
        p = np.array(self.plant.forward_kinematics(
            *q)).reshape((2, 1))  # Cartesian position of EE
        # Cartesian velocity of EE
        pd = np.array(self.plant.forward_velocity(*q, *qd)).reshape(2, 1)
        qdes = self.plant.inverse_kinematics(p_d[0],p_d[1],knee_direction)
        J = self.plant.jacobian(qdes[0], qdes[1])  # Jacobian
        qdot_des = self.plant.inverse_velocity(qdes[0], qdes[1], pd_d[0], pd_d[1])

        # format gain matrices
        Kpc = np.array(Kpc)
        if Kpc.size == 2:
            Kpc = np.diag(Kpc)

        Kdc = np.array(Kdc)
        if Kdc.size == 2:
            Kdc = np.diag(Kdc)
            
        f_ff = np.array(f_ff).reshape((2, 1))

        # reshape input variables
        p_d = np.matrix(p_d).reshape((2, 1))
        pd_d = np.matrix(pd_d).reshape((2, 1))

        tau = np.matmul(J.T, (f_ff  # force control
                                 + (np.matmul(Kpc, (p_d - p))  # cartesian position control
                                    + np.matmul(Kdc, (pd_d - pd)))  # cartesian velocity control
                                 ))

        tau_gravity = self.plant.gravity_vector(qdes[0], qdes[1])
        tau[0,0] = tau[0,0] + tau_gravity[1]
        tau[1,0] = tau[1,0] + tau_gravity[2]

        if clip_torques:
            tau[0,0] = np.clip(tau[0,0],-self.plant.torque_limits[0],self.plant.torque_limits[0])
            tau[1,0] = np.clip(tau[1,0],-self.plant.torque_limits[1],self.plant.torque_limits[1])
        return qdes[0], qdes[1], qdot_des[0], qdot_des[1], tau[0, 0], tau[1, 0]

    def joint_space_stiffness(self, state_vector,
                            tau_ff=np.zeros((2, 1)),
                            Kpj=np.zeros((2, 2)), q_d=np.zeros((2, 1)),
                            Kdj=np.zeros((2, 2)), qd_d=np.zeros((2, 1)), clip_torques=True):
        """Low level pd controler

        Parameters
        ----------
        state_vector : list or array_like
            current state in following order: [q1,q2,dq1,dq2]
        tau_ff : list or arraylike (2,1), optional
            feed forward torque, by default np.zeros((2, 1))
        Kpj : list or arraylike (2,1) or (2,2), optional
            Kp value joint space, by default np.zeros((2, 2))
        q_d : list or arraylike (2,1), optional
            desired joint positions, by default np.zeros((2, 1))
        Kdj : list or arraylike (2,1) or (2,2), optional
            Kd value joint space, by default np.zeros((2, 2))
        qd_d : list or arraylike (2,1), optional
            desired joint velocities, by default np.zeros((2, 1))
        clip_torques: bool, optional
            clip torque to values set in plant, by default True
        
        Returns
        -------
        tuple
            tau1, tau2
        """
        q = np.array(state_vector[:2]).reshape((2, 1))  # joint angles
        qd = np.array(state_vector[2:4]).reshape((2, 1))  # joint velocities
        
        # format gain matrices
        Kpj = np.array(Kpj)
        if Kpj.size == 2:
            Kpj = np.diag(Kpj)

        Kdj = np.array(Kdj)
        if Kdj.size == 2:
            Kdj = np.diag(Kdj)

        
        # reshape input variables
        q_d = np.matrix(q_d).reshape((2, 1))
        qd_d = np.matrix(qd_d).reshape((2, 1))
        
        tau = (tau_ff  # torque control
               + np.matmul(Kpj, (q_d - q))  # joint position control
               + np.matmul(Kdj, (qd_d - qd))  # joint velocity control
               )
        if clip_torques:
            tau[0,0] = np.clip(tau[0,0],-self.plant.torque_limits[0],self.plant.torque_limits[0])
            tau[1,0] = np.clip(tau[1,0],-self.plant.torque_limits[1],self.plant.torque_limits[1])
        return tau[0, 0], tau[1, 0]



if __name__ == "__main__":
    import sys 
    sys.path.append("../../hopper_plant/")
    from hopper_plant import HopperPlant
    
    plant = HopperPlant()
    control = LowLevelControl(plant)
    
    erg = control.combined_stiffness((*plant.inverse_kinematics(.4,0,1),0,0), f_ff=(.2,0),p_d=(.3,0),knee_direction=1)
    
    print("result", erg)

    print("\n \nInverse: \n")
    erg = control.combined_stiffness((*plant.inverse_kinematics(.4,0,0),0,0), f_ff=(.2,0),p_d=(.3,0),knee_direction=0)
    
    print("result", erg)
    
    # erg = control.cartesian_stiffness((*plant.inverse_kinematics(.4,0,1),0,0), f_ff=(.2,0),p_d=(.3,-.1),knee_direction=1)
    
    # print(erg)
    
    # erg = control.cartesian_stiffness((*plant.inverse_kinematics(.4,0,0),0,0), f_ff=(.2,0),p_d=(.3,.1),knee_direction=0)
    
    # print(erg)
