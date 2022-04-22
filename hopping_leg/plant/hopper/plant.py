import numpy as np


class HopperPlant():
    def __init__(self, mass=[0.91482, 1.2148, 0.14334], Izz=[0.0014078, 0.00012493], com1=[0.056105, 1.0785E-05], com2=[0.096585, 9.8958E-09], link_length=[0.205, 0.25], gravity=9.81, acc=0, torque_limits=(6,6)) -> None:
        """Load plant model. All defualt values are from the urdf with rail and long leg.

        Args:
            mass (list, optional): Mass of baselink, link1 and link2. Defaults to [0.91482, 1.2148, 0.14334].
            Izz (list, optional): Izz from link1 and link2. Defaults to [0.0014078, 0.00012493].
            com1 (list, optional): COM position link1. Defaults to [0.056105, 1.0785E-05].
            com2 (list, optional): COM Position link2. Defaults to [0.096585, 9.8958E-09].
            link_length (list, optional): Lenghts of link1 and link2. Defaults to [0.205, 0.25].
            gravity (float, optional): Gravity. Defaults to 9.81.
            acc (int, optional): Vertical acceleration. Defaults to 0.
            torque_limits (tuple, optional): Motor torque limits for hip and knee motor. Defaults to (6,6).
        """
        self.L1 = link_length[0]
        self.L2 = link_length[1]
        self.g = gravity
        self.acc = acc  # vertical acceleration
        self.m0 = mass[0]
        self.m1 = mass[1]
        self.m2 = mass[2]
        self.m = self.m0 + self.m1 + self.m2
        self.Izz1 = Izz[0]
        self.Izz2 = Izz[1]
        self.cx1 = com1[0]
        self.cy1 = com1[1]
        self.cx2 = com2[0]
        self.cy2 = com2[1]
        self.torque_limits = torque_limits

    def jacobian(self, q1, q2):
        """Hybrid jacobian of the ee wrt base. 

        Args:
            q1 (float): q1
            q2 (float): q2

        Returns:
            np.array: Jacobian
        """
        J = np.array([[-self.L1*np.sin(q1) - self.L2*np.sin(q1+q2), -self.L2*np.sin(q1+q2)],
                      [self.L1*np.cos(q1) + self.L2*np.cos(q1+q2), self.L2*np.cos(q1+q2)]])
        return J

    def jacobian_dot(self, q1, q2, dq1, dq2):
        """Derivative of Jacobian

        Args:
            q1 (float): q1
            q2 (flaot): q2
            dq1 (float): dq1
            dq2 (float): dq2

        Returns:
            np.array: Jacobian dot
        """
        Jdot = np.array([[-self.L1*np.cos(q1)*dq1 - self.L2*(dq1 + dq2)*np.cos(q1 + q2), -self.L2*(dq1 + dq2)*np.cos(q1 + q2)],
                         [-self.L1*np.sin(q1)*dq1 - self.L2*(dq1 + dq2)*np.sin(q1 + q2), -self.L2*(dq1 + dq2)*np.sin(q1 + q2)]])
        return Jdot
    
    def inverse_jacobian(self, q1, q2):
        """Inverse jacobian

        Args:
            q1 (float): q1
            q2 (float): q2

        Returns:
            np.array: Inverse jacobian
        """
        Jinv = np.array([[np.cos(q1 + q2)/(self.L1*np.sin(q2)), np.sin(q1 + q2)/(self.L1*np.sin(q2))],
        [-(self.L1*np.cos(q1) + self.L2*np.cos(q1 + q2))/(self.L1*self.L2*np.sin(q2)), -(self.L1*np.sin(q1) + self.L2*np.sin(q1 + q2))/(self.L1*self.L2*np.sin(q2))]])
        return Jinv
        
    def forward_kinematics(self, q1, q2):
        """forward kinematics

        Args:
            q1 (float): q1
            q2 (flaot): q2

        Returns:
            tuple: x,y 
        """
        x = self.L1*np.cos(q1) + self.L2*np.cos(q1+q2)
        y = self.L1*np.sin(q1) + self.L2*np.sin(q1+q2)
        return x, y

    def inverse_kinematics(self, x, y, knee_direction=0):
        """Inverse kinematics for the hopping leg

        Args:
            x (float): x position (axis pointing downwards relative to base)
            y (float): y position (axis pointing to the right relative to base)
            L1 (float, optional): Link lenght one. Defaults to L1.
            L2 (float, optional): Link lenght two. Defaults to L2.
            knee_direction (int, optional): 
                0: Knee to the left.
                1: Knee to the right.
                2: Knee always to the outer side.
                3: Knee always to the inner side.
                    Defaults to 0.

        Raises:
            ValueError: Position is not possible with given input variables.

        Returns:
            [type]: [description]
        """
        q2 = np.arccos((x*x+y*y-self.L1**2-self.L2**2)/(2*self.L1*self.L2))
        # knee to the right
        if knee_direction == 1:
            q2 = -q2
        # Direction of the knee always outside
        elif knee_direction == 2:
            if y >= 0:
                q2 = -q2
        elif knee_direction == 3:
            if y < 0:
                q2 = -q2
        # avoid ZeroDevisionError
        if x != 0:
            q1 = np.arctan(y/x) - np.arctan((self.L2*np.sin(q2)) /
                                            (self.L1+self.L2*np.cos(q2)))
        else:
            q1 = np.pi/2 - \
                np.arctan((self.L2*np.sin(q2))/(self.L1+self.L2*np.cos(q2)))
        # for angles largen than pi:
        if q1 > np.pi:
            q1 = - (q1 - np.pi)
        elif q1 < -np.pi:
            q1 = - (q1 + np.pi)
        # for case when q1 should be larger pi/2:
        if not np.isclose(np.array(self.forward_kinematics(q1, q2)), np.array([x, y]), rtol=0.01, atol=0.01).all():
            if q2 < 0:
                q1 += np.pi
            else:
                q1 -= np.pi
        if np.isnan(q1) or np.isnan(q2):
            raise ValueError("Coordinates out of working space")
        # use only q1 values between -pi and pi
        if q1 > np.pi:
            q1 = q1 - 2 * np.pi
        elif q1 < -np.pi:
            q1 = q1 + 2 * np.pi
        return q1, q2

    def forward_velocity(self, q1, q2, dq1, dq2):
        """Forward Velocity

        Args:
            q1 (float): q1
            q2 (flaot): q2
            dq1 (float): dq1
            dq2 (float): dq2

        Returns:
            tuple: dx, dy
        """
        vx = -dq1*self.L1*np.sin(q1) - (dq1+dq2) * self.L2 * np.sin(q1+q2)
        vy = dq1*self.L1*np.cos(q1) + (dq1+dq2) * self.L2 * np.cos(q1+q2)
        return vx, vy

    def inverse_velocity(self, q1, q2, xd, yd):
        """Inverse velocity

        Args:
            q1 (float): q1
            q2 (flaot): q2
            xd (float): dx
            yd (float): dy

        Returns:
            tuple: dq1, dq2
        """
        Jinv = self.inverse_jacobian(q1,q2)
        v_input = np.asarray([xd, yd])
        v_input.reshape(2, 1)
        dq = np.matmul(Jinv, v_input)
        return dq[0], dq[1]

    def forward_acceleration(self, q1, q2, dq1, dq2, ddq1, ddq2):
        """Forward acceleration

        Args:
            q1 (float): q1
            q2 (flaot): q2
            dq1 (float): dq1
            dq2 (float): dq2
            ddq1 (flaot): ddq1
            ddq2 (flaot): ddq2

        Returns:
            tuple: ddx, ddy
        """
        ddx = -self.L1*np.sin(q1)*ddq1 - self.L1*np.cos(q1)*dq1**2 - self.L2*(dq1 + dq2)**2*np.cos(q1 + q2) - self.L2*(ddq1 + ddq2)*np.sin(q1 + q2)
        ddy = -self.L1*np.sin(q1)*dq1**2 + self.L1*np.cos(q1)*ddq1 - self.L2*(dq1 + dq2)**2*np.sin(q1 + q2) + self.L2*(ddq1 + ddq2)*np.cos(q1 + q2)
        return ddx, ddy
    
    def inverse_acceleration(self, q1, q2, dq1, dq2, ddx, ddy):
        """Inverse acceleration

        Args:
            q1 (float): q1
            q2 (flaot): q2
            dq1 (float): dq1
            dq2 (float): dq2
            ddx (float): ddx
            ddy (float): ddy

        Returns:
            tuple: ddq1, ddq2
        """
        Jinv = self.inverse_jacobian(q1,q2)
        Jdot = self.jacobian_dot(q1,q2,dq1,dq2)
        a_input = np.asarray([ddx, ddy])
        a_input.reshape(2, 1)
        dq_input = np.asarray([dq1, dq2])
        dq_input.reshape(2, 1)
        ddq = np.matmul(Jinv, a_input) - np.matmul(np.matmul(Jinv,Jdot),dq_input)
        return ddq[0], ddq[1]

    

    def global_jacobian(self,q1,q2):
        """Hybrid jacobian matrix including prismatic joint

        Args:
            q1 (float): q1
            q2 (flaot): q2
        
        Returns:
            np.matrix: Jacobian
        """
        L1, L2 = self.L1, self.L2
        J = np.matrix([[-1, -L1*np.sin(q1) - L2*np.sin(q1 + q2), -L2*np.sin(q1 + q2)],
                   [0, L1*np.cos(q1) + L2*np.cos(q1 + q2), L2*np.cos(q1 + q2)]])
        return J
        

    def rhs(self, state):
        """
        Returns a concatenated vector of velocities and accelerations (Right-Hand Side of ODE).
        Input:
            state - [position, velocity] in the z direction
        Returns:
            res - [velocity, acceleration]
        """
        res = np.zeros(2)
        res[0] = state[1]
        res[1] = self.acc
        return res

    def euler_integrator(self, y, dt):
        """
        Euler method for integration.
        Returns a concatenated vector of velocities and accelerations.
        Input:
            y - [position, velocity]
            dt - time interval
        Returns:
            res - [velocity, acceleration]
        """
        return self.rhs(y)

    def runge_integrator(self, y, dt):
        """
        Runge-Kutta method for integration.
        Returns a concatenated vector of velocities and accelerations.
        Input:
            y - [position, velocity]
            dt - time interval
        Returns:
            res - [velocity, acceleration]
        """
        k1 = self.rhs(y)
        k2 = self.rhs(y + 0.5 * dt * k1)
        k3 = self.rhs(y + 0.5 * dt * k2)
        k4 = self.rhs(y + dt * k3)
        return (k1 + 2 * (k2 + k3) + k4) / 6.0

    def step(self, state, acc, dt, integrator="runge_kutta"):
        """
           Apply an integration method (euler, runge_kutta) to get the new state.
           Returns the new state as a concatenated vector of position and velocity (height and vertical velocity).
           Input:
               state - [position, velocity]
               acc - vertical acceleration
               dt - time interval
               integrator - "runge_kutta", "euler"
           Returns:
              state - [position, velocity]
           """
        self.acc = acc
        if integrator == "runge_kutta":
            state += dt * self.runge_integrator(state, dt)
        elif integrator == "euler":
            state += dt * self.euler_integrator(state, dt)
        return state
    
    
    def generalized_mass_inertia_matrix(self, q1, q2):
        """Mass matrix, generated with kinematics generator

        Args:
            q1 (float): q1
            q2 (flaot): q2
            
        Returns:
            np.array: Mass matrix
        """
        Izz1, Izz2, L1, cx1, cx2, cy2, m0, m1, m2 = self.Izz1, self.Izz2, self.L1, self.cx1, self.cx2, self.cy2, self.m0, self.m1, self.m2
        generalized_mass_inertia_matrix = np.array([[m0 + m1*np.sin(q1)**2 + m1*np.cos(q1)**2 + m2*np.sin(q1 + q2)**2 + m2*np.cos(q1 + q2)**2, -L1*m2*np.sin(q2)*np.cos(q1 + q2) + L1*m2*np.sin(q1 + q2)*np.cos(q2) + cx1*m1*np.sin(q1) + cx2*m2*np.sin(q1 + q2) + cy2*m1*np.cos(q1) + cy2*m2*np.cos(q1 + q2), cx2*m2*np.sin(q1 + q2) + cy2*m2*np.cos(q1 + q2)], [L1*m2*np.sin(q1) + cx1*m1*np.sin(q1) + cx2*m2*np.sin(q1 + q2) + cy2*m1*np.cos(q1) + cy2*m2*np.cos(q1 + q2), Izz1 + Izz2 + L1*cx2*m2*np.cos(q2) - L1*cy2*m2*np.sin(q2) + L1*(L1*m2*np.sin(q2) - cy2*m2)*np.sin(q2) + L1*(L1*m2*np.cos(q2) + cx2*m2)*np.cos(q2), Izz2 + L1*cx2*m2*np.cos(q2) - L1*cy2*m2*np.sin(q2)], [m2*(cx2*np.sin(q1 + q2) + cy2*np.cos(q1 + q2)), Izz2 + L1*cx2*m2*np.cos(q2) - L1*cy2*m2*np.sin(q2), Izz2]])
        return generalized_mass_inertia_matrix
    
    def gravity_vector(self, q1, q2):
        """Gravity vector generate with kinematics generator

        Args:
            q1 (float): q1
            q2 (flaot): q2

        Returns:
            np.array: Gravity vector
        """
        L1, cx1, cx2, cy2, g, m0, m1, m2 = self.L1, self.cx1, self.cx2, self.cy2, self.g, self.m0, self.m1, self.m2
        gravity_vector = np.array([[-g*m0 - g*m1 - g*m2], [-L1*g*m2*np.sin(q1) - cx1*g*m1*np.sin(q1) - cx2*g*m2*np.sin(q1 + q2) - cy2*g*m1*np.cos(q1) - cy2*g*m2*np.cos(q1 + q2)], [-cx2*g*m2*np.sin(q1 + q2) - cy2*g*m2*np.cos(q1 + q2)]])
        return gravity_vector
    
    def coriolis_centrifugal_matrix(self, q1, q2, dq0, dq1, dq2):
        """Coriolis centrifugal matirx

        Args:
            q1 (float): q1
            q2 (flaot): q2
            dq0 (float): dq0 (prismatic joint velocity)
            dq1 (float): dq1
            dq2 (float): dq2
        Returns:
            np.array: Coriolis Matrix
        """
        L1, cx1, cx2, cy2, m1, m2 = self.L1, self.cx1, self.cx2, self.cy2, self.m1, self.m2
        coriolis_centrifugal_matrix = np.array([[0, L1*dq1*m2*np.cos(q1) + cx1*dq1*m1*np.cos(q1) + cx2*dq1*m2*np.cos(q1 + q2) + cx2*dq2*m2*np.cos(q1 + q2) - cy2*dq1*m1*np.sin(q1) - cy2*dq1*m2*np.sin(q1 + q2) - cy2*dq2*m2*np.sin(q1 + q2), cx2*dq1*m2*np.cos(q1 + q2) + cx2*dq2*m2*np.cos(q1 + q2) - cy2*dq1*m2*np.sin(q1 + q2) - cy2*dq2*m2*np.sin(q1 + q2)], [L1*dq1*m2*np.cos(q1) + cx1*dq1*m1*np.cos(q1) + cx2*dq1*m2*np.cos(q1 + q2) + cx2*dq2*m2*np.cos(q1 + q2) - cy2*dq1*m1*np.sin(q1) - cy2*dq1*m2*np.sin(q1 + q2) - cy2*dq2*m2*np.sin(q1 + q2), -2*L1*cx2*dq2*m2*np.sin(q2) - 2*L1*cy2*dq2*m2*np.cos(q2) - L1*dq0*m2*np.cos(q1) - cx1*dq0*m1*np.cos(q1) - cx2*dq0*m2*np.cos(q1 + q2) + cy2*dq0*m1*np.sin(q1) + cy2*dq0*m2*np.sin(q1 + q2), -L1*cx2*dq2*m2*np.sin(q2) - L1*cy2*dq2*m2*np.cos(q2) - cx2*dq0*m2*np.cos(q1 + q2) + cy2*dq0*m2*np.sin(q1 + q2)], [L1*dq1*m2*np.cos(q1) + cx2*dq1*m2*np.cos(q1 + q2) + cx2*dq2*m2*np.cos(q1 + q2) - cy2*dq1*m2*np.sin(q1 + q2) - cy2*dq2*m2*np.sin(q1 + q2), L1*cx2*dq1*m2*np.sin(q2) - L1*cx2*dq2*m2*np.sin(q2) + L1*cy2*dq1*m2*np.cos(q2) - L1*cy2*dq2*m2*np.cos(q2) - L1*dq0*m2*np.cos(q1) - cx2*dq0*m2*np.cos(q1 + q2) + cy2*dq0*m2*np.sin(q1 + q2), L1*cx2*dq1*m2*np.sin(q2) + L1*cy2*dq1*m2*np.cos(q2) - cx2*dq0*m2*np.cos(q1 + q2) + cy2*dq0*m2*np.sin(q1 + q2)]])
        return coriolis_centrifugal_matrix
    
    def inverse_dynamics(self, q1, q2, dq1, dq2, ddq0, ddq1, ddq2):
        """Inverse dynamics. Generated with kinematics generator.

        Args:
            q1 (float): q1
            q2 (flaot): q2
            dq1 (float): dq1
            dq2 (float): dq2
            ddq0 (float): ddq0
            ddq1 (float): ddq1
            ddq2 (float): ddq2

        Returns:
            np.array: [tau1, tau2]
        """
        Izz1, Izz2, L1, cx1, cx2, cy2, g, m0, m1, m2 = self.Izz1, self.Izz2, self.L1, self.cx1, self.cx2, self.cy2, self.g, self.m0, self.m1, self.m2
        inverse_dynamics = np.array([
            [L1*ddq1*m2*np.sin(q1) + L1*dq1**2*m2*np.cos(q1) + cx1*ddq1*m1*np.sin(q1) + cx1*dq1**2*m1*np.cos(q1) + cx2*ddq1*m2*np.sin(q1 + q2) + cx2*ddq2*m2*np.sin(q1 + q2) + cx2*dq1**2*m2*np.cos(q1 + q2) + 2*cx2*dq1*dq2*m2*np.cos(q1 + q2) + cx2*dq2**2*m2*np.cos(q1 + q2) + cy2*ddq1*m1*np.cos(q1) + cy2*ddq1*m2*np.cos(q1 + q2) + cy2*ddq2*m2*np.cos(q1 + q2) - cy2*dq1**2*m1*np.sin(q1) - cy2*dq1**2*m2*np.sin(q1 + q2) - 2*cy2*dq1*dq2*m2*np.sin(q1 + q2) - cy2*dq2**2*m2*np.sin(q1 + q2) + ddq0*m0 + ddq0*m1 + ddq0*m2 + g*m0 + g*m1 + g*m2], 
            [Izz1*ddq1 + Izz2*ddq1 + Izz2*ddq2 + L1**2*ddq1*m2 + 2*L1*cx2*ddq1*m2*np.cos(q2) + L1*cx2*ddq2*m2*np.cos(q2) - 2*L1*cx2*dq1*dq2*m2*np.sin(q2) - L1*cx2*dq2**2*m2*np.sin(q2) - 2*L1*cy2*ddq1*m2*np.sin(q2) - L1*cy2*ddq2*m2*np.sin(q2) - 2*L1*cy2*dq1*dq2*m2*np.cos(q2) - L1*cy2*dq2**2*m2*np.cos(q2) + L1*ddq0*m2*np.sin(q1) + L1*g*m2*np.sin(q1) + cx1*ddq0*m1*np.sin(q1) + cx1*g*m1*np.sin(q1) + cx2*ddq0*m2*np.sin(q1 + q2) + cx2*g*m2*np.sin(q1 + q2) + cy2*ddq0*m1*np.cos(q1) + cy2*ddq0*m2*np.cos(q1 + q2) + cy2*g*m1*np.cos(q1) + cy2*g*m2*np.cos(q1 + q2)], 
            [Izz2*ddq1 + Izz2*ddq2 + L1*cx2*ddq1*m2*np.cos(q2) + L1*cx2*dq1**2*m2*np.sin(q2) - L1*cy2*ddq1*m2*np.sin(q2) + L1*cy2*dq1**2*m2*np.cos(q2) + cx2*ddq0*m2*np.sin(q1 + q2) + cx2*g*m2*np.sin(q1 + q2) + cy2*ddq0*m2*np.cos(q1 + q2) + cy2*g*m2*np.cos(q1 + q2)]])
        return inverse_dynamics
    
    def friction_cone(self, mu, Fx):
        """Calculate maximum y-Forces to stay in friction cone.

        Args:
            mu (float): Friction coefficient
            Fx (flaot): Forces in x-direction

        Raises:
            Exception: Normal forces have to be positive (m*g + Fx > 0)

        Returns:
            float: max forces in y-direction.
        """
        if abs(self.g)*self.m + Fx > 0:
            Fy = mu*(abs(self.g)*self.m + Fx)
            return Fy
        else:
            raise Exception("Contact Forces are negative")

    def in_friction_cone(self, mu, Fx, Fy):
        """Checks, wether forces are in the friction cone.

        Args:
            mu (float): Friction coefficient
            Fx (flaot): Forces in x-direction
            Fy (float): Forces in y-direciton

        Returns:
            bool: Returns True if forces are in friction cone.
        """
        return abs(Fy) <= mu * (abs(self.g) * self.m + Fx)

    
if __name__ == "__main__":
    # for testing functions 
    p = HopperPlant(Izz=[0.0014078, 0.00012493], com1=[0.056105, 1.0785E-05], com2=[0.096585, 9.8958E-09], link_length=[0.205, 0.25])
    # p = HopperPlant(Izz=[0.0014078, 0.00012493], com1=[0.056105, 0.039497], com2=[0.096585, -6.9822E-05], link_length=[0.205, 0.25])
    q1 = 1.
    q2 = 1.
    dq1 = 1
    dq2 = 1
    ddq0 = 1
    ddq1 = 1
    ddq2 = 1
    tau1 = 0
    tau2 = 0
    idyn = p.inverse_dynamics(q1, q2, dq1, dq2, ddq0, ddq1, ddq2)
    print(idyn)
    