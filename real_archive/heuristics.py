"""
Heuristics
==========
"""

def contact_detection(effort):
    if abs(effort[0]) >= contact_force_threshold or abs(effort[1]) >= contact_force_threshold:
        return True
    else:
        return False

def estimate_state(state, phase, t0, v0, g = -9.81):
        
    if phase not in ("FLIGHT", "FLIGHT_ASCEND"):
        return plant.forward_kinematics(*state[:2])[0]
    t = time.time() - tstart
    # t0 = self.time_at_liftoff
    x0 = x_des+liftoff_extension
    # v0 = self.liftoff_vel
    # g = -self.plant.g
    x = g/2 *(t**2-t0**2) - g*t0*(t-t0) + v0*(t-t0) + x0
    return x

def get_control(controller, phase, state, effort, desired_height):

    if phase == "FLIGHT":
        h = estimate_state(state,current_phase, t0,v0)          
        if h > maxh:
            maxh = h        
        shoulder_pos_des, elbow_pos_des, shoulder_vel_des, elbow_vel_des, tau1, tau2 = controller.cartesian_stiffness(state, p_d = [x_des,y_des], pd_d = [0.0,0.0], Kpc = (1000,2000), Kdc=(10,10), f_ff = [0.0,0.0], knee_direction=1)
        
        if contact_detection(effort) and plant.forward_kinematics(*state[:2])[0]<=x_des-0.005:
            phase = "TOUCHDOWN"         

    if phase == "TOUCHDOWN":
        #change
        shoulder_pos_des, elbow_pos_des, shoulder_vel_des, elbow_vel_des, tau1, tau2 = controller.cartesian_stiffness(state, p_d = [x_des,y_des], pd_d = [0.0,0.0], Kpc = (10.0,2000), Kdc=(10,10), f_ff = [0.0,0.0], knee_direction=1)

        #change
        if np.linalg.norm(plant.forward_velocity(*state))<0.1 and plant.forward_kinematics(*state[:2])[0]<=x_des:
            phase = "LIFTOFF"
            try:
                height_factor *= (desired_height/maxh)**2        
            except ZeroDivisionError:
                pass
            maxh = 0
            

    elif current_phase == "LIFTOFF":
        # change
        shoulder_pos_des, elbow_pos_des, shoulder_vel_des, elbow_vel_des, tau1, tau2 = controller.cartesian_stiffness(state, p_d = [x_des+liftoff_extension,y_des], pd_d = [np.sqrt(2*abs(plant.g)*(desired_height - (x_des+liftoff_extension))),0.0], Kpc = (0.0,500), Kdc=(10,10), f_ff = [height_factor * (desired_height-.1)*plant.m * abs(plant.g)/0.2,-15.0], knee_direction=1)

        if plant.forward_kinematics(*state[:2])[0]>=x_des+liftoff_extension: 
            v0 = plant.forward_velocity(*state)[0]
            t0 = time.time()-tstart
            phase = "FLIGHT_ASCEND"
        
        #if not contact_detection(effort):
        #    phase = "FLIGHT"

    elif current_phase == "FLIGHT_ASCEND":
        shoulder_pos_des, elbow_pos_des, shoulder_vel_des, elbow_vel_des, tau1, tau2 = controller.cartesian_stiffness(state, p_d = [x_des,y_des], pd_d = [0.0,0.0], Kpc = (1000,2000), Kdc=(10,10), f_ff = [0.0,0.0], knee_direction=1)
        
        if not contact_detection(effort):
            flight_counter = flight_counter + 1
            phase = "FLIGHT"

    return shoulder_pos_des, elbow_pos_des, shoulder_vel_des, elbow_vel_des, tau1, tau2, phase, flight_counter, maxh, height_factor, v0, t0