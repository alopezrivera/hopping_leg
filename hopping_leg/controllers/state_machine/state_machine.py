import sys

from numpy import empty
sys.path.append("../low_level_control/low_level_control.py")
from low_level_control import LowLevelControl 

class AbstractStateMachine(LowLevelControl):
    TOUCHDOWN = "TOUCHDOWN"
    FLIGHT = "FLIGHT"
    FLIGHT_ASCEND = "FLIGHT_ASCEND"
    LIFTOFF = "LIFTOFF"
    PHASE_DICT = {FLIGHT:0,TOUCHDOWN:1,LIFTOFF:2,FLIGHT_ASCEND:3}
    
    def __init__(self, plant, flight_control_dict={},touchdown_control_dict={},liftoff_control_dict={},flight_ascend_control_dict={}) -> None:
        """State machine

        Args:
            plant (class): HopperPlant
            flight_control_dict (dict, optional): dictionary with combined_stiffness gains for flight phase. Defaults to {}.
            touchdown_control_dict (dict, optional): dictionary with combined_stiffness gains for touchdown phase. Defaults to {}.
            liftoff_control_dict (dict, optional): dictionary with combined_stiffness gains for liftoff phase. Defaults to {}.
            flight_ascend_control_dict (dict, optional): dictionary with combined_stiffness gains for flight_ascend phase. Defaults to {}.
        """
        self.plant = plant
        self.phase = self.LIFTOFF
        self.flight_control_dict = flight_control_dict
        self.touchdown_control_dict = touchdown_control_dict
        self.liftoff_control_dict = liftoff_control_dict
        self.flight_ascend_control_dict = flight_ascend_control_dict
        if not self.flight_ascend_control_dict:
            self.flight_ascend_control_dict = self.flight_control_dict
    
    def do_liftoff(self, state, **kwargs):
        """Command to do during liftoff

        Args:
            state (list): [q1,q2,dq1,dq2]

        Returns:
            tuple: return values of combined_stiffness. Currently tau1, tau2
        """
        return self.combined_stiffness(state,**self.liftoff_control_dict)
        
    def do_flight_ascend(self, state, **kwargs):
        """Command to do during flight ascend

        Args:
            state (list): [q1,q2,dq1,dq2]

        Returns:
            tuple: return values of combined_stiffness. Currently tau1, tau2
        """
        
        if self.flight_ascend_control_dict is empty:
            return self.do_flight(state, **kwargs)
        return self.combined_stiffness(state,**self.flight_ascend_control_dict)
        
    def do_flight(self, state, **kwargs):
        """Command to do during flight

        Args:
            state (list): [q1,q2,dq1,dq2]

        Returns:
            tuple: return values of combined_stiffness. Currently tau1, tau2
        """
        return self.combined_stiffness(state,**self.flight_control_dict)
        
    def do_touchdown(self, state, **kwargs):
        """Command to do during touchdown

        Args:
            state (list): [q1,q2,dq1,dq2]

        Returns:
            tuple: return values of combined_stiffness. Currently tau1, tau2
        """
        return self.combined_stiffness(state,**self.touchdown_control_dict)
        
    def prepare_liftoff(self, state, **kwargs):
        """What to do between touchdown and liftoff

        Args:
            state (list): [q1,q2,dq1,dq2]
        """
        pass
        
    def prepare_flight_ascend(self, state, **kwargs):
        """What to do between liftoff and flight ascend

        Args:
            state (list): [q1,q2,dq1,dq2]
        """
        pass
        
    def prepare_flight(self, state, **kwargs):
        """What to do between flight ascend and flight 

        Args:
            state (list): [q1,q2,dq1,dq2]
        """
        pass
    
    def prepare_touchdown(self, state, **kwargs):
        """What to do between flight and touchdown

        Args:
            state (list): [q1,q2,dq1,dq2]
        """
        pass
    
    def liftoff_condition(self, state, **kwargs):
        """Condition for transition to liftoff phase

        Args:
            state (list): [q1,q2,dq1,dq2]

        Returns:
            bool: Condition fullfilled?
        """
        return True
    
    def flight_ascend_condition(self, state, **kwargs):
        """Condition for transition to flight ascend phase

        Args:
            state (list): [q1,q2,dq1,dq2]

        Returns:
            bool: Condition fullfilled?
        """
        return self.flight_condition(state, **kwargs) # skips flight_ascend
        
    def flight_condition(self, state, **kwargs):
        """Condition for transition to flight phase

        Args:
            state (list): [q1,q2,dq1,dq2]

        Returns:
            bool: Condition fullfilled?
        """
        return True
    
    def touchdown_condition(self, state, **kwargs):
        """Condition for transition to touchdown phase

        Args:
            state (list): [q1,q2,dq1,dq2]

        Returns:
            bool: Condition fullfilled?
        """
        return True
    
    
    def do_control(self, state, **kwargs):
        """Calculate actual motor torques and keep track off the current phase.

        Args:
            state (list): [q1,q2,dq1,dq2]

        Returns:
            tuple: return values of combined_stiffness. Currently tau1, tau2
        """
        if self.phase == self.LIFTOFF:
            tau = self.do_liftoff(state, **kwargs)
            if self.flight_ascend_condition(state, **kwargs):
                self.phase = self.FLIGHT_ASCEND
                self.prepare_flight_ascend(state, **kwargs)
        elif self.phase == self.FLIGHT_ASCEND:
            tau = self.do_flight_ascend(state, **kwargs)
            if self.flight_condition(state, **kwargs):
                self.phase = self.FLIGHT
                self.prepare_flight(state, **kwargs)
        # may skip flight ascend phase
        if self.phase == self.FLIGHT:
            tau = self.do_flight(state, **kwargs)
            if self.touchdown_condition(state, **kwargs):
                self.phase = self.TOUCHDOWN
                self.prepare_touchdown(state, **kwargs)
        elif self.phase == self.TOUCHDOWN:
            tau = self.do_touchdown(state,**kwargs)
            if self.liftoff_condition(state, **kwargs):
                self.phase = self.LIFTOFF
                self.prepare_liftoff(state, **kwargs)
        return tau
        
           
# Example class how to use the abstract state machine
class GainScheduling(AbstractStateMachine):
    def __init__(self, plant, se, flight_control_dict={},touchdown_control_dict={},liftoff_control_dict={},flight_ascend_control_dict={}) -> None:
        # call init of super class
        super().__init__(plant, flight_control_dict, touchdown_control_dict, liftoff_control_dict, flight_ascend_control_dict)
        self.se = se
        # predefine some gains
        if not flight_control_dict:
            self.flight_control_dict = {"Kpj" : (100,600), "Kdj" : (4,4), "q_d" : self.plant.inverse_kinematics(.30,0), "Kdc" :(100,20)}
        
        if not touchdown_control_dict:
            self.touchdonw_control_dict = {"Kpj" : (100,400), "q_d" : self.plant.inverse_kinematics(.30,0), "Kdc" :(100,40)}

        if not liftoff_control_dict:
            self.liftoff_control_dict = {"Kpj" : (1000,1000), "q_d" : self.plant.inverse_kinematics(.33,0), "Kdc" : (0,20), "pd_d" : (0,0)}    
            
    def flight_condition(self, state, **kwargs):
        return not self.se.get_contact() and self.plant.forward_kinematics(*state[:2])[0]>.33
    
    def touchdown_condition(self, state, **kwargs):
        return self.se.get_contact()
    
    def liftoff_condition(self, state, **kwargs):
        return self.plant.forward_velocity(*state)[0]>=-0.01    