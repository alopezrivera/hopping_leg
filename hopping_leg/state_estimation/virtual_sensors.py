#!/usr/bin/env python
# coding: utf-8

"""
Collection of virtual sensor classes used for hopping leg state estimation based on PyBullet simulator.

List of sensor classes:
    - JointEncoder
    - ContactSensor
    - Imu
    - MotionTracking

Reference: 
    - qiBullet - Bullet-based python simulation for SoftBank Robotics' robots.
    - Repository: https://github.com/softbankrobotics-research/qibullet/tree/486e5943a9886a777eeacdc06e97e323ccd0cc31  
    - License: Apache License 2.0 

Maintainer:
    Mihaela Popescu, mihaela.popescu@dfki.de
"""


import time
import pybullet as p
import threading

from hopping_leg.state_estimation import JointEncoderAbstract, ContactSensorAbstract, ImuAbstract, MotionTrackingAbstract


class JointEncoder(JointEncoderAbstract):
    """
    Class representing a virtual joint encoder.
    """

    def __init__(self, robot_body_unique_id, joint_indices):
        """
        Constructor.
        Parameters:
            robot_body_unique_id - The pybullet model of the robot
            joint_indices - list of robot joint indices
        """
        self.robot_body_unique_id = robot_body_unique_id
        self.joint_indices = joint_indices

    def get_joint_positions(self):
        q1, q2 = [j[0] for j in p.getJointStates(self.robot_body_unique_id, self.joint_indices)]
        return q1, q2

    def get_joint_velocities(self):
        q1_dot, q2_dot = [j[1] for j in p.getJointStates(self.robot_body_unique_id, self.joint_indices)]
        return q1_dot, q2_dot


class ContactSensor(ContactSensorAbstract):
    """
    Class representing a virtual contact sensor.
    """

    def __init__(self, robot_body_unique_id, plane_body_unique_id):
        """
        Constructor.
        Parameters:
            robot_body_unique_id - The pybullet model of the robot
            plane_body_unique_id - The pybullet model of the ground surface
        """
        self.robot_body_unique_id = robot_body_unique_id
        self.plane_body_unique_id = plane_body_unique_id

    def get_contact(self):
        """
        Returns true if there is contact between the robot's second link and the ground plane.
        Returns:
            true - if the robot's second link is in contact with the ground plane
            false - otherwise
        """
        if p.getNumJoints(self.robot_body_unique_id) == 4:
            contact_points = p.getContactPoints(self.robot_body_unique_id, self.plane_body_unique_id, 3, -1)
        else:        
            contact_points = p.getContactPoints(self.robot_body_unique_id, self.plane_body_unique_id, 2, -1)
        if len(contact_points) > 0:
            return 1
        else:
            return 0


class Imu(ImuAbstract):
    """
    Class representing a virtual inertial unit.
    """

    def __init__(self, robot_body_unique_id, imu_link, dt):
        """
        Constructor.
        Parameters:
            robot_body_unique_id - The pybullet model of the robot
            imu_link - the robot link index where the IMU is attached
            dt - The time period of the IMU, in s
        """
        self.robot_body_unique_id = robot_body_unique_id
        self.imu_link = imu_link
        self.dt = dt
        self.angular_velocity = [0.0, 0.0, 0.0]
        self._linear_velocity = [0.0, 0.0, 0.0]  # private class member
        self.linear_acceleration = [0.0, 0.0, 0.0]
        self.values_lock = threading.Lock()

    def start_imu_scan(self):
        """
        Start the IMU scanning process in a new thread
        """
        imu_scan_thread = threading.Thread(target=self._imu_scan)
        imu_scan_thread.start()

    def get_gyroscope_values(self):
        """
        Returns the angular velocity of the IMU in rad/s in the
        world frame
        Returns:
            angular_velocity - The angular velocity in rad/s
        """
        with self.values_lock:
            return self.angular_velocity

    def get_accelerometer_values(self):
        """
        Returns the linear acceleration of the IMU in m/s^2 in the
        world frame
        Returns:
            linear_acceleration - The linear acceleration in m/s^2
        """
        with self.values_lock:
            return self.linear_acceleration

    def get_values(self):
        """
        Returns the values of the gyroscope and the accelerometer of the IMU
        (angular_velocity, linear_acceleration) in the world frame
        Returns:
            angular_velocity - The angular velocity values in rad/s
            linear_acceleration - The linear acceleration values in m/s^2
        """
        with self.values_lock:
            return self.angular_velocity, self.linear_acceleration

    def _imu_scan(self):
        """
        INTERNAL METHOD, retrieves and update the IMU data
        """
        sampling_time = time.time()

        while True:
            current_time = time.time()
            if current_time - sampling_time < self.dt:
                time.sleep(self.dt / 10)  # delete this line, sleep not accurate for low timings
                continue

            link_state = p.getLinkState(
                self.robot_body_unique_id,
                self.imu_link,
                computeLinkVelocity=True)

            # Compute acceleration (derivative of velocity) using finite differences method
            with self.values_lock:
                self.angular_velocity = link_state[7]
                self.linear_acceleration = [
                    (i - j) / (time.time() - sampling_time) for i, j in zip(
                        link_state[6],
                        self._linear_velocity)]
                self._linear_velocity = link_state[6]

            sampling_time = current_time


class MotionTracking(MotionTrackingAbstract):
    """
    Class representing a virtual motion tracking system for ground truth localization.
    """

    def __init__(self, robot_body_unique_id, tracking_link):
        """
        Constructor.
        Parameters:
            robot_body_unique_id - The pybullet model of the robot
            tracking_link - the robot link index where the tracking markers are attached
        """
        self.robot_body_unique_id = robot_body_unique_id
        self.imu_link = tracking_link

    def get_ground_truth(self):
        """
        Returns the position and orientation of the robot rigid body link in the world frame.
        Returns:
            position - world position of the robot URDF link frame, as list of 3 floats, in m
            orientation -  world orientation of the robot URDF link frame, as list of 4 floats in [x,y,z,w] order.
                           Use getEulerFromQuaternion to convert the quaternion to Euler if needed.
        """
        link_state = p.getLinkState(self.robot_body_unique_id, self.imu_link)
        position = link_state[4]
        orientation = link_state[5]
        return position, orientation
