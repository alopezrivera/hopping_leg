#!/usr/bin/env python
# coding: utf-8

import time
import threading
import numpy as np
import virtual_sensors as sensors

"""
    State estimation for a hopping leg robot. This framework estimates the jumping height h of the leg based on 
    IMU measurements, contact sensor, joint encoders and forward kinematics.
    Maintainer:
        Mihaela Popescu, mihaela.popescu@dfki.de
"""


class StateEstimation:
    """
    Class for estimating the state of a hopping leg.
    """

    def __init__(self, robot_body_unique_id, plane_body_unique_id, plant, dt, imu_link=0, l1=0.2, l2=0.2):
        """
        Constructor.
        Parameters:
            robot_body_unique_id - The PyBullet model of the robot
            plane_body_unique_id - The PyBullet model of the ground plane
            frequency - The frequency of the state estimation, in Hz
            imu_link - the robot link index where the IMU is attached
            l1, l2 - length pf robot link 1, link 2
        """
        self.dt = dt
        self.l1 = l1
        self.l2 = l2
        self.x = [0.5, 0]  # state vector [height, vertical_velocity]

        self.contact_sensor = sensors.ContactSensor(robot_body_unique_id, plane_body_unique_id)
        self.joint_encoder = sensors.JointEncoder(robot_body_unique_id, [1, 2])
        self.imu = sensors.Imu(robot_body_unique_id, imu_link, dt)
        self.ground_truth = sensors.MotionTracking(robot_body_unique_id, imu_link)
        self.plant = plant

    def start_imu_scan(self):
        """
        Start the IMU scanning process in a new thread.
        """
        self.imu.start_imu_scan()

    def start_state_estimation(self):
        """
        Start the IMU scanning process in a new thread.
        """
        state_estimation_thread = threading.Thread(target=self._estimate_height)
        state_estimation_thread.start()

    def get_height(self):
        """
        Returns the jumping height h of a hopping robot leg.
        Returns:
            h - height of the robot with respect to the ground plane, in m
        """
        return self.x[0]

    def get_ground_truth(self):
        """
        Returns the position and orientation of the robot rigid body link in the world frame.
        Returns:
            position - world position of the robot URDF link frame, as list of 3 floats, in m
            orientation -  world orientation of the robot URDF link frame, as list of 4 floats in [x,y,z,w] order.
                           Use getEulerFromQuaternion to convert the quaternion to Euler if needed.
        """
        return self.ground_truth.get_ground_truth()

    def get_contact(self):
        """
        Returns true if there is contact between the robot's second link and the ground plane.
        Returns:
            true - if the robot's second link is in contact with the ground plane
            false - otherwise
        """
        return self.contact_sensor.get_contact()

    def get_vertical_acceleration(self):
        """
        Returns the vertical acceleration of the IMU.
        Returns:
            vertical acceleration - the linear acceleration along z-axis, in m/s^2
        """
        return self.imu.get_accelerometer_values()[2]

    def _estimate_height(self):
        """
        Estimates the jumping height h of a hopping robot leg using IMU, contact sensor and joint encoder.
        """
        prev_time = time.time()

        while True:
            # Compute time interval between two updates
            current_time = time.time()
            delta_t = current_time - prev_time
            if delta_t < self.dt:
                time.sleep(self.dt / 10)  # delete this line, sleep not accurate for low timings
                continue
            prev_time = current_time

            # FLIGHT PHASE: estimate height based on integration of vertical IMU acceleration
            method = "runge_kutta"
            if method == "runge_kutta" or method == "euler":
                self.x = self.plant.step(self.x, self.get_vertical_acceleration(), delta_t, method)
            else:
                # Use directly velocity measurements from PyBullet
                self.x[0] = self.x[0] + delta_t * self.imu._linear_velocity[2]

            # STANCE PHASE: estimate height based on contact sensor and joint encoder using forward kinematics
            if self.get_contact():
                q1, q2 = self.joint_encoder.get_joint_positions()
                height = self.l1 * np.cos(q1) + self.l2 * np.cos(q1 + q2)
                self.x[0] = height
