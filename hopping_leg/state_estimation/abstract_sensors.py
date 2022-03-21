from abc import ABC, abstractmethod


class JointEncoderAbstract(ABC):

    @abstractmethod
    def get_joint_positions(self):
        pass

    @abstractmethod
    def get_joint_velocities(self):
        pass


class ContactSensorAbstract(ABC):

    @abstractmethod
    def get_contact(self):
        pass


class ImuAbstract(ABC):

    @abstractmethod
    def start_imu_scan(self):
        pass

    @abstractmethod
    def get_gyroscope_values(self):
        pass

    @abstractmethod
    def get_accelerometer_values(self):
        pass

    @abstractmethod
    def get_values(self):
        pass


class MotionTrackingAbstract(ABC):

    @abstractmethod
    def get_ground_truth(self):
        pass
