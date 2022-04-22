import os
import sys
import itertools


import matplotlib.pyplot as plt

import pybullet as pb
import pybullet_data
import numpy as np
import pandas 
import time
from time import sleep
import scipy.interpolate as interp

# import pinocchio
# from pinocchio.robot_wrapper import RobotWrapper


client = pb.connect(pb.GUI)
pb.setAdditionalSearchPath(pybullet_data.getDataPath()) #optionally
pb.resetDebugVisualizerCamera( cameraDistance=2, cameraYaw=150, cameraPitch=-15, cameraTargetPosition=[-0.008,0.008,0.64])

ground = pb.loadURDF("plane.urdf")
# Analyse the contact parameter
groundParams = pb.getDynamicsInfo(ground, -1)
# print(groundParams)
print('Pybubllet Ground parameter:')
print('lateralFriction: ' + str(groundParams[1]))
print('restitution: ' + str(groundParams[5]))
print('contactDamping: ' + str(groundParams[8]))
print('contactStiffness: ' + str(groundParams[9]))
pb.changeDynamics(ground, -1, lateralFriction=0.7, restitution=.97)
print(pb.getDynamicsInfo(ground, -1))
pb.setGravity(0, 0, -9.81)

pb.setTimeStep(0.001)


# robot = pb.loadURDF("./urdf/Lower_Body.urdf", [0, 0, 0.9], flags=pb.URDF_USE_INERTIA_FROM_FILE) #default flag: collision off
# robot = pb.loadURDF("../../rh5-models/abstract-urdf/urdf/RH5Humanoid_PkgPath_FixedArmsNHead.urdf", [0, 0, 0.91], flags=pb.URDF_USE_INERTIA_FROM_FILE) 
initPos = [0, 0, 0.353]
intiOri = [0, 0, 0, 1]
filename="/home/dfki.uni-bremen.de/mboukheddimi/DEVEL/DFKI/OC_UNDERACTUATED/hopping_leg/Controllers/OC_FDDP/without_rails/urdf/v6_new_joint_limits_pino.urdf"
# robot = pb.loadURDF("../../rh5-models/abstract-urdf/urdf/RH5Humanoid_PkgPath_FixedArmsNHead.urdf", basePosition=initPos, baseOrientation=intiOri, flags=pb.URDF_USE_INERTIA_FROM_FILE) 
robot = pb.loadURDF(filename, basePosition=initPos, flags=pb.URDF_USE_INERTIA_FROM_FILE, useFixedBase=0)

cid = pb.createConstraint(robot, -1, -1, -1, pb.JOINT_PRISMATIC, [0, 0, 1], [0, 0, 0], [0, 0, 0])   #


#, useFixedBase=1

#gravId = pb.addUserDebugParameter("gravity",0.,0.,-9.81)

# Load the data
simName = '/home/dfki.uni-bremen.de/mboukheddimi/DEVEL/DFKI/OC_UNDERACTUATED/hopping_leg/Controllers/OC_FDDP/results/withoutRAILPZ/2hops/logs/'
# simName = 'results/Jump_Forward_NoJLim/'
# simName = 'results/DynamicWalking_Test_Fast/'
saveDir = simName + 'pybullet/without_railsPZ/'
if not os.path.exists(saveDir):
                os.makedirs(saveDir)
#data = pandas.read_csv(simName + 'logs/logJointSpace_filled.csv')
data = pandas.read_csv(simName + 'logJointSpace.csv')
base_data = pandas.read_csv(simName + 'logBase.csv')
#data = pandas.read_csv(simName + 'logJointSpace_filled_interp.csv')
#base_data = pandas.read_csv(simName + 'logBase_interp.csv')

total_time = data['t[s]'][data['t[s]'].size-1]-data['t[s]'][0]

print("Time range:" + str(total_time) + ' [s]')







# Get the states
joints = {}
for i in range(pb.getNumJoints(robot)):
        jinfo = pb.getJointInfo(robot, i)
        joints.update(
            {jinfo[1].decode("utf-8") : i }
        )
print(pb.getNumJoints(robot))
print(joints)




# Get all the data for one joint and store it into a dict
joint_trajectories = {}
fixed_joint_trajectories = {}
removes = []

for jname in joints.keys():
    try:
        joint_trajectories.update(
            {jname : data[['q_'+jname, 'qd_'+jname, 'Tau_'+jname]].to_numpy()}
        )
    except Exception as e: 
        removes.append(jname)

for v in removes:
    joints.pop(v)
print(joints)
# Set joints to desired initial configuration
for jn in joints.keys():
    pb.resetJointState(robot, joints[jn], joint_trajectories[jn][0,0])




fixedJoints = []

fixedJointsConfig = []                   # head

fixedJointIDs = []
for jnn in fixedJoints:
    fixedJointIDs.append(joints[jnn])

print ("fixedJointIDs", fixedJointIDs)

#fixedJointIDs = [ 8, 9, 10, 
#                  17, 18, 19]

for jnID, i in zip(fixedJointIDs, range(len(fixedJointsConfig))): 
    pb.resetJointState(robot, jnID, fixedJointsConfig[i])



pos_interp = {jn: interp.CubicHermiteSpline(data["t[s]"], joint_trajectories[jn][:,0], joint_trajectories[jn][:,1]) for jn in joints.keys()}
#print(joint_trajectories['LRHip3'][:,0])
#print(pos_interp['LRHip3'](1))
vel_interp = {jn: interp.CubicSpline(data["t[s]"], joint_trajectories[jn][:,1], bc_type="not-a-knot")
              for jn in joints.keys()}
tau_interp = {jn: interp.CubicSpline(data["t[s]"], joint_trajectories[jn][:,2], bc_type="not-a-knot")
              for jn in joints.keys()}

base_coord_names = ['Z', 
                    'vz',
                    'vzd']

base_interp = {coord: interp.CubicSpline(data["t[s]"], base_data[coord], bc_type="not-a-knot")
              for coord in base_coord_names}

des_positions, act_positions = [], []
des_velocities = []
des_torques, act_torques = [], [] 
des_base_trajectories, act_base_positions = [], [] 

time_step = pb.getPhysicsEngineParameters()['fixedTimeStep']

for t in np.arange(0, total_time + time_step, time_step):

    desPositions = [np.asscalar(pos_interp[jn](t)) for jn in joints.keys()]
    desVelocities = [np.asscalar(vel_interp[jn](t)) for jn in joints.keys()]
    desTorques = [np.asscalar(tau_interp[jn](t)) for jn in joints.keys()]
    base_trajectory = [np.asscalar(base_interp[coord](t)) for coord in base_coord_names]
    des_positions.append(desPositions)
    des_velocities.append(desVelocities)
    des_torques.append(desTorques)
    des_base_trajectories.append(base_trajectory)



# Load the model with pinocchio
# modelPath = os.path.join(os.environ.get('HOME'), "Dev/rh5-models")
# URDF_FILENAME = "RH5Humanoid_PkgPath_FixedArmsNHead.urdf"
# URDF_SUBPATH = "/abstract-urdf-deprecated/urdf/" + URDF_FILENAME
# rh5_robot = RobotWrapper.BuildFromURDF(modelPath + URDF_SUBPATH, [modelPath], pinocchio.JointModelFreeFlyer())
# rmodel = rh5_robot.model
# print(rmodel.effortLimit)

for jn in joints.keys(): 
    print(jn)





'''
from pinocchio effort limits the 7 first values ae for the free flayer, the data are extracted from the urdf
In [1]: modelPath = os.path.join(os.environ.get('HOME'), "DEVEL/DFKI/rh5_V2_abstract")
   ...: URDF_FILENAME = "RH5v2_pino_abs_solar.urdf"
   ...: URDF_SUBPATH = "/urdf/" + URDF_FILENAME
   ...: 
   ...: # Load the full model 
   ...: robot2 = RobotWrapper.BuildFromURDF(modelPath + URDF_SUBPATH, [modelPath], pinocchio.JointModelFreeFlyer())
   ...: rmodel1 = robot2.model
   ...: print(rmodel1.effortLimit)


'''


#print('RH5V2 Torque Limits: ' + str(max_torque_from_URDF))

base_pose = pb.getBasePositionAndOrientation(robot)
#base_offset = list(np.array(base_pose[0]) - np.array(initPos))
#base_pose = pb.getBasePosition(robot)
base_offset = list(np.array(base_pose[0]) - np.array(initPos))

print("base_pose:", base_pose)

print("base_offset:", base_offset)


# q0 = np.matrix([0,0,0.8793,0,0,0,1,     
#                 0,0,-0.33,0.63,0,-0.30,       
#                 0,0,-0.33,0.63,0,-0.30]).T

# # Stabilize desired initial position for few seconds
# pb.setJointMotorControlArray(
# robot, [i for i in  joints.values()], pb.POSITION_CONTROL,
# targetPositions = [joint_trajectories[jn][0,0] for jn in joints.keys()]
# )
# pb.setRealTimeSimulation(1)

save_id = pb.saveState()



# Set control mode
pb.restoreState(save_id)

logID = pb.startStateLogging(pb.STATE_LOGGING_VIDEO_MP4, "hopPZ2.mp4") # TODO: Improvements: (i) Increase video quality; switching to full-screen crashes (ii) Match video length with velocity; too fast now. 
pb.setRealTimeSimulation(0)
# pb.resetBasePositionAndOrientation(robot, posObj=[0, 0, 0.89], ornObj=[0, 0, 0, 1])
# sleep(0.5)
time = 0
count = 0
time_step = pb.getPhysicsEngineParameters()['fixedTimeStep']

# fast_forward=0.03/0.008
while time < total_time:

    #************************************************************************
    pb.getCameraImage(320,200)
    #pb.setGravity(0,0,pb.readUserDebugParameter(gravId))

    #************************************************************************


    # TORQUE_CONTROL (array version)
    # pb.setJointMotorControlArray(
    #     robot, [idx for idx in  joints.values()], pb.VELOCITY_CONTROL,
    #     targetVelocities = [0 for jn in joints.keys()], 
    #     forces = [0 for jn in joints.keys()]
    # )
    # pb.setJointMotorControlArray(
    #     robot, [idx for idx in joints.values()], pb.TORQUE_CONTROL,
    #     forces = des_torques[count]
    # )
    # TORQUE_CONTROL (non-array version)
    # for name in joints.keys():
    #     idx = joints[name]
    #     pb.setJointMotorControl2(robot, idx, pb.VELOCITY_CONTROL, targetVelocity=0, force=0)
    # for name, j in zip(joints.keys(), range(len(joints))):
    #     idx = joints[name]
    #     pb.setJointMotorControl2(robot, idx, pb.TORQUE_CONTROL, force=des_torques[count][j])
        
    # POSITION_CONTROL
    pb.setJointMotorControlArray(
        robot, [idx for idx in  joints.values()], pb.POSITION_CONTROL,
        targetPositions = des_positions[count],
        targetVelocities = des_velocities[count]
        #forces = max_torque_from_URDF
    )


    #positionGains=[3, 3], 
    #velocityGains=[10, 10]
   
    time += time_step 
    count += 1

    pb.stepSimulation()
    #sleep(time_step)

    joint_states = pb.getJointStates(robot, [idx for idx in  joints.values()])
    # get joint states
    joint_positions = [state[0] for state in joint_states]
    act_positions.append(joint_positions)
    # get joint torques
    joint_torques = [state[3] for state in joint_states]
    act_torques.append(joint_torques)
    # get base pose
    base_pose = pb.getBasePositionAndOrientation(robot)
    act_base_positions.append(list(np.array(base_pose[0]) - np.array(base_offset))) # compensate for static offset between pb base and crocoddyl base definition
    #act_base_positions.append(list(np.array(base_pose[0]))) 
# pb.stopStateLogging(logID)
# print("Replay velocity", fast_forward, "x")

# pb.setRealTimeSimulation(1)
# while(True):
#      pb.setJointMotorControlArray(
#         robot, [idx for idx in  joints.values()], pb.POSITION_CONTROL,
#         targetPositions = des_positions[-1],
#         targetVelocities = des_velocities[-1]
#     )



'''

print(joint_states)
print('###########################')
print(joint_states[0])
print(joint_states[1])
print(joint_states[2])
print(joint_states[3])
print(joint_torques)
print('###########################')
print(joint_states[4])
print(joint_states[5])
print(joint_states[6])
print(joint_states[7])
print(joint_states[8])

'''



# Convert trajectories for plotting
nx = len(des_positions[0])
nu = len(des_torques[0])
X_des, X_act = [0.] * nx, [0.] * nx
U_des, U_act = [0.] * nu, [0.] * nu 
X_base_trajecory_des, X_basePosition_act = [0.] * 1, [0.] * 1
for i in range(nx):
        X_des[i] = [x[i] for x in des_positions]
        X_act[i] = [x[i] for x in act_positions]
for i in range(nu):
        U_des[i] = [u[i] for u in des_torques]
        U_act[i] = [u[i] for u in act_torques]
for i in range(3):
        X_base_trajecory_des[i] = [x[i] for x in des_base_trajectories]
for i in range(3):
        X_basePosition_act[i] = [x[i] for x in act_base_positions]



#**********************************************************************************************************************
#**********************************************************************************************************************

'''
# Plotting joint tracking error


torsoJointNames = ['BodyPitch','BodyRoll','BodyYaw']
#legJointNames = ['shoulder1', 'shoulder2', 'shoulder3', 'Elbow', 'wrist1', 'wrist2l', 'wrist3'] 
legJointNames = ['shoulder1', 'shoulder2', 'shoulder3', 'Elbow'] 

wristsJointNames = ['WristRoll', 'WristPitch', 'WristYaw']

HeadJointNames = ['HeadPitch', 'HeadRoll', 'HeadYaw']


#**********************************************************************************************************************
#**********************************************************************************************************************


plt.figure(1, figsize=(16,13)) # (16,9) for bigger headings

ax1 = plt.subplot(3, 1, 1)
[plt.plot(X_des[k], label=torsoJointNames[i]+'_des', linestyle=':') for i, k in enumerate(range(0, 3))]
plt.gca().set_prop_cycle(None) # Reset used colors
[plt.plot(X_act[k], label=torsoJointNames[i]+'_act') for i, k in enumerate(range(0, 3))]
plt.setp(ax1.get_xticklabels(), visible=False) # Don't show x numbers
plt.legend(loc='upper right')
plt.ylabel('Body Pitch, Roll ,Yaw')

# left foot
ax2 = plt.subplot(3, 1, 2)
[plt.plot(X_des[k], label=legJointNames[i]+'_des', linestyle=':') for i, k in enumerate(range(3, 6))]
plt.gca().set_prop_cycle(None)
[plt.plot(X_act[k], label=legJointNames[i]+'_act') for i, k in enumerate(range(3, 6))]
plt.setp(ax2.get_xticklabels(), visible=False)
plt.ylabel('LF Shoulder rotations')
plt.legend(loc='upper right')

# right foot
plt.subplot(3, 1, 3)
[plt.plot(X_des[k], label=legJointNames[3]+'_des', linestyle=':') for i, k in enumerate(range(6, 7))]
plt.gca().set_prop_cycle(None)
[plt.plot(X_act[k], label=legJointNames[3]+'_act') for i, k in enumerate(range(6, 7))]
plt.ylabel('LF Elbow rotations')
plt.xlabel('t [ms] (joint Rotation)')
plt.legend(loc='upper right')

plt.savefig(saveDir + 'pybulletTracking_body_and_LF.pdf', facecolor='w', dpi = 300, bbox_inches='tight')
plt.show()

#**********************************************************************************************************************
#**********************************************************************************************************************

plt.figure(2, figsize=(16,13)) # (16,9) for bigger headings

ax1 = plt.subplot(3, 1, 1)
[plt.plot(X_des[k], label=HeadJointNames[i]+'_des', linestyle=':') for i, k in enumerate(range(36, 39))]
plt.gca().set_prop_cycle(None) # Reset used colors
[plt.plot(X_act[k], label=HeadJointNames[i]+'_act') for i, k in enumerate(range(36, 39))]
plt.setp(ax1.get_xticklabels(), visible=False) # Don't show x numbers
plt.legend(loc='upper right')
plt.ylabel('Head Pitch, Roll ,Yaw')

# left foot
ax2 = plt.subplot(3, 1, 2)
[plt.plot(X_des[k], label=legJointNames[i]+'_des', linestyle=':') for i, k in enumerate(range(23, 26))]
plt.gca().set_prop_cycle(None)
[plt.plot(X_act[k], label=legJointNames[i]+'_act') for i, k in enumerate(range(23, 26))]
plt.setp(ax2.get_xticklabels(), visible=False)
plt.ylabel('RF Shoulder rotations')
plt.legend(loc='upper right')


# right foot
plt.subplot(3, 1, 3)
[plt.plot(X_des[k], label=legJointNames[3]+'_des', linestyle=':') for i, k in enumerate(range(26, 27))]
plt.gca().set_prop_cycle(None)
[plt.plot(X_act[k], label=legJointNames[3]+'_act') for i, k in enumerate(range(26, 27))]
plt.ylabel('RF Elbow rotations')
plt.xlabel('t [ms] (joint Rotation)')
plt.legend(loc='upper right')

plt.savefig(saveDir + 'pybulletTracking_body_and_RF.pdf', facecolor='w', dpi = 300, bbox_inches='tight')
plt.show()


#**********************************************************************************************************************
#**********************************************************************************************************************

plt.figure(3, figsize=(16,13)) # (16,9) for bigger headings


# left arm
ax1 = plt.subplot(2, 1, 1)
[plt.plot(X_des[k], label=wristsJointNames[i]+'_des', linestyle=':') for i, k in enumerate(range(7, 10))]
plt.gca().set_prop_cycle(None)
[plt.plot(X_act[k], label=wristsJointNames[i]+'_act') for i, k in enumerate(range(7, 10))]
plt.setp(ax2.get_xticklabels(), visible=False)
plt.ylabel('LF WRIST')
plt.legend(loc='upper right')

# right arm
plt.subplot(2, 1, 2)
[plt.plot(X_des[k], label=wristsJointNames[i]+'_des', linestyle=':') for i, k in enumerate(range(27, 30))]
plt.gca().set_prop_cycle(None)
[plt.plot(X_act[k], label=wristsJointNames[i]+'_act') for i, k in enumerate(range(27, 30))]
plt.ylabel('RF WRIST')
plt.xlabel('t [ms] (joint Rotation)')
plt.legend(loc='upper right')

plt.savefig(saveDir + 'pybulletTracking_Wrist.pdf', facecolor='w', dpi = 300, bbox_inches='tight')
plt.show()


#**********************************************************************************************************************
#**********************************************************************************************************************

plt.figure(4, figsize=(16,13)) # (16,9) for bigger headings

ax1 = plt.subplot(3, 1, 1)
[plt.plot(U_des[k], label=torsoJointNames[i]+'_des', linestyle=':') for i, k in enumerate(range(0, 3))]
plt.gca().set_prop_cycle(None) # Reset used colors
[plt.plot(U_act[k], label=torsoJointNames[i]+'_act') for i, k in enumerate(range(0, 3))]
plt.setp(ax1.get_xticklabels(), visible=False) # Don't show x numbers
plt.legend(loc='upper right')
plt.ylabel('Body Pitch, Roll ,Yaw torques')

# left foot
ax2 = plt.subplot(3, 1, 2)
[plt.plot(U_des[k], label=legJointNames[i]+'_des', linestyle=':') for i, k in enumerate(range(3, 6))]
plt.gca().set_prop_cycle(None)
[plt.plot(U_act[k], label=legJointNames[i]+'_act') for i, k in enumerate(range(3, 6))]
plt.setp(ax2.get_xticklabels(), visible=False)
plt.ylabel('LF Shoulder torques')
plt.legend(loc='upper right')


# right foot
plt.subplot(3, 1, 3)
[plt.plot(U_des[k], label=legJointNames[3]+'_des', linestyle=':') for i, k in enumerate(range(6, 7))]
plt.gca().set_prop_cycle(None)
[plt.plot(U_act[k], label=legJointNames[3]+'_act') for i, k in enumerate(range(6, 7))]
plt.ylabel('LF Elbow TORQUES')
plt.xlabel('t [ms]   (joint torques)')
plt.legend(loc='upper right')

plt.savefig(saveDir + 'pybulletTORQUES_body_and_LF.pdf', facecolor='w', dpi = 300, bbox_inches='tight')
plt.show()

#**********************************************************************************************************************
#**********************************************************************************************************************

plt.figure(5, figsize=(16,13)) # (16,9) for bigger headings

ax1 = plt.subplot(3, 1, 1)
[plt.plot(U_des[k], label=HeadJointNames[i]+'_des', linestyle=':') for i, k in enumerate(range(36, 39))]
plt.gca().set_prop_cycle(None) # Reset used colors
[plt.plot(U_act[k], label=HeadJointNames[i]+'_act') for i, k in enumerate(range(36, 39))]
plt.setp(ax1.get_xticklabels(), visible=False) # Don't show x numbers
plt.legend(loc='upper right')
plt.ylabel('Head Pitch, Roll ,Yaw torques')

# left foot
ax2 = plt.subplot(3, 1, 2)
[plt.plot(U_des[k], label=legJointNames[i]+'_des', linestyle=':') for i, k in enumerate(range(23, 26))]
plt.gca().set_prop_cycle(None)
[plt.plot(U_act[k], label=legJointNames[i]+'_act') for i, k in enumerate(range(23, 26))]
plt.setp(ax2.get_xticklabels(), visible=False)
plt.ylabel('RF Shoulder torques')
plt.legend(loc='upper right')

# right foot
plt.subplot(3, 1, 3)
[plt.plot(U_des[k], label=legJointNames[3]+'_des', linestyle=':') for i, k in enumerate(range(26, 27))]
plt.gca().set_prop_cycle(None)
[plt.plot(U_act[k], label=legJointNames[3]+'_act') for i, k in enumerate(range(26, 27))]
plt.ylabel('RF Elbow torques')
plt.xlabel('t [ms] (joint torques')
plt.legend(loc='upper right')

plt.savefig(saveDir + 'pybulletTorque_body_and_RF.pdf', facecolor='w', dpi = 300, bbox_inches='tight')
plt.show()


#**********************************************************************************************************************
#**********************************************************************************************************************

plt.figure(6, figsize=(16,13)) # (16,9) for bigger headings


# left arm
ax1 = plt.subplot(2, 1, 1)
[plt.plot(U_des[k], label=wristsJointNames[i]+'_des', linestyle=':') for i, k in enumerate(range(7, 10))]
plt.gca().set_prop_cycle(None)
[plt.plot(U_act[k], label=wristsJointNames[i]+'_act') for i, k in enumerate(range(7, 10))]
plt.setp(ax2.get_xticklabels(), visible=False)
plt.ylabel('LF WRIST')
plt.legend(loc='upper right')

# right arm
plt.subplot(2, 1, 2)
[plt.plot(U_des[k], label=wristsJointNames[i]+'_des', linestyle=':') for i, k in enumerate(range(27, 30))]
plt.gca().set_prop_cycle(None)
[plt.plot(U_act[k], label=wristsJointNames[i]+'_act') for i, k in enumerate(range(27, 30))]
plt.ylabel('RF WRIST')
plt.xlabel('t [ms] (joint torques')
plt.legend(loc='upper right')

plt.savefig(saveDir + 'pybulletTracking_Wrist.pdf', facecolor='w', dpi = 300, bbox_inches='tight')
plt.show()


#**********************************************************************************************************************
#**********************************************************************************************************************

#**********************************************************************************************************************
#**********************************************************************************************************************


# Plotting floating base difference
plt.figure(7, figsize=(16,9))
baseTranslationNames = ['X', 'Y', 'Z']
# [plt.plot(X_basePosition_des[k], label=baseTranslationNames[i], linestyle=':') for i, k in enumerate(range(0, 3))]
# [plt.plot(X_basePosition_act[k], label=baseTranslationNames[i]) for i, k in enumerate(range(0, 3))]
ax1 = plt.subplot(3, 1, 1)
plt.plot(X_base_trajecory_des[0], label='des', linestyle=':')
plt.plot(X_basePosition_act[0], label='act')
plt.setp(ax1.get_xticklabels(), visible=False)
plt.ylabel('X [m]')
plt.legend()
ax2 = plt.subplot(3, 1, 2)
plt.plot(X_base_trajecory_des[1], linestyle=':')
plt.plot(X_basePosition_act[1])
plt.setp(ax2.get_xticklabels(), visible=False)
plt.ylabel('Y [m]')
ax3 = plt.subplot(3, 1, 3)
plt.plot(X_base_trajecory_des[2], linestyle=':')
plt.plot(X_basePosition_act[2])
plt.ylabel('Z [m]')
plt.xlabel('t [ms]')
plt.savefig(saveDir + 'pybulletBase.pdf', bbox_inches = 'tight', dpi = 300)
plt.show()

#**********************************************************************************************************************
#**********************************************************************************************************************

# # For fast executing: Convert .ipynb to .pylint.d
# # $ jupyter nbconvert --to python StabilizeOptimalTrajectories.ipynb

'''










