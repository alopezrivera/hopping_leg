import sys

import eigenpy  #""""""""""""""""" this is a modification
#eigenpy.switchToNumpyMatrix()  #""""""""""""""""" this is a modification
eigenpy.switchToNumpyArray() # Matrix deprecated -> this should ensure that everything is treated/converted in array

import subprocess
import time
import numpy as np
import os
import sys
import math
import matplotlib.pyplot as plt
from math import pi
import pinocchio as se3

import crocoddyl

import example_robot_data
from crocoddyl.utils.pendulum import CostModelDoublePendulum, ActuationModelDoublePendulum
from pinocchio.robot_wrapper import RobotWrapper

import pinocchio
from pinocchio.utils import *
from pinocchio import GeometryModel, GeometryObject
import pinocchio as pin

from canmotorlib import CanMotorController

from scipy.signal import lfilter, lfilter_zi, filtfilt, butter

from crocoddyl.utils.quadruped import SimpleQuadrupedalGaitProblem, plotSolution

from utilsV2 import setLimits, calcAverageCoMVelocity, plotSolution, logSolution, addObstacleToViewer




class TaskSE3:
    def __init__(self, oXf, frameId):
        self.oXf = oXf
        self.frameId = frameId



def plotSolution(rmodel, rdata , xs, us, figIndex=1, show=True):
    import matplotlib.pyplot as plt
    from matplotlib.patches import Rectangle
    # Getting the state and control trajectories
    nx, nq, nu = xs[0].shape[0], rmodel.nq, us[0].shape[0]


    X = [0.] * nx
    U = [0.] * nu
    pFCALR = []
    pFCALL = []
    for i in range(nx):
        X[i] = [x[i] for x in xs]
    for i in range(nu):
        #U[i] = [(u[i]/mass) for u in us]   # [Nm/kg]
        U[i] = [(u[i]) for u in us]

    np.save('torques.npy', U)

    # Plotting the joint rotations

 
    legJointNames = ['hip', 'knee']
    #legJointNames = ['shoulder_1_Roll', 'Shoulder_2_pitch', 'shoulder_3_yaw', 'Elbow', 'Body', 'Body']


    #**************************************************************************************************************
    #plt.figure(figIndex)
    plt.figure(figIndex, figsize=(16,9))
    # left arm
    
    #plt.subplot(2, 3, 1)
    plt.subplot(1, 2, 1)
    plt.title('joint rotation hip [rad]')
    [plt.plot(X[k], label=legJointNames[0]) for i, k in enumerate(range(1, 2))]
    plt.axhline(y=3.14, linestyle='--', label= 'Upper joint limits from URDF')
    plt.axhline(y=-3.14, linestyle='--', label= 'Lower joint limits from URDF')
    plt.ylabel('LA')
    plt.xlabel('knots')
    plt.legend()
    plt.grid(True)

    plt.subplot(1, 2, 2)
    plt.title('joint rotation knee [rad]')
    [plt.plot(X[k], label=legJointNames[1]) for i, k in enumerate(range(2, 3))]
    plt.axhline(y=3.14, linestyle='--', label= 'Upper joint limits from URDF')
    plt.axhline(y=-3.14, linestyle='--', label= 'Lower joint limits from URDF')
    plt.ylabel('LA')
    plt.xlabel('knots')
    plt.legend()
    plt.grid(True)    





    #**************************************************************************************************************

    #Left arm TORQUES
    plt.figure(figIndex+3, figsize=(16,9))

    plt.subplot(1, 2, 1)
    plt.title('joint torque shoulder1_l_roll [Nm]')
    [plt.plot(U[k], label=legJointNames[0]) for i, k in enumerate(range(0, 1))]
    plt.axhline(y=12, linestyle='--', label= 'Upper Torques limits from URDF')
    plt.axhline(y=-12, linestyle='--', label= 'Lower Torques limits from URDF')
    plt.ylabel('LA')
    plt.xlabel('knots')
    plt.legend()
    plt.grid(True)

    plt.subplot(1, 2, 2)
    plt.title('joint torque shoulder2_l_pitch [Nm]')
    [plt.plot(U[k], label=legJointNames[1]) for i, k in enumerate(range(1, 2))]
    plt.axhline(y=12, linestyle='--', label= 'Upper Torques limits from URDF')
    plt.axhline(y=-12, linestyle='--', label= 'Lower Torques limits from URDF')
    plt.ylabel('LA')
    plt.xlabel('knots')
    plt.legend()
    plt.grid(True)


    #**************************************************************************************************************



    #plt.figure(figIndex)
    plt.figure(figIndex+7, figsize=(16,9))
    # left arm
    
    #plt.subplot(2, 3, 1)
    plt.subplot(1, 2, 1)
    plt.title('joint velocity hip [rad/s]')
    [plt.plot(X[k], label=legJointNames[0]) for i, k in enumerate(range(4, 5))]
    plt.axhline(y=38, linestyle='--', label= 'Upper joint limits from URDF')
    plt.axhline(y=-38, linestyle='--', label= 'Lower joint limits from URDF')
    plt.ylabel('LA')
    plt.xlabel('knots')
    plt.legend()
    plt.grid(True)

    plt.subplot(1, 2, 2)
    plt.title('joint velocity knee [rad/s]')
    [plt.plot(X[k], label=legJointNames[1]) for i, k in enumerate(range(5, 6))]
    plt.axhline(y=38, linestyle='--', label= 'Upper joint limits from URDF')
    plt.axhline(y=-38, linestyle='--', label= 'Lower joint limits from URDF')
    plt.ylabel('LA')
    plt.xlabel('knots')
    plt.legend()
    plt.grid(True)    


    #**************************************************************************************************************


    #CoM trajectories

    rdata = rmodel.createData()
    Cx = []
    Cy = []
    Cz = []
    Hand_L = []
    Hand_R = []
    for x in xs:
        q1 = x[:rmodel.nq] # a2m removed, to be checked
        c = pinocchio.centerOfMass(rmodel, rdata, q1)
        Cx.append(c[0].item()) # np.asscalar deprecated
        Cy.append(c[1].item())
        Cz.append(c[2].item())


    #CoM
    plt.figure(figIndex+6, figsize=(16,9))

    plt.subplot(1, 3, 1)
    plt.title('CoM translation en X')
    plt.plot(Cy, label = 'simu')
    plt.xlabel('knots')
    plt.ylabel('y [m]')
    plt.legend()
    plt.grid(True)


    plt.subplot(1, 3, 2)
    plt.title('CoM translation en Y')
    plt.plot(Cx, label = 'simu')
    plt.xlabel('knots')
    plt.ylabel('x [m]')
    plt.legend()
    plt.grid(True)    


    plt.subplot(1, 3, 3)
    plt.title('CoM translation en z')
    plt.plot(Cz, label = 'simu')
    plt.xlabel('knots')
    plt.ylabel('z [m]')
    plt.legend()
    plt.grid(True)  
    
    if show:
        plt.show()



#********************************************************************************************************************************

# call the gepetto viewer server
gvs = subprocess.Popen(["./gepetto-viewer.sh","&"]) 
print('Loading the viewer ...')
time.sleep(1)


#********************************************************************************************************************************



#****************************************************************************
#****************************************************************************

WITHDISPLAY = 'display' in sys.argv or 'CROCODDYL_DISPLAY' in os.environ
WITHPLOT = 'plot' in sys.argv or 'CROCODDYL_PLOT' in os.environ
WITHLOG = 'log' in sys.argv
#****************************************************************************
#****************************************************************************

simName = 'results/withoutRAILPZ/backflip/'
if not os.path.exists(simName):
    os.makedirs(simName)
#****************************************************************************
#****************************************************************************



class TaskSE3:
    def __init__(self, oXf, frameId):
        self.oXf = oXf
        self.frameId = frameId


#********************************************************************************************************************************

# Loading the hoping leg model with a free-flayer

filename="/home/dfki.uni-bremen.de/mboukheddimi/DEVEL/DFKI/OC_UNDERACTUATED/hopping_leg/Controllers/OC_FDDP/without_rails/urdf/v6_new_joint_limits_pino.urdf"
meshfile="/home/dfki.uni-bremen.de/mboukheddimi/DEVEL/DFKI/OC_UNDERACTUATED/hopping_leg/Controllers/OC_FDDP/without_rails/"

#robot =  RobotWrapper.BuildFromURDF(filename, [meshfile], pinocchio.JointModelFreeFlyer())
robot =  RobotWrapper.BuildFromURDF(filename, [meshfile], pinocchio.JointModelPZ())
#robot =  RobotWrapper.BuildFromURDF(filename, [meshfile])
#********************************************************************************************************************************

#robot_model = robot.model

rmodel = robot.model

#********************************************************************************************************************************

#add frame on the ankle to add contact

'''
FrameTpl 	( 	const std::string &  	name,
		const JointIndex  	parent,
		const FrameIndex  	previousFrame,
		const SE3 &  	frame_placement,
		const FrameType  	type,
		const Inertia &  	inertia = Inertia::Zero() 
	) 	

'''

subtalar = np.asarray([0.20274169999999997, 0., 0.])  # -0.22  0.20000058  [z,    0.18000058, 0., 0  0.27           #0.22             0.20274169999999997
jointPlacement = pinocchio.SE3(np.eye(3), subtalar)


knee_JointId = rmodel.getJointId('urdf_knee')


knee_FrameId = rmodel.getFrameId('urdf_knee')


robot.model.addFrame( se3.Frame('subtalar',knee_JointId, knee_FrameId, jointPlacement, se3.FrameType.OP_FRAME) ) # the frame Id = 9

sub_FrameId = rmodel.getFrameId('subtalar')


hip_FrameId = rmodel.getFrameId('urdf_hip')


#********************************************************************************************************************************


rdata  = rmodel.createData()

robot.data = rdata


#********************************************************************************************************************************

state = crocoddyl.StateMultibody(rmodel)

#actuation = crocoddyl.ActuationModelFull(state) # the catuated link is the 2
actuation = crocoddyl.ActuationModelFloatingBase(state)

lims = rmodel.effortLimit
lims *= 0.9  # reduced artificially the torque limits 0.4 is working well without the modifications in the time horizon 
rmodel.effortLimit = lims
#setLimits(rmodel)

robot.initViewer(loadModel=True)



#********************************************************************************************************************************
v0 = np.zeros(rmodel.nv)

q0 = np.array([0.353, 0.5, -1.0])
qpush = np.array([0.162, 1.2, -2.35])
qf = np.array([0.36348193,  1.2, -2.6])

qff = np.array([0.36348193, -3.14, -0.6])

qf2 = np.array([0.36348193,  -5.4, -2.6])

rmodel.defaultState = np.concatenate([q0, v0]) 


#********************************************************************************************************************************

stateInit = np.concatenate([q0, v0])   #q1_2 or q0

statePush = np.concatenate([qpush, v0])

stateFly = np.concatenate([qf, v0])

stateFly2 = np.concatenate([qf2, v0])

stateUp = np.concatenate([qff, v0])

#********************************************************************************************************************************

contactModel = crocoddyl.ContactModelMultiple(state, actuation.nu)


contactModelVide = crocoddyl.ContactModelMultiple(state, actuation.nu)



#********************************************************************************************************************************


# Cost for self-collision
maxfloat = sys.float_info.max
xlb = np.concatenate([
    -maxfloat * np.ones(1),
    np.array([-3.14, -3.14]),  #+2
    -maxfloat * np.ones(1),
    np.array([-20 , -20])
])
xub = np.concatenate([
    maxfloat * np.ones(1),
    np.array([3.14, 3.14]),
    maxfloat * np.ones(1),
    np.array([20 , 20])
])
bounds = crocoddyl.ActivationBounds(xlb, xub, 1.)

limitCost = crocoddyl.CostModelState(state, crocoddyl.ActivationModelQuadraticBarrier(bounds), statePush,
                                     actuation.nu)


#********************************************************************************************************************************


# Cost for state and control
stateWeights = np.array([10] * 1 + [10] * (state.nv-1) + [1] * state.nv)
#for the terminal model
stateWeightsTerm = np.array([10] * 1 + [10.] * 2 + [100] * state.nv)

#****************************************************************************
#****************************************************************************

xRegCost1 = crocoddyl.CostModelState(state, crocoddyl.ActivationModelWeightedQuad(stateWeights**2), stateInit,
                                    actuation.nu) #state2


xRegCost2 = crocoddyl.CostModelState(state, crocoddyl.ActivationModelWeightedQuad(stateWeights**2), statePush,
                                    actuation.nu)

xRegCost3 = crocoddyl.CostModelState(state, crocoddyl.ActivationModelWeightedQuad(stateWeights**2), stateFly,
                                    actuation.nu)

xRegCostFly2 = crocoddyl.CostModelState(state, crocoddyl.ActivationModelWeightedQuad(stateWeights**2), stateFly2,
                                    actuation.nu)

xRegCost4 = crocoddyl.CostModelState(state, crocoddyl.ActivationModelWeightedQuad(stateWeights**2), stateUp,
                                    actuation.nu)


#****************************************************************************
#****************************************************************************

xRegTermCost = crocoddyl.CostModelState(state, crocoddyl.ActivationModelWeightedQuad(stateWeightsTerm**2),
                                        stateInit, actuation.nu)


#****************************************************************************
#****************************************************************************

uRegCost = crocoddyl.CostModelControl(state, actuation.nu)

#****************************************************************************
#****************************************************************************

#pinocchio.forwardKinematics(rmodel, rdata, qf)
#pinocchio.updateFramePlacements(rmodel, rdata)

#********************************************************************************************************************************


#rest position for q0

sub0 = np.array([ 0.09631652,  0.11425107, -0.00])    #sub0 = np.array([ 0.09631652,  0.11425107, -0.00043851])

knee0 = np.array([-0.00088368,  0.11425113,  0.17748372])

hip0 = np.array([0.09500006, 0.01524978, 0.35300022])



#********************************************************************************************************************************

#Flay phase final position

#sub = rdata.oMf[rmodel.getFrameId("subtalar")].translation
subFlay = np.array([0.10838511, 0.11425008, 0.25655158])

#knee = rdata.oMf[rmodel.getFrameId("urdf_knee")].translation
kneeFlay = np.array([-0.09140677,  0.11425142,  0.29101027])

#hip = rdata.oMf[rmodel.getFrameId("urdf_hip")].translation
hipFlay = np.array([0.09500006, 0.01524978, 0.36348215])


#********************************************************************************************************************************

#pushig position


subPush = np.array([0.09364885, 0.11425036, 0.00671159])

kneePush = np.array([-0.09140677,  0.11425142,  0.08952834])

hipPush = np.array([0.09500006, 0.01524978, 0.16200022])

#********************************************************************************************************************************
#swingUp max ellongation position

subUp = np.array([0.09564073, 0.1142483 , 0.7662237 ])


kneeUp = np.array([0.09531858, 0.11424905, 0.56348226])


hipUp = np.array([0.09500006, 0.01524978, 0.36348215])

#********************************************************************************************************************************
#********************************************************************************************************************************

com0ref = pinocchio.centerOfMass(rmodel, rdata, qpush)
comTrack1 = crocoddyl.CostModelCoMPosition(state, com0ref , actuation.nu)

comF = pinocchio.centerOfMass(rmodel, rdata, qf)
comTrack2 = crocoddyl.CostModelCoMPosition(state, comF , actuation.nu)

comFF =pinocchio.centerOfMass(rmodel, rdata, qff)
comTrack3 = crocoddyl.CostModelCoMPosition(state, comFF , actuation.nu)

#********************************************************************************************************************************
#********************************************************************************************************************************


timeStep = 0.005
DT = 0.005


groundKnots = 100 #100



recoveryKnots = 100


recoveryKnots2 = 100

#jumpHeight = 0.1

#jumpHeight = 0.7662237
jumpHeight = 1.5

recoveryKnotsUP = 1

UpHeight = 1.5

#UpHeight = 0.7662237

flyingKnots = round(2*math.sqrt(2*jumpHeight/9.81)/timeStep)
print(flyingKnots)


numKnots2 = round(2*math.sqrt(2*UpHeight/9.81)/timeStep)
print ("numKnots2", numKnots2)

#flyingKnots = 57


numKnots = flyingKnots

#********************************************************************************************************************************
'''
#tref = []
for k in range(numKnots):
    Foot1Task = []
    Foot2Task = []


    phKnots = (numKnots / 2)              #-0.5
    if k < phKnots:
        dp = np.array([0.09, 0.11, jumpHeight * k / phKnots])
    elif k == phKnots:
        dp = np.array([0.09,0.11, jumpHeight])
    else:
        dp = np.array([0.09, 0.11, jumpHeight * (1 - float(k - phKnots) / phKnots)])
    #tref += [subPush +dp]

    tref = subPush + dp
    trefHip = hipPush + dp

    Foot1Task += [TaskSE3(pinocchio.SE3(np.eye(3), tref), sub_FrameId)]
    Foot2Task += [TaskSE3(pinocchio.SE3(np.eye(3), trefHip), hip_FrameId)]

'''
jumpK = round(numKnots2/2) +1

landK = round(numKnots2/2)

flipK1 = round(landK/2)

flipK2 = round(landK/2)

#********************************************************************************************************************************

for k in range(jumpK+1):
    Foot1TaskUP = []
    #Foot1Task = []
    Foot2Task = []


    phKnots = jumpK              #-0.5
    if k < jumpK:
        dp = np.array([0.09, 0.11, jumpHeight * k / phKnots])
    elif k == jumpK:
        dp = np.array([0.09,0.11, jumpHeight])
    #tref += [subPush +dp]

    tref = subPush + dp
    trefHip = hipPush + dp

    Foot1TaskUP += [TaskSE3(pinocchio.SE3(np.eye(3), tref), sub_FrameId)]

    Foot2Task += [TaskSE3(pinocchio.SE3(np.eye(3), trefHip), hip_FrameId)]

#********************************************************************************************************************************#
#********************************************************************************************************************************

for k in range(10):
    Foot1Task = []

    phKnots = landK 
    if k < landK:
        dp = np.array([0.09, 0.11, jumpHeight * (-float(k - landK) / landK)])
    elif k == landK:
        dp = np.array([0.09,0.11, 0.])
    #tref += [subPush +dp]

    tref = subPush + dp


    Foot1Task += [TaskSE3(pinocchio.SE3(np.eye(3), tref), sub_FrameId)]


#********************************************************************************************************************************


framePlacement = crocoddyl.FrameTranslation(sub_FrameId, sub0)

supportContactModel = crocoddyl.ContactModel2D(state, framePlacement, actuation.nu, np.array([10, 10]))        #0. 50 or 0. 100

contactModel.addContact("sub_Frame_contact", supportContactModel)

#********************************************************************************************************************************

lowerLimit1=np.array([-1e15, -1e15, 0.2 , -1e15, -1e15, -1e15])    #0.3
lowerLimit2=np.array([-1e15, -1e15, 0., -1e15, -1e15, -1e15])   #0.05
lowerLimit3=np.array([-1e15, -1e15, 0., -1e15, -1e15, -1e15])   #0.08

upperLimit=np.array([1e15, 1e15, 1e15, 1e15, 1e15, 1e15])


act_ineq_swing1 = crocoddyl.ActivationModelQuadraticBarrier(crocoddyl.ActivationBounds(lowerLimit1, upperLimit, 1.)) # check role of beta = 1.
act_ineq_swing2 = crocoddyl.ActivationModelQuadraticBarrier(crocoddyl.ActivationBounds(lowerLimit2, upperLimit, 1.)) # check role of beta = 1.
act_ineq_swing3 = crocoddyl.ActivationModelQuadraticBarrier(crocoddyl.ActivationBounds(lowerLimit3, upperLimit, 1.)) # check role of beta = 1.

Foot1Task3 = TaskSE3(pinocchio.SE3(np.eye(3), sub0), sub_FrameId)

Pref3 = crocoddyl.FramePlacement(Foot1Task3.frameId, Foot1Task3.oXf)


cost_swing3 = crocoddyl.CostModelFramePlacement(state, Mref=Pref3, nu=actuation.nu, activation=act_ineq_swing2)


cost_swing4 = crocoddyl.CostModelFramePlacement(state, Mref=Pref3, nu=actuation.nu, activation=act_ineq_swing3)

for i in Foot1Task:  # the ankel

    Pref = crocoddyl.FramePlacement(i.frameId, i.oXf) 

    footTrack = crocoddyl.CostModelFramePlacement(state, Pref, actuation.nu)
            
    cost_swing = crocoddyl.CostModelFramePlacement(state, Mref=Pref, nu=actuation.nu, activation=act_ineq_swing1)


for i in Foot1TaskUP:  # the ankel

    PrefUP = crocoddyl.FramePlacement(i.frameId, i.oXf) 

    footTrackUP = crocoddyl.CostModelFramePlacement(state, PrefUP, actuation.nu)
            
    cost_swingUP = crocoddyl.CostModelFramePlacement(state, Mref=PrefUP, nu=actuation.nu, activation=act_ineq_swing1)


for i in Foot2Task:  # the knee

    Pref = crocoddyl.FramePlacement(i.frameId, i.oXf) 

    footTrack2 = crocoddyl.CostModelFramePlacement(state, Pref, actuation.nu)
            
    #cost_swing2 = crocoddyl.CostModelFramePlacement(state, Mref=Pref, nu=actuation.nu, activation=act_ineq_swing)

#********************************************************************************************************************************


# Create cost model per each action model. We divide the motion in 3 phases plus its terminal model
runningCostModel1 = crocoddyl.CostModelSum(state, actuation.nu)

runningCostModel2 = crocoddyl.CostModelSum(state, actuation.nu)

runningCostModel2UP = crocoddyl.CostModelSum(state, actuation.nu)

runningCostModel2UPR = crocoddyl.CostModelSum(state, actuation.nu)

runningCostModel3 = crocoddyl.CostModelSum(state, actuation.nu)



#****************************************************************************
 
terminalCostModel = crocoddyl.CostModelSum(state, actuation.nu)

#****************************************************************************
#****************************************************************************

# Then let's added the running and terminal cost functions


runningCostModel1.addCost("stateReg", xRegCost2, 1e6) #  #e6
runningCostModel1.addCost("ctrlReg", uRegCost, 0.002) #1e-4   0.002


runningCostModel1.addCost("limitCost", limitCost, 1e3) #1e3
#runningCostModel1.addCost("swing_orientation1", cost_swing3, 1e10)  #e7



#****************************************************************************
#****************************************************************************

runningCostModel2UP.addCost("footTrack1", footTrackUP, 1e13) #13

#runningCostModel2UP.addCost("swing_orientation1", cost_swingUP, 1e10)

runningCostModel2UP.addCost("stateReg2", xRegCost4, 1e-2) #1e-2
runningCostModel2UP.addCost("ctrlReg", uRegCost, 0.002) #1e-4
runningCostModel2UP.addCost("limitCost", limitCost, 1e7) #1e3

#****************************************************************************
#runningCostModel2UPR.addCost("swing_orientation1", cost_swingUP, 1e11)


runningCostModel2UPR.addCost("stateReg2", xRegCost4, 1e3) #1e3
runningCostModel2UPR.addCost("ctrlReg", uRegCost, 0.002) #1e-4
runningCostModel2UPR.addCost("limitCost", limitCost, 1e7) #1e3

#****************************************************************************

runningCostModel2.addCost("footTrack1", footTrack, 1e13) #13

#runningCostModel2.addCost("swing_orientation1", cost_swing3, 1e13)

runningCostModel2.addCost("stateReg2", xRegCostFly2 , 1e-4) #1e-3  xRegCost3
runningCostModel2.addCost("ctrlReg", uRegCost, 0.002) #1e-4
runningCostModel2.addCost("limitCost", limitCost, 1e7) #1e3

#****************************************************************************

runningCostModel3.addCost("stateReg3", xRegCost1, 1e5) #1e-3
runningCostModel3.addCost("ctrlReg", uRegCost, 0.002) #1e-4
runningCostModel3.addCost("limitCost", limitCost, 1e7) #1e3

#runningCostModel3.addCost("swing_orientation1", cost_swing3, 1e11)

#****************************************************************************
#****************************************************************************
#terminalCostModel.addCost("swing_orientation1", cost_swing3, 1e11)
terminalCostModel.addCost("stateReg",xRegTermCost, 1e9) #1e-5  # xRegTermCost
terminalCostModel.addCost("limitCost", limitCost, 1e7) #1e7

#****************************************************************************
#****************************************************************************



# Create the action model
dmodelRunning1 = crocoddyl.DifferentialActionModelContactFwdDynamics(state, actuation, contactModel,
                                                                     runningCostModel1, 1e-5)


dmodelRunning2UP = crocoddyl.DifferentialActionModelContactFwdDynamics(state, actuation, contactModelVide,
                                                                     runningCostModel2UP, 1e-5)

dmodelRunning2UPR = crocoddyl.DifferentialActionModelContactFwdDynamics(state, actuation, contactModelVide,
                                                                     runningCostModel2UPR, 1e-5)

dmodelRunning2 = crocoddyl.DifferentialActionModelContactFwdDynamics(state, actuation, contactModelVide,
                                                                     runningCostModel2, 1e-5)


dmodelRunning3 = crocoddyl.DifferentialActionModelContactFwdDynamics(state, actuation, contactModel,
                                                                     runningCostModel3, 1e-5)


#****************************************************************************

dmodelTerminal = crocoddyl.DifferentialActionModelContactFwdDynamics(state, actuation, contactModel,
                                                                     terminalCostModel, 1e-5)


#****************************************************************************
#****************************************************************************

runningModel1 = crocoddyl.IntegratedActionModelEuler(dmodelRunning1, DT)
runningModel2UP = crocoddyl.IntegratedActionModelEuler(dmodelRunning2UP, DT)
runningModel2UPR = crocoddyl.IntegratedActionModelEuler(dmodelRunning2UPR, DT)
runningModel2 = crocoddyl.IntegratedActionModelEuler(dmodelRunning2, DT)
runningModel3 = crocoddyl.IntegratedActionModelEuler(dmodelRunning3, DT)


#****************************************************************************

terminalModel = crocoddyl.IntegratedActionModelEuler(dmodelTerminal, 0)

#****************************************************************************
#****************************************************************************

# Problem definition

x0 = statePush  #q1_2 or q0
#+ [runningModel3] * recoveryKnots

problem = crocoddyl.ShootingProblem(x0, [runningModel1] * groundKnots + [runningModel2UP] * jumpK +  [runningModel2UPR] * recoveryKnotsUP + [runningModel2] * landK + [runningModel3] * recoveryKnots, terminalModel)

#problem = crocoddyl.ShootingProblem(x0, [runningModel1] * groundKnots + [runningModel2] * flyingKnots + [runningModel3] * recoveryKnots + [runningModel1] * groundKnots + [runningModel2] * flyingKnots #+ [runningModel3] * recoveryKnots + [runningModel1] * groundKnots + [runningModel2UP] * numKnots2 + [runningModel3] * recoveryKnots, terminalModel)
#****************************************************************************
#****************************************************************************

# Creating the DDP solver for this OC problem, defining a logger
ddp = crocoddyl.SolverBoxFDDP(problem)
#ddp = crocoddyl.SolverFDDP(problem)

ddp.setCallbacks([crocoddyl.CallbackVerbose()])

#****************************************************************************
#****************************************************************************
v00 = np.reshape(v0, [1,3])
vNew = np.repeat(v00, (ddp.problem.T + 1), axis=0)

xssN = np.load("xss.npy")
xssNE = xssN[:, :3]

xs = list(np.concatenate((xssNE, vNew), axis=1))

# Solving it with the DDP algorithm
#xs = [rmodel.defaultState] * (ddp.problem.T + 1)
#xs = list(np.concatenate([xssNE, vNew]))

us = ddp.problem.quasiStatic([rmodel.defaultState] * ddp.problem.T)

#ddp.th_stop = 1e-7

ddp.solve(xs, us, 100, False, 1e-4)

#ddp.solve()

#****************************************************************************
#****************************************************************************


qddp = [x[:rmodel.nq] for x in ddp.xs]

a = np.save('/home/dfki.uni-bremen.de/mboukheddimi/DEVEL/DFKI/OC_UNDERACTUATED/hopping_leg/Controllers/OC_FDDP/results/withoutRAILPZ/backflip/q_hopLegbackflip.npy',qddp)
qqq = np.asmatrix(np.load('/home/dfki.uni-bremen.de/mboukheddimi/DEVEL/DFKI/OC_UNDERACTUATED/hopping_leg/Controllers/OC_FDDP/results/withoutRAILPZ/backflip/q_hopLegbackflip.npy'))

#****************************************************************************
#****************************************************************************

# Get final state and end effector position
xT = ddp.xs[-1]

#****************************************************************************
#****************************************************************************

def playMotions(first=0, last=1, step=1, t=0):
    for i in range(first, last, step):

        robot.display(qqq[i].T)
        time.sleep(0.01)




plotSolution(rmodel, rdata , ddp.xs, ddp.us)

#****************************************************************************
#****************************************************************************

time.sleep(2)
playMotions(0,615,1,0.)


ddpL = [ddp]

if WITHLOG:
    logPath = simName + '/logs/'
    if not os.path.exists(logPath):
        os.makedirs(logPath)
    logSolution(ddpL, DT,logPath)



