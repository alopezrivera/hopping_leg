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

class TaskSE3:
    def __init__(self, oXf, frameId):
        self.oXf = oXf
        self.frameId = frameId


#********************************************************************************************************************************

# call the gepetto viewer server
gvs = subprocess.Popen(["./gepetto-viewer.sh","&"]) 
print('Loading the viewer ...')
time.sleep(1)


#********************************************************************************************************************************

# Loading the hoping leg model with a free-flayer

filename="/home/dfki.uni-bremen.de/mboukheddimi/DEVEL/DFKI/OC_UNDERACTUATED/hopping_leg/controllers/OC_FDDP/without_rails/urdf/v6_new_joint_limits_pino.urdf"
meshfile="/home/dfki.uni-bremen.de/mboukheddimi/DEVEL/DFKI/OC_UNDERACTUATED/hopping_leg/controllers/OC_FDDP/without_rails"

robot =  RobotWrapper.BuildFromURDF(filename, [meshfile], pinocchio.JointModelFreeFlyer())
#robot =  RobotWrapper.BuildFromURDF(filename, [meshfile])

#********************************************************************************************************************************

robot_model = robot.model

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

subtalar = np.asarray([0.18000058, 0., 0])  # -0.22  0.20000058  [z,    0.18000058, 0., 0.
jointPlacement = pinocchio.SE3(np.eye(3), subtalar)


knee_JointId = rmodel.getJointId('knee_joint')
knee_FrameId = rmodel.getFrameId('knee_joint')


robot.model.addFrame( se3.Frame('subtalar',knee_JointId, knee_FrameId, jointPlacement, se3.FrameType.OP_FRAME) ) # the frame Id = 9

sub_FrameId = rmodel.getFrameId('subtalar')

#********************************************************************************************************************************


rdata  = robot_model.createData()
robot.data = rdata

lims = rmodel.effortLimit

lims = rmodel.velocityLimit


#********************************************************************************************************************************

state = crocoddyl.StateMultibody(robot_model)

actModel = crocoddyl.ActuationModelFloatingBase(state) # the catuated link is the 2

#********************************************************************************************************************************
v0 = np.zeros(rmodel.nv)

#q0 = pinocchio.neutral(rmodel)

#q0 = np.asarray([0, 0, 0.4, 0, 0, 0, 1, 0, 0])
q1 = np.asarray([0, 0, 0.4, 0, 0, 0, 1, 0., 0.])
#q1 = np.asarray([0., 0.])
rmodel.defaultState = np.concatenate([q1, v0])

x1 = np.concatenate([q1, v0])

#****************************************************************************
#****************************************************************************

pinocchio.forwardKinematics(rmodel, rdata, q1)
pinocchio.updateFramePlacements(rmodel, rdata)

com0 = pinocchio.centerOfMass(rmodel, rdata, q1)

#****************************************************************************
#****************************************************************************
#robot.initViewer(loadModel=True)
'''
robot.initViewer(loadModel=True)

# Add the floor and scale it
robot.viewer.gui.addFloor('hpp-gui/floor')
robot.viewer.gui.setScale('hpp-gui/floor', [0.5, 0.5, 0.5])
robot.viewer.gui.setColor('hpp-gui/floor', [0.7, 0.7, 0.7, 1.])
robot.viewer.gui.setLightingMode('hpp-gui/floor', 'OFF')

#robot.display(q1)
'''
#robot.display(q1)
#****************************************************************************
#****************************************************************************


# Set integration time
dt = 1e-2 #5e-2 
T = 25    #60


#****************************************************************************
#****************************************************************************

#parametters of the filter
b, a = butter(3, 0.05)

#parametres of the cone frictions
mu = 0.7  
Rsurf = np.eye(3)

#****************************************************************************
#****************************************************************************

# Creating a 3D multi-contact model, and then including the supporting foot

contactModel1 = crocoddyl.ContactModelMultiple(state, actModel.nu)

contactModel2 = crocoddyl.ContactModelMultiple(state, actModel.nu)

#****************************************************************************
#****************************************************************************

# Creating the cost model for a contact phase

costModel1 = crocoddyl.CostModelSum(state, actModel.nu)

costModel2 = crocoddyl.CostModelSum(state, actModel.nu)

#****************************************************************************
#****************************************************************************
#ref = np.array([0., 0., 0.])

pinocchio.framesForwardKinematics(robot.model, robot.data, q1)
frame_pos = robot.data.oMf[sub_FrameId].translation
xref = crocoddyl.FrameTranslation(sub_FrameId, frame_pos)

#xref = crocoddyl.FrameTranslation(sub_FrameId, np.array([0. , 0.,  0.]))  #  0.0949999 , -0.10574948,  0.19999977

supportContactModel = crocoddyl.ContactModel2D(state, xref, actModel.nu, np.array([1., 1.]))

#xref = crocoddyl.FramePlacement(sub_FrameId, pinocchio.SE3(np.eye(3),ref))
#supportContactModel = crocoddyl.ContactModel6D(state, xref, actModel.nu, np.array([0., 0.]))

#crocoddyl.FrictionCone(np.array([0.,0.,1.]), 0.2)
cone = crocoddyl.FrictionCone(Rsurf, mu, 4, False)
frictionCone = crocoddyl.CostModelContactFrictionCone(state, crocoddyl.ActivationModelQuadraticBarrier(crocoddyl.ActivationBounds(cone.lb, cone.ub)),
                                                      crocoddyl.FrameFrictionCone(sub_FrameId, cone), actModel.nu)




#****************************************************************************

#comTask = pinocchio.centerOfMass(rmodel, rdata, q1)

#comTask = np.asarray([ 0.08023916, 0.03546606, 0.35441062]) # for qo it is the reference

comTask = np.asarray([ 0.08023916, 0.03546606, 0.25441062])

comTrack = crocoddyl.CostModelCoMPosition(state, comTask, actModel.nu)



#****************************************************************************

#trajFoot1 = rdata.oMf[rmodel.getFrameId("subtalar")].translation
#trajFoot1 = array([ 0.0949999 , -0.10574948,  0.19999977])  #traj ref on the ground

'''
trajFoot1 = np.array([ 0.0949999 , -0.10574948,  0.09999977]) 


swingFootTask = []
#swingFootTask += [TaskSE3(pinocchio.SE3(np.eye(3), trajFoot1), sub_FrameId)]
swingFootTask = [TaskSE3(pinocchio.SE3(np.eye(3), trajFoot1), sub_FrameId)]

for i in swingFootTask:
    xref = crocoddyl.FrameTranslation(i.frameId, i.oXf.translation)
    footTrack = crocoddyl.CostModelFrameTranslation(state, xref, actModel.nu)

    costModel2.addCost(rmodel.frames[i.frameId].name + "_footTrack", footTrack, 1e6)

'''

#****************************************************************************

#stateWeights = np.array([0.] * 3 + [500.] * 3 + [0.01] * (rmodel.nv - 6) + [10.] * 6 + [1.] *
#                                (rmodel.nv - 6))

stateWeights = np.array([0.] * 3 + [0.] * 3 + [0.01] * (rmodel.nv - 6) + [1.] *(rmodel.nv))
#stateWeights = np.array([0.01] * (rmodel.nv) + [1.] *(rmodel.nv))

stateReg = crocoddyl.CostModelState(state, crocoddyl.ActivationModelWeightedQuad(stateWeights**2),
                                    x1, actModel.nu)


#****************************************************************************

ctrlReg = crocoddyl.CostModelControl(state, actModel.nu)



#****************************************************************************
#****************************************************************************

lb = np.concatenate([state.lb[1:state.nv + 1], state.lb[-state.nv:]])
ub = np.concatenate([state.ub[1:state.nv + 1], state.ub[-state.nv:]])
stateBounds = crocoddyl.CostModelState(
    state, crocoddyl.ActivationModelQuadraticBarrier(crocoddyl.ActivationBounds(lb, ub)),
    0 * rmodel.defaultState, actModel.nu)


#****************************************************************************
#****************************************************************************

'''
# Extract Time
time_traj = np.linspace(0.0000, 6.0000, 3000)
time_traj = time_traj.reshape(3000, 1).T
print("Time Array Shape: {}".format(time_traj.shape))
'''

#****************************************************************************
#****************************************************************************
#cost and contact model 1

#contact
contactModel1.addContact("3DContact1", supportContactModel)

#costs

#costModel1.addCost("frictionCone1", frictionCone, 1e1)
costModel1.addCost("comTrack1", comTrack, 1e1)
costModel1.addCost("stateReg1", stateReg, 1e-1)
#costModel1.addCost("stateBounds1", stateBounds, 1e1)



costModel1.addCost("ctrlReg", ctrlReg, 1e-5)

#****************************************************************************
#****************************************************************************
#cost and contact model 2

#contact
contactModel2.addContact("3DContact2", supportContactModel)

#costs

#costModel2.addCost("frictionCone2", frictionCone, 1e1)
costModel2.addCost("comTrack2", comTrack, 1e1)
costModel2.addCost("stateReg2", stateReg, 1)
#costModel2.addCost("stateBounds2", stateBounds, 1e1)

#****************************************************************************
#****************************************************************************
#models
#0., True
runningModel = crocoddyl.IntegratedActionModelEuler(
    crocoddyl.DifferentialActionModelContactFwdDynamics(state, actModel, contactModel1, costModel1), dt)  #1e-5, True

terminalModel = crocoddyl.IntegratedActionModelEuler(
    crocoddyl.DifferentialActionModelContactFwdDynamics(state, actModel, contactModel2, costModel2), dt)  #1e-5, True

#****************************************************************************
#****************************************************************************

x1 = np.concatenate([q1, v0])


problem = crocoddyl.ShootingProblem(x1, [runningModel] * T, terminalModel)

# Creating the DDP solver for this OC problem, defining a logger
ddp = crocoddyl.SolverBoxFDDP(problem)


ddp.setCallbacks([crocoddyl.CallbackVerbose()])

# Solving it with the DDP algorithm
xs = [rmodel.defaultState] * (ddp.problem.T + 1)
us = ddp.problem.quasiStatic([rmodel.defaultState] * ddp.problem.T)

ddp.th_stop = 1e-7

#input('weight')

ddp.solve(xs, us, 500, False, 1e-7)

#input('weight')

xT = ddp.xs[-1]



#****************************************************************************
#****************************************************************************


cameraTF = [1.4, 0., 0.2, 0.5, 0.5, 0.5, 0.5]

display = crocoddyl.GepettoDisplay(robot, 4, 4, cameraTF, True)
ddp.setCallbacks([crocoddyl.CallbackLogger(), crocoddyl.CallbackVerbose(), crocoddyl.CallbackDisplay(display)])

startdtTest = time.time()


#****************************************************************************
#****************************************************************************






#****************************************************************************
#****************************************************************************

enddtTest = time.time()

dt = (enddtTest - startdtTest) / 1000
cmd_freq = 1 / dt
print("Dt = {}".format(dt))
print("Command Frequency: {} Hz".format(cmd_freq))

log = ddp.getCallbacks()[0]


#*********************************************************************************************************************************************************8
#***************************************************************************************************


# Plotting the entire motion

#crocoddyl.plotOCSolution(log.xs, log.us, figIndex=2, show=True)
#crocoddyl.plotConvergence(log.costs, log.u_regs, log.x_regs, log.grads, log.stops, log.steps, figIndex=2)

# Display the entire motion

display = crocoddyl.GepettoDisplay(robot, floor=False)
display.displayFromSolver(ddp)


