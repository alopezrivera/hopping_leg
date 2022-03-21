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

filename="/home/dfki.uni-bremen.de/mboukheddimi/DEVEL/DFKI/OC_UNDERACTUATED/hopping_leg/controllers/OC_FDDP/with_rails/urdf/v7_pino.urdf"
meshfile="/home/dfki.uni-bremen.de/mboukheddimi/DEVEL/DFKI/OC_UNDERACTUATED/hopping_leg/controllers/OC_FDDP/with_rails"


robot =  RobotWrapper.BuildFromURDF(filename, [meshfile])
#********************************************************************************************************************************

robot_model = robot.model

rmodel = robot.model

#********************************************************************************************************************************

#add frame on the ankle to add contact

subtalar = np.asarray([0.27, 0., 0.]) 
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

actModel = crocoddyl.ActuationModelFloatingBase(state)


#********************************************************************************************************************************
v0 = np.zeros(rmodel.nv)


#without free flyer
q1 = np.array([0.4, 0.5, -1.0])
qpush = np.array([0.185, 1.2, -2.35])
qf = np.array([0.40348193,  1.2, -2.6])

rmodel.defaultState = np.concatenate([q1, v0])

x1 = np.concatenate([q1, v0])
xpush = np.concatenate([qpush, v0])
xf = np.concatenate([qf, v0])


#****************************************************************************
#****************************************************************************
#trajFoot1 = rdata.oMf[rmodel.getFrameId("subtalar")].translation
#for q1
pinocchio.forwardKinematics(rmodel, rdata, q1)
pinocchio.updateFramePlacements(rmodel, rdata)

comTask1 = pinocchio.centerOfMass(rmodel, rdata, q1)

frame_posq1 = robot.data.oMf[sub_FrameId].translation

#for qpush
pinocchio.forwardKinematics(rmodel, rdata, q1)
pinocchio.updateFramePlacements(rmodel, rdata)


#for qf

pinocchio.forwardKinematics(rmodel, rdata, q1)
pinocchio.updateFramePlacements(rmodel, rdata)




#****************************************************************************
#****************************************************************************

# Set integration time
dt = 1e-2 #5e-2 
T = 25    #60



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

costModel3 = crocoddyl.CostModelSum(state, actModel.nu)

costModel4 = crocoddyl.CostModelSum(state, actModel.nu)

costModel5 = crocoddyl.CostModelSum(state, actModel.nu)

#****************************************************************************
#****************************************************************************
#ref = np.array([0., 0., 0.])

#creat 2D contact

FootRefq1 = crocoddyl.FrameTranslation(sub_FrameId, frame_posq1)

supportContactModel = crocoddyl.ContactModel2D(state, FootRefq1, actModel.nu, np.array([0., 50.]))


#****************************************************************************
#frame translation

PrefFoot = crocoddyl.FrameTranslation(sub_FrameId, trajFoot1) 

footTrack = crocoddyl.CostModelFrameTranslation(state, PrefFoot, actModel.nu)

#****************************************************************************
#****************************************************************************

#state cost

stateWeights = np.array([10]*1 + [0.01] * (rmodel.nv-1) + [10.] *(rmodel.nv))


stateReg1 = crocoddyl.CostModelState(state, crocoddyl.ActivationModelWeightedQuad(stateWeights**2),
                                    x1, actModel.nu)

stateReg2 = crocoddyl.CostModelState(state, crocoddyl.ActivationModelWeightedQuad(stateWeights**2),
                                    xpush, actModel.nu)

stateReg3 = crocoddyl.CostModelState(state, crocoddyl.ActivationModelWeightedQuad(stateWeights**2),
                                    xf, actModel.nu)

#****************************************************************************

ctrlReg = crocoddyl.CostModelControl(state, actModel.nu)



#****************************************************************************
#****************************************************************************
#contact

contactModel1.addContact("2DContact1", supportContactModel)

#****************************************************************************
#****************************************************************************

#costs

costModel1.addCost("stateReg1", stateReg1, 1e5)
costModel1.addCost("ctrlReg", ctrlReg, 1e-5)

#****************************************************************************
#****************************************************************************
#cost and contact model 2

 
costModel2.addCost("stateReg2", stateReg2, 1e5)
costModel2.addCost("ctrlReg", ctrlReg, 1e-5)

#****************************************************************************
#****************************************************************************
 #cost and contact model 3


costModel3.addCost("stateReg3", stateReg3, 1e5)
costModel3.addCost("ctrlReg", ctrlReg, 1e-5)


#****************************************************************************
#****************************************************************************
 #cost and contact model 4

costModel4.addCost("stateReg4", stateReg1, 1e5)
costModel4.addCost("ctrlReg", ctrlReg, 1e-5)


#****************************************************************************
#****************************************************************************
 #cost and contact model 5
costModel5.addCost("stateReg4", stateReg1, 1e5)      
costModel5.addCost("footTrack", footTrack, 1e5)



#****************************************************************************
#****************************************************************************

#models
#0., True
runningModel1 = crocoddyl.IntegratedActionModelEuler(
    crocoddyl.DifferentialActionModelContactFwdDynamics(state, actModel, contactModel1, costModel1), dt)  #1e-5, True

runningModel2 = crocoddyl.IntegratedActionModelEuler(
    crocoddyl.DifferentialActionModelContactFwdDynamics(state, actModel, contactModel1, costModel2), dt)  #1e-5, True

runningModel3 = crocoddyl.IntegratedActionModelEuler(
    crocoddyl.DifferentialActionModelContactFwdDynamics(state, actModel, contactModel2, costModel3), dt)  #1e-5, True

runningModel4 = crocoddyl.IntegratedActionModelEuler(
    crocoddyl.DifferentialActionModelContactFwdDynamics(state, actModel, contactModel2, costModel4), dt)  #1e-5, True

terminalModel = crocoddyl.IntegratedActionModelEuler(
    crocoddyl.DifferentialActionModelContactFwdDynamics(state, actModel, contactModel1, costModel5), dt)  #1e-5, True

#****************************************************************************
#****************************************************************************

x1 = np.concatenate([q1, v0])


problem = crocoddyl.ShootingProblem(x1, [runningModel1] * T + [runningModel2] * T + [runningModel3] * T + [runningModel4] * T, terminalModel)


# Creating the DDP solver for this OC problem, defining a logger
ddp = crocoddyl.SolverFDDP(problem)


ddp.setCallbacks([crocoddyl.CallbackVerbose()])

# Solving it with the DDP algorithm
xs = [rmodel.defaultState] * (ddp.problem.T + 1)
us = ddp.problem.quasiStatic([rmodel.defaultState] * ddp.problem.T)

ddp.th_stop = 1e-7

#input('weight')
startdtTest = time.time()

ddp.solve(xs, us, 500, False, 1e-7)
xT = ddp.xs[-1]

enddtTest = time.time()
#input('weight')


#****************************************************************************
#****************************************************************************


cameraTF = [1.4, 0., 0.2, 0.5, 0.5, 0.5, 0.5]

display = crocoddyl.GepettoDisplay(robot, cameraTF, True)
ddp.setCallbacks([crocoddyl.CallbackLogger(), crocoddyl.CallbackVerbose(), crocoddyl.CallbackDisplay(display)])


dt = (enddtTest - startdtTest) / 1000
cmd_freq = 1 / dt
print("Dt = {}".format(dt))
print("Command Frequency: {} Hz".format(cmd_freq))

log = ddp.getCallbacks()[0]


#*********************************************************************************************************************************************************
#*********************************************************************************************************************************************************

qddp = [x[:rmodel.nq] for x in ddp.xs]
a = np.save('q_hop.npy',qddp)
qqq = np.asmatrix(np.load('q_hop.npy'))

#*********************************************************************************************************************************************************
#*********************************************************************************************************************************************************


# Display the entire motion

display = crocoddyl.GepettoDisplay(robot, floor=False)
display.displayFromSolver(ddp)



