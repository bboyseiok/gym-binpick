import gym
from gym import error, spaces, utils
from gym.utils import seeding

import os
import pybullet as p
import pybullet_data
import math
import numpy as np
import random

import pybullet_object_models
from pybullet_object_models import ycb_objects


MAX_EPISODE_LEN = 20*100

class BinPickEnv3(gym.Env):
    metadata = {'render.modes': ['human']}

    def __init__(self):
        self.step_counter = 0
        p.connect(p.GUI)
        p.resetDebugVisualizerCamera(cameraDistance=1.5, cameraYaw=0, cameraPitch=0, cameraTargetPosition=[1,0,0.7])
        self.action_space = spaces.Box(np.array([-1]*4), np.array([1]*4))
        self.observation_space = spaces.Box(np.array([-1]*5), np.array([1]*5))

    def step(self, action):
        p.configureDebugVisualizer(p.COV_ENABLE_SINGLE_STEP_RENDERING)
        orientation = p.getQuaternionFromEuler([0.,-math.pi/2,math.pi])
        #         orientation = p.getQuaternionFromEuler([0.,-math.pi,math.pi/2.])
        dv = 0.005
        dx = action[0] 
        dy = action[1]
        dz = action[2]
        fingers = action[3]

        currentPose = p.getLinkState(self.pandaUid, 11)
        currentPosition = currentPose[0]
        newPosition = [currentPosition[0] + dx,
                       currentPosition[1] + dy,
                       currentPosition[2] + dz]
        jointPoses = p.calculateInverseKinematics(self.pandaUid,11,newPosition, orientation)[0:7]

        # p.setJointMotorControlArray(self.pandaUid, list(range(7))+[9,10], p.POSITION_CONTROL, list(jointPoses)+2*[fingers])
        for joint, value in zip(list(range(7))+[9,10], list(jointPoses)+2*[fingers]):
            p.resetJointState(self.pandaUid, joint, value)

        p.stepSimulation()

        state_object, _ = p.getBasePositionAndOrientation(self.objectUid)
        state_robot = p.getLinkState(self.pandaUid, 11)[0]
        state_fingers = (p.getJointState(self.pandaUid,9)[0], p.getJointState(self.pandaUid, 10)[0])



        if state_object[2]>0.45:
            reward = 1
            done = True
        else:
            reward = 0
            done = False

        self.step_counter += 1

        if self.step_counter > MAX_EPISODE_LEN:
            reward = 0
            done = True

        info = {'object_position': state_object}
        self.observation = state_robot + state_fingers
        return np.array(self.observation).astype(np.float32), reward, done, info

    def reset(self):
        self.step_counter = 0
        p.resetSimulation()
        p.configureDebugVisualizer(p.COV_ENABLE_RENDERING,0) # enable rendering after loading everything
        urdfRootPath=pybullet_data.getDataPath()
        p.setGravity(0,0,-10)

        planeUid = p.loadURDF(os.path.join(urdfRootPath,"plane.urdf"), basePosition=[0,0,0])

        rest_poses = [0,-0.6, 0.0,-2.57,0,3.5,0.75, 0.08,0.08]
        self.pandaUid = p.loadURDF(os.path.join(urdfRootPath, "franka_panda/panda.urdf"),useFixedBase=True)
        for i in range(7):
            p.resetJointState(self.pandaUid,i, rest_poses[i])
        p.resetJointState(self.pandaUid, 9, 0.08)
        p.resetJointState(self.pandaUid,10, 0.08)
        tableUid = p.loadURDF(os.path.join(os.path.dirname(os.path.realpath(__file__)),"cabinet.urdf"),basePosition=[0.70 ,0, 0], useFixedBase=True)    

        state_object= [0.7, -0.05,0.53]
        state_object_5= [0.58,-0.05,0.63]
        self.objectUid = p.loadURDF(os.path.join(ycb_objects.getDataPath(), 'YcbPowerDrill', "model.urdf"), basePosition=state_object, baseOrientation=p.getQuaternionFromEuler([math.pi/2, math.pi/6, math.pi]))

        self.objectUid_5 = p.loadURDF(os.path.join(ycb_objects.getDataPath(), 'YcbCrackerBox', "model.urdf"), basePosition=state_object_5)
        state_robot = p.getLinkState(self.pandaUid, 11)[0]
        state_fingers = (p.getJointState(self.pandaUid,9)[0], p.getJointState(self.pandaUid, 10)[0])
        self.observation = state_robot + state_fingers
        p.configureDebugVisualizer(p.COV_ENABLE_RENDERING,1)
        return np.array(self.observation).astype(np.float32)

    def render(self, mode='human'):
        view_matrix = p.computeViewMatrixFromYawPitchRoll(cameraTargetPosition=[0.7,0,0.6],
                                                            distance=.35,
                                                            yaw=-90,
                                                            pitch=0,
                                                            roll=0,
                                                            upAxisIndex=2)
        proj_matrix = p.computeProjectionMatrixFOV(fov=60,
                                                     aspect=float(960) /720,
                                                     nearVal=0.1,
                                                     farVal=0.4)
        (_, _, px, px_d, px_id) = p.getCameraImage(width=960,
                                              height=720,
                                              viewMatrix=view_matrix,
                                              projectionMatrix=proj_matrix,
                                              renderer=p.ER_BULLET_HARDWARE_OPENGL)

        rgb_array = np.array(px, dtype=np.uint8)
        rgb_array = np.reshape(rgb_array, (720,960, 4))
        rgb_array = rgb_array[:, :, :3]

        depth_array = np.array(px_d, dtype=np.float32)

        mask_array = np.array(px_id, dtype=np.uint8)

        return rgb_array, depth_array, mask_array, view_matrix, proj_matrix

    def _get_state(self):
        return self.observation

    def close(self):
        p.disconnect()
