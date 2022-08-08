''' Generate a forest environment, and collect paths using sst on the environnment
'''

import numpy as np
import sys

import skimage.morphology as skim
from skimage import io
import pickle
import os
from os import path as osp

from math import sin, cos, tan
from functools import partial

try:
    from ompl import base as ob
    # from ompl import geometric as og
    from ompl import control as oc

except ImportError:
    print("Could not find OMPL")
    raise ImportError("Run inside docker!!")

from utils import geom2pix  #, ValidityChecker
from generateMaps import generate_random_maps
import argparse

# All measurements are mentioned in meters
# Define global parameters
length = 24 # Size of the map
robot_radius = 0.2
dist_resl = 0.05
carLength = 0.3

# Define the space
# space = ob.RealVectorStateSpace(2)
space = ob.SE2StateSpace()

# Set the bounds 
bounds = ob.RealVectorBounds(2)
bounds.setLow(0)
bounds.setHigh(length)
space.setBounds(bounds)

cspace = oc.RealVectorControlSpace(space, 2)
cbounds = ob.RealVectorBounds(2)
cbounds.setLow(0, 0.0)
cbounds.setHigh(0, .3)
cbounds.setLow(1, -.5)
cbounds.setHigh(1, .5)
cspace.setBounds(cbounds)

# Define the SpaceInformation object.
si = ob.SpaceInformation(space)

class ValidityChecker(ob.StateValidityChecker):
    '''A class to check if an obstacle is in collision or not.
    '''
    def __init__(self, si, CurMap, MapMask=None, res=0.05, robot_radius=robot_radius):
        '''
        Intialize the class object, with the current map and mask generated
        from the transformer model.
        :param si: an object of type ompl.base.SpaceInformation
        :param CurMap: A np.array with the current map.
        :param MapMask: Areas of the map to be masked.
        '''
        super().__init__(si)
        self.size = CurMap.shape
        # Dilate image for collision checking
        InvertMap = np.abs(1-CurMap)
        InvertMapDilate = skim.dilation(InvertMap, skim.disk((robot_radius+0.1)/res))
        MapDilate = abs(1-InvertMapDilate)
        if MapMask is None:
            self.MaskMapDilate = MapDilate>0.5
        else:
            self.MaskMapDilate = np.logical_and(MapDilate, MapMask)
            
    def isValid(self, state):
        '''
        Check if the given state is valid.
        :param state: An ob.State object to be checked.
        :returns bool: True if the state is valid.
        '''
        x, y = state.getX(), state.getY()
        pix_dim = geom2pix([x, y], size=self.size)
        if pix_dim[0] < 0 or pix_dim[0] >= self.size[0] or pix_dim[1] < 0 or pix_dim[1] >= self.size[1]:
            return False
        return self.MaskMapDilate[pix_dim[1], pix_dim[0]]

def kinematicCarODE(q, u, qdot):
    '''
    Define the ODE of the car.
    '''
    theta = q[2]
    
    qdot[0] = u[0] * cos(theta)
    qdot[1] = u[0] * sin(theta)
    qdot[2] = u[0] * tan(u[1]) / carLength

def get_path(start, goal, ValidityCheckerObj=None, max_time=500):
    '''
    Get a path from start to goal using SST.
    :param start: og.State object.
    :param goal: og.State object.
    :param ValidityCheckerObj: An object of class ompl.base.StateValidityChecker
    :param max_time： float max seconds for planning 
    returns (np.array, np.array, success): A tuple of numpy arrays of a valid path,  
    interpolated path and whether the plan was successful or not.
    '''
    def isStateValid(spaceInformation, state):
        return spaceInformation.satisfiesBounds(state) and ValidityCheckerObj.isValid(state)
    success = False
    # Create a simple setup
    ss = oc.SimpleSetup(cspace)

    validityChecker = ob.StateValidityCheckerFn(partial(isStateValid, ss.getSpaceInformation()))
    ss.setStateValidityChecker(validityChecker)

    # Set the start and goal states:
    ss.setStartAndGoalStates(start, goal, 2.0)

    ode = oc.ODE(kinematicCarODE)
    odeSolver = oc.ODEBasicSolver(ss.getSpaceInformation(), ode)
    propagator = oc.ODESolver.getStatePropagator(odeSolver)
    ss.setStatePropagator(propagator)
    ss.getSpaceInformation().setPropagationStepSize(0.1)
    ss.getSpaceInformation().setMinMaxControlDuration(1, 20)

    # Use SST
    planner = oc.SST(ss.getSpaceInformation())

    ss.setPlanner(planner)

    # Attempt to solve within the given time
    time_inc = 60
    time = time_inc
    solved = ss.solve(time)
    while not ss.haveExactSolutionPath():
        solved = ss.solve(time_inc)
        time += time_inc
        if time > max_time:
            break
    if ss.haveExactSolutionPath():
        success = True
        print("Found solution")
        path = [
            [ss.getSolutionPath().getState(i).getX(), ss.getSolutionPath().getState(i).getY(), ss.getSolutionPath().getState(i).getYaw()]
            for i in range(ss.getSolutionPath().getStateCount())
            ]
        
        # Define path
        ss.getSolutionPath().interpolate()
        path_obj = ss.getSolutionPath()
        path_interpolated = np.array([
            [path_obj.getState(i).getX(), path_obj.getState(i).getY(), path_obj.getState(i).getYaw()] 
            for i in range(path_obj.getStateCount())
            ])        
       
    else:
        path = [[start().getX(), start().getY(), start().getYaw()], [goal().getX(), goal().getY(), goal().getYaw()]]
        path_interpolated = []

    return np.array(path), np.array(path_interpolated), success


def start_experiment_sst(start, samples, fileDir=None):
    '''
    Run the experiment for random start and goal points.
    :param start: The start index of the samples
    :param samples: The number of samples to collect
    :param fileDir: Directory with the map and paths
    '''
    assert osp.isdir(fileDir), f"{fileDir} is not a valid directory"

    envNum = int(fileDir[-6:])
    CurMap = io.imread(osp.join(fileDir, f'map_{envNum}.png'), as_gray=True)
    # Validity checking
    ValidityCheckerObj = ValidityChecker(si, CurMap=CurMap)
    si.setStateValidityChecker(ValidityCheckerObj)

    for i in range(start, start+samples):
        path_param = {}
        success = False
        while not success:
            sg_ok = False
            while not sg_ok:
                # Define the start and goal location
                start = ob.State(space)
                start.random()
                while not ValidityCheckerObj.isValid(start()):
                    start.random()
                goal = ob.State(space)
                goal.random()
                while not ValidityCheckerObj.isValid(goal()):   
                    goal.random()
                    dist = np.sqrt((start().getX() - goal().getX()) ** 2 + (start().getY() - goal().getY()) ** 2)
                    if dist > 4 and dist < 15:
                        sg_ok = True

            path, path_interpolated, success = get_path(start, goal, ValidityCheckerObj)
            if success:
                path_param['path'] = path
                path_param['path_interpolated'] = path_interpolated
                path_param['success'] = success

                pickle.dump(path_param, open(osp.join(fileDir,f'path_{i}.p'), 'wb'))

def start_map_collection_sst(start, samples, numPaths, fileDir):
    '''
    Collect a single path for the given number of samples.
    :param start: The start index of the samples.
    :param samples: The number of samples to collect.
    :params numPaths: Number of paths to collect for each environment
    :param fileDir: The base folder to save the data files
    '''
    for i in range(start, start+samples):
        fileDir = osp.join(fileDir, f'env{i:06d}')
        if not osp.isdir(fileDir):
            os.mkdir(fileDir) 
        fileName = osp.join(fileDir, f'map_{i}.png')
        generate_random_maps(width=length, seed=i+200, fileName=fileName)
        start_experiment_sst(0, numPaths, fileDir=fileDir)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--start', help="The start index of the environment", type=int)
    parser.add_argument('--numEnv', help="The number of environments to collect data", type=int)
    parser.add_argument('--numPaths', help="Number of paths to collect", type=int)
    parser.add_argument('--fileDir', help="Location to save collected data.")

    args = parser.parse_args()
    start_map_collection_sst(args.start, args.numEnv, args.numPaths, args.fileDir)