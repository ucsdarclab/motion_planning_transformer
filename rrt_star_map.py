''' Generate a map using matplotlib and save it.
'''
import numpy as np
import sys

import skimage.morphology as skim
from skimage import io
from skimage import color
import pickle
import os
from os import path as osp

try:
    from ompl import base as ob
    from ompl import geometric as og
except ImportError:
    print("Could not find OMPL")
    raise ImportError("Run inside docker!!")

from utils import geom2pix, ValidityChecker
from generateMaps import generate_random_maps


# All measurements are mentioned in meters
# Define global parameters
length = 24 # Size of the map
robot_radius = 0.1
dist_resl = 0.05

# Define the space
space = ob.RealVectorStateSpace(2)

# Set the bounds 
bounds = ob.RealVectorBounds(2)
bounds.setLow(0)
bounds.setHigh(length)
space.setBounds(bounds)
# Define the SpaceInformation object.
si = ob.SpaceInformation(space)

def get_path(start, goal, ValidityCheckerObj=None):
    '''
    Get a RRT path from start and goal.
    :param start: og.State object.
    :param goal: og.State object.
    :param ValidityCheckerObj: An object of class ompl.base.StateValidityChecker
    returns (np.array, np.array, success): A tuple of numpy arrays of a valid path,  
    interpolated path and whether the plan was successful or not.
    '''
    success = False
    # Create a simple setup
    ss = og.SimpleSetup(si)

    # Set the start and goal states:
    ss.setStartAndGoalStates(start, goal, 0.1)

    # # Use RRT*
    planner = og.RRTstar(si)

    ss.setPlanner(planner)

    # Attempt to solve within the given time
    time = 2
    solved = ss.solve(time)
    while not ss.haveExactSolutionPath():
        solved = ss.solve(2.0)
        time +=2
        if time>10:
            break
    if ss.haveExactSolutionPath():
        success = True
        print("Found solution")
        path = [
            [ss.getSolutionPath().getState(i)[0], ss.getSolutionPath().getState(i)[1]]
            for i in range(ss.getSolutionPath().getStateCount())
            ]
        # Define path
        ss.getSolutionPath().interpolate(100)
        path_obj = ss.getSolutionPath()
        path_interpolated = np.array([
            [path_obj.getState(i)[0], path_obj.getState(i)[1]] 
            for i in range(path_obj.getStateCount())
            ])
    else:
        path = [[start[0], start[1]], [goal[0], goal[1]]]
        path_interpolated = []

    return np.array(path), np.array(path_interpolated), success


def start_experiment_rrt(start, samples, fileDir=None):
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
        # Define the start and goal location
        start = ob.State(space)
        start.random()
        while not ValidityCheckerObj.isValid(start()):
            start.random()
        goal = ob.State(space)
        goal.random()
        while not ValidityCheckerObj.isValid(goal()):   
            goal.random()

        path, path_interpolated, success = get_path(start, goal, ValidityCheckerObj)
        path_param['path'] = path
        path_param['path_interpolated'] = path_interpolated
        path_param['success'] = success

        pickle.dump(path_param, open(osp.join(fileDir,f'path_{i}.p'), 'wb'))

def start_map_collection_rrt(start, samples):
    '''
    Collect a single path for the given number of samples.
    :param start: The start index of the samples.
    :param samples: The number of samples to collect.
    '''
    for i in range(start, start+samples):
        fileDir = f'/root/data/train2/env{i:06d}'
        if not osp.isdir(fileDir):
            os.mkdir(fileDir) 
        fileName = osp.join(fileDir, f'map_{i}.png')
        generate_random_maps(length=length, seed=i+200, fileName=fileName)
        start_experiment_rrt(0, 10, fileDir=fileDir)

if __name__ == "__main__":
    start, samples = int(sys.argv[1]), int(sys.argv[2])
    # start_experiment_rrt(start, samples, '/root/data/val')
    start_map_collection_rrt(start, samples)