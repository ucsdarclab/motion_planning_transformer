'''A script for planning usng MPNet
'''
import skimage.io
import skimage.morphology as skim
import numpy as np
import pickle

import torch

try:
    from ompl import base as ob
    from ompl import geometric as og
    from ompl import util as ou
except ImportError:
    raise ImportError("Container does not have OMPL installed")

from os import path as osp
import argparse
import time
import json

from utils import ValidityChecker
from mpnet import Models

res = 0.05

def normalize_state(state, worldBounds):
    '''
    Returns a normalized state between -1 and 1 for the given state.
    :param state: A np.array of the state to be normalized.
    :param worldBounds: A np.array of the world bounds in.
    :return np.array: A normalized state between -1 adn 1.
    '''
    return (state/worldBounds)*2 - 1

def scale_state(state, length=24):
    '''
    Scale up state
    :param state: The state of the robot
    '''
    return length*(state+1)/2


def check_edge_collision(start, goal, checkerFn):
    '''
    Returns True if collision between the edge b/w start and goal in collision
    :param start: The start state of the robot
    :param goal: The goal state of the robot
    :param checkerFn: A collision checking function that takes a 2d state
    :returns bool: True if path is in collision else False
    '''
    alpha = np.linspace(0, 1, int(np.floor(np.linalg.norm(start-goal)/0.5)))[:, None]
    interTraj = (1-alpha)*start + alpha*goal
    for pos in interTraj:
        if not checkerFn(pos):
            return True
    return False


def check_trajectory_collision(traj, collisionFn):
    '''
    Returns True if a planned path has collision
    :param traj: A trajectory represented by a numpy array
    :param collisionFn: A collision checking function
    '''
    for i, pos  in enumerate(traj[:-1,:]):
        if check_edge_collision(pos, traj[i+1, :], collisionFn):
            return True
    return False


def plan_path_mpnet(model, enc, start, goal, device, collisionFn, worldBounds):
    '''
    Plan a path using MPNet.
    :param model: The NN model used for planning.
    :param enc: The latent space representation of the map.
    :param start: A numpy array of size 2 with the normalized start state.
    :param goal: A numpy array of size 2 with the normalized goal state.
    :param device: The device used for torch computations.
    :param collisionFn: A function to check if a state is in collision
    :param worldBounds: A numpy array of worldBounds
    :returns np.array: A set of scaled trajectories
    '''
    goalS = torch.tensor(normalize_state(goal, worldBounds)[None,:], device=device, dtype=torch.float)
    startS = torch.tensor(normalize_state(start, worldBounds)[None,:], device=device, dtype=torch.float)
    reachedGoal = False
    normPredTraj = [startS.cpu().numpy().squeeze()]
    vertex = 0
    for _ in range(10):
        # Check if we can connect start and goal state
        if not check_edge_collision(scale_state(startS.cpu().numpy().squeeze()), goal, collisionFn):
            reachedGoal = True
            break
        inputs = torch.cat([enc, startS, goalS], dim=1)
        for _ in range(10):
            with torch.no_grad():
                tempS = model(inputs)
                temp = tempS.cpu().numpy().squeeze()
            if collisionFn(scale_state(temp)):
                vertex +=1                
                startS = tempS
                normPredTraj.append(temp)
                break
        # Check if we can connect to the goal or reached near it
        if torch.linalg.norm(startS-goalS)*24<0.1:
            reachedGoal = True
            break
    if reachedGoal:
        normPredTraj.append(normalize_state(goal, worldBounds))
    return scale_state(np.array(normPredTraj)), reachedGoal, vertex


def plan_path_mpnet_bidirection(model, enc, start, goal, device, collisionFn, worldBounds):
    '''
    Return a path planned in a bi-direction manner.
    :param model: The NN model used for planning.
    :param enc: The latent space representation of the map.
    :param start: A numpy array of size 2 with the normalized start state.
    :param goal: A numpy array of size 2 with the normalized goal state.
    :param device: The device used for torch computations.
    :param collisionFn: A function to check if a state is in collision
    :returns np.array: A set of scaled trajectories
    '''
    goalS = torch.tensor(normalize_state(goal, worldBounds)[None,:], device=device, dtype=torch.float)
    startS = torch.tensor(normalize_state(start, worldBounds)[None,:], device=device, dtype=torch.float)
    reachedGoal = False
    normPredTrajF = [startS.cpu().numpy().squeeze()]
    normPredTrajB = [goalS.cpu().numpy().squeeze()]
    normPredTraj = [normalize_state(start, worldBounds), normalize_state(goal, worldBounds)]
    forward = True
    vertex = 0
    for _ in range(10):
        # Check if we can connect start and goal state
        if not check_edge_collision(
            scale_state(startS.cpu().numpy().squeeze()), 
            scale_state(goalS.cpu().numpy().squeeze()), 
            collisionFn
        ):
            reachedGoal = True
            break
        inputs = torch.cat([enc, startS, goalS], dim=1)
        with torch.no_grad():
            tempS = model(inputs)
            temp = tempS.cpu().numpy().squeeze()
        if collisionFn(scale_state(temp)):
            vertex += 1
            if forward:
                startS = goalS
                goalS = tempS
                normPredTrajF.append(temp)
            else:
                goalS = startS
                startS = tempS
                normPredTrajB.insert(0, temp)
        # Check if we can connect to the goal or have reached near it
        if torch.linalg.norm(startS-goalS)*24<0.1:
            reachedGoal = True
            break
    if reachedGoal:
        normPredTraj = normPredTrajF + normPredTrajB
    return scale_state(np.array(normPredTraj)), reachedGoal, vertex


def plan_path_rrt(start, goal, space, si):
    '''
    Returns a planned path using RRT for given start and goal state.
    :param start: The start state of the robot
    :param goal: The goal state of the robot
    :param space: A ompl.base.Space object
    :param si: An ompl.base.SpaceInformation object
    '''
    startState = ob.State(space)
    startState[0] = float(start[0])
    startState[1] = float(start[1])

    goalState = ob.State(space)
    goalState[0] = float(goal[0])
    goalState[1] = float(goal[1])
    
    pdef = ob.ProblemDefinition(si)
    pdef.setStartAndGoalStates(startState, goalState, 0.1)
    
    planner = og.RRTstar(si)
    
    planner.setProblemDefinition(pdef) 
    planner.setup()
    
    for _ in range(4):
        solved = planner.solve(0.25)
        if pdef.hasExactSolution():
            path = np.array( [[pdef.getSolutionPath().getState(i)[0], pdef.getSolutionPath().getState(i)[1]]
                for i in range(pdef.getSolutionPath().getStateCount())])
            plannerData = ob.PlannerData(si)
            planner.getPlannerData(plannerData)
            return path, True, plannerData.numVertices()
    return [], False, 0


def simplify_path(traj, collisionFn):
    '''
    Simplify a given trajectory by removing un-necessary nodes.
    :param traj: A numpy array, with the trajectory to be simplified.
    :param collisionFn: A function that can be use to check the collision status of a state
    :return np.array: A simplified trajectory
    '''
    if len(traj)==1:
        return traj
    for i, pos in enumerate(traj[:1:-1]):
        if not check_edge_collision(traj[0, :], pos, collisionFn):
            return np.r_[traj[0,:][None, :], pos[None, :], simplify_path(traj[-(i+1):], collisionFn)]
    return np.r_[traj[0, :][None,:], simplify_path(traj[1:], collisionFn)]


def get_path(start, goal, small_map, model, device, worldBounds):
    '''
    Plan a path using MPNet
    :param start: the start state of the robot.
    :param goal: The goal state of the robot
    :param small_map: The map used for planning
    :param model: The NN model for MPNet
    :param device: The device for torch computations
    :param worldBounds: The bounds of the training data
    '''
    mapSize = small_map.shape
    space = ob.RealVectorStateSpace(2)
    bounds = ob.RealVectorBounds(2)
    bounds.setLow(0.0)
    bounds.setHigh(0, mapSize[1]*res) # Set width bounds (x)
    bounds.setHigh(1, mapSize[0]*res) # Set height bounds (y)
    space.setBounds(bounds)
    si = ob.SpaceInformation(space)
    ValidityCheckerObj = ValidityChecker(si, small_map)
    si.setStateValidityChecker(ValidityCheckerObj)
    v = 0

    startTime = time.time()
    enc = model.get_environment_encoding(torch.tensor(small_map[None, None, :, :], dtype=torch.float, device=device))
    for _ in range(10):
        predTraj, reachedGoal, vertex = plan_path_mpnet(model, enc, start, goal, device, ValidityCheckerObj.isValid, worldBounds)
        v += vertex
        validTraj = predTraj[0, :][None, :]
        for i, pos in enumerate(predTraj[:-1]):
            if check_edge_collision(pos, predTraj[i+1, :], ValidityCheckerObj.isValid):
                newTraj, success, tmpvertex = plan_path_mpnet(model, enc, pos, predTraj[i+1, :], device, ValidityCheckerObj.isValid, worldBounds)
                v += tmpvertex if success else 0
                # Check if trajectory is successful, else replan
                if check_trajectory_collision(newTraj, ValidityCheckerObj.isValid):
                    newTraj, success, tmpvertex = plan_path_rrt(pos, predTraj[i+1, :], space, si)
                    v += tmpvertex if success else 0
                if success:
                    validTraj = np.r_[validTraj, newTraj[1:, :]]
            else:
                validTraj = np.r_[validTraj, predTraj[i+1, :][None, :]]
        if reachedGoal:
            validTraj = simplify_path(validTraj, ValidityCheckerObj.isValid)
            break
    planTime = time.time() - startTime

    return validTraj, planTime, v, reachedGoal

device='cuda' if torch.cuda.is_available() else 'cpu'

if __name__=="__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--modelFolder', help='Directory where model_params.json exists', required=True)
    parser.add_argument('--valDataFolder', help='Directory where training data exists', required=True)
    parser.add_argument('--start', help='Start of environment number', required=True, type=int)
    parser.add_argument('--samples', help='Number of envs', required=True, type=int)
    parser.add_argument('--numPaths', help='Number of start and goal pairs for each env', default=1, type=int)
    parser.add_argument('--epoch', help='Model epoch number to test', required=True, type=int)

    args = parser.parse_args()

    modelFolder = args.modelFolder
    modelFile = osp.join(modelFolder, f'model_params.json')
    assert osp.isfile(modelFile), f"Cannot find the model_params.json file in {modelFolder}"

    start = args.start
    samples = args.samples

    model_param = json.load(open(modelFile))
    model = Models.MPNet(**model_param)

    model.to(device)

    # Load model parameters
    epoch = args.epoch
    checkpoint = torch.load(osp.join(modelFolder, f'model_epoch_{epoch}.pkl'))
    model.load_state_dict(checkpoint['state_dict'])

    # valDataFolder
    valDataFolder = args.valDataFolder
    pathSuccess = []
    pathTime = []
    pathVertices = []
    for env_num in range(start, start+samples):
        temp_map =  osp.join(valDataFolder, f'env{env_num:06d}/map_{env_num}.png')
        small_map = skimage.io.imread(temp_map, as_gray=True)

        for pathNum in range(args.numPaths):
            pathFile = osp.join(valDataFolder, f'env{env_num:06d}/path_{pathNum}.p')
            data = pickle.load(open(pathFile, 'rb'))
            path = data['path_interpolated']
            print(f"Env Num: {env_num}: {data['success']}")
            if data['success']:
                cost = np.linalg.norm(np.diff(path, axis=0), axis=1).sum()
                _, t, v, s = get_path(path[0, :], path[-1, :], small_map, model, device, np.array([24.0, 24.0]))
                pathSuccess.append(s)
                pathTime.append(t)
                pathVertices.append(v)
            else:
                pathSuccess.append(False)
                pathTime.append(0)
                pathVertices.append(0)

    pathData = {'Time':pathTime, 'Success':pathSuccess, 'Vertices':pathVertices}
    pickle.dump(pathData, open(osp.join(modelFolder, f'eval_val_plan_mpnet_{start:06d}.p'), 'wb'))
    print(sum(pathSuccess))