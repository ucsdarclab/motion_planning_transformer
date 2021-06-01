'''A script for generating patches
'''

import torch
import torch.nn as nn
import torch.nn.functional as F
import skimage.io
import numpy as np
import pickle

from os import path as osp
import argparse
import json
import time

try:
    from ompl import base as ob
    from ompl import geometric as og
    from ompl import util as ou
except ImportError:
    raise ImportError("Container does not have OMPL installed")

from transformer import Models
from utils import geom2pix, ValidityChecker
from dataLoader import get_encoder_input

res = 0.05


def pix2geom(pos, res=0.05, length=24):
    """
    Converts pixel co-ordinates to geometrical positions. 
    :param pos: The (x,y) pixel co-ordinates.
    :param res: The distance represented by each pixel.
    :param length: The length of the map in meters.
    :returns (float, float): The associated eucledian co-ordinates.
    """
    return (pos[0]*res, length-pos[1]*res)


receptive_field = 32

def getHashTable(mapSize):
    '''
    Return the hashTable for the given map
    NOTE: This hastable only works for the  patch_embedding network defined in the
    transformers/Models.py file.
    :param mapSize: The size of the map
    :returns list: the hashTable to convert 1D token index to 2D image positions
    '''
    H, W = mapSize
    Hhat = np.floor((H-8)/4) - 1
    What = np.floor((W-8)/4) - 1
    if Hhat%5>0:
        padH = int(((Hhat//5)+1)*5 - Hhat)
    else:
        padH = 0

    if What%5>0:
        padW = int(((What//5)+1)*5 - What)
    else:
        padW = 0

    tokenH = int((Hhat+padH)/5)
    tokenW = int((What+padW)/5)
    return [(20*r+4, 20*c+4) for c in range(tokenH) for r in range(tokenW)]


def getPathLengthObjective(cost, si):
    '''
    Return the threshold objective for early termination
    :param cost: The cost of the original RRT* path
    :param si: An object of class ob.SpaceInformation
    :returns : An object of class ob.PathLengthOptimizationObjective
    '''
    obj = ob.PathLengthOptimizationObjective(si)
    obj.setCostThreshold(ob.Cost(cost))
    return obj


def get_path(start, goal, input_map, patch_map, plannerType, cost):
    '''
    Plan a path given the start, goal and patch_map.
    :param start:
    :param goal:
    :param patch_map:
    :param plannerType: The planner type to use
    :param cost: The cost of the path
    returns bool: Returns True if a path was planned successfully.
    '''
    mapSize = input_map.shape
    # Planning parametersf
    space = ob.RealVectorStateSpace(2)
    bounds = ob.RealVectorBounds(2)
    bounds.setLow(0.0)
    bounds.setHigh(0, mapSize[1]*res) # Set width bounds (x)
    bounds.setHigh(1, mapSize[0]*res) # Set height bounds (y)
    space.setBounds(bounds)
    si = ob.SpaceInformation(space)
    ValidityCheckerObj = ValidityChecker(si, input_map, patch_map)
    si.setStateValidityChecker(ValidityCheckerObj)

    StartState = ob.State(space)
    StartState[0] = start[0]
    StartState[1] = start[1]

    GoalState = ob.State(space)
    GoalState[0] = goal[0]
    GoalState[1] = goal[1]

    success = False

    # Define planning problem
    pdef = ob.ProblemDefinition(si)
    pdef.setStartAndGoalStates(StartState, GoalState, 0.1)

    # Set up the objective function
    obj = getPathLengthObjective(cost, si)
    pdef.setOptimizationObjective(obj)
    
    if plannerType=='rrtstar':
        planner = og.RRTstar(si)
    elif plannerType=='informedrrtstar':
        planner = og.InformedRRTstar(si)
    else:
        raise TypeError(f"Planner Type {plannerType} not found")
    
    # Set the problem instance the planner has to solve

    planner.setProblemDefinition(pdef)
    planner.setup()

    # Attempt to solve the planning problem in the given time
    startTime = time.time()
    solved = planner.solve(30.0)
    planTime = time.time()-startTime

    plannerData = ob.PlannerData(si)
    planner.getPlannerData(plannerData)
    numVertices = plannerData.numVertices()

    if pdef.hasExactSolution():
        success = True

        print("Found Solution")
        path = [
            [pdef.getSolutionPath().getState(i)[0], pdef.getSolutionPath().getState(i)[1]]
            for i in range(pdef.getSolutionPath().getStateCount())
            ]
    else:
        path = [[start[0], start[1]], [goal[0], goal[1]]]

    return path, planTime, numVertices, success

device='cuda' if torch.cuda.is_available() else 'cpu'


def get_patch(model, start_pos, goal_pos, input_map):
    '''
    Return the patch map for the given start and goal position, and the network
    architecture.
    :param model:
    :param start: 
    :param goal:
    :param input_map:
    '''
    # Identitfy Anchor points
    encoder_input = get_encoder_input(input_map, goal_pos[::-1], start_pos[::-1])
    hashTable = getHashTable(input_map.shape)
    predVal = model(encoder_input[None,:].float().cuda())
    predClass = predVal[0, :, :].max(1)[1]

    predProb = F.softmax(predVal[0, :, :], dim=1)
    possAnchor = [hashTable[i] for i, label in enumerate(predClass) if label==1]

    # Generate Patch Maps
    patch_map = np.zeros_like(input_map)
    map_size = input_map.shape
    for pos in possAnchor:
        goal_start_x = max(0, pos[0]- receptive_field//2)
        goal_start_y = max(0, pos[1]- receptive_field//2)
        goal_end_x = min(map_size[1], pos[0]+ receptive_field//2)
        goal_end_y = min(map_size[0], pos[1]+ receptive_field//2)
        patch_map[goal_start_y:goal_end_y, goal_start_x:goal_end_x] = 1.0
    return patch_map

device='cuda' if torch.cuda.is_available() else 'cpu'

if __name__=="__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        '--plannerType', 
        help='The underlying sampler to use', 
        required=True, 
        choices=['rrtstar', 'informedrrtstar']
    )
    parser.add_argument('--modelFolder', help='Directory where model_params.json exists', required=True)
    parser.add_argument('--valDataFolder', help='Directory where training data exists', required=True)
    parser.add_argument('--start', help='Start of environment number', required=True, type=int)
    parser.add_argument('--numEnv', help='Number of environments', required=True, type=int)
    parser.add_argument('--epoch', help='Model epoch number to test', required=True, type=int)
    parser.add_argument('--numPaths', help='Number of start and goal pairs for each env', default=1, type=int)

    args = parser.parse_args()

    modelFolder = args.modelFolder
    modelFile = osp.join(modelFolder, f'model_params.json')
    assert osp.isfile(modelFile), f"Cannot find the model_params.json file in {modelFolder}"

    start = args.start

    model_param = json.load(open(modelFile))
    transformer = Models.Transformer(
        **model_param
    )

    transformer.to(device)

    receptive_field=32
    # Load model parameters
    epoch = args.epoch
    checkpoint = torch.load(osp.join(modelFolder, f'model_epoch_{epoch}.pkl'))
    transformer.load_state_dict(checkpoint['state_dict'])

    # valDataFolder
    valDataFolder = args.valDataFolder
    # Only do evaluation - Need this for the problem to work with maps of different sizes.
    transformer.eval()
    # Get path data
    pathSuccess = []
    pathTime = []
    pathVertices = []
    for env_num in range(start, start+args.numEnv):
        temp_map =  osp.join(valDataFolder, f'env{env_num:06d}/map_{env_num}.png')
        small_map = skimage.io.imread(temp_map, as_gray=True)
        mapSize = small_map.shape
        hashTable = getHashTable(mapSize)
        for pathNum in range(args.numPaths):
        # pathNum = 0
            pathFile = osp.join(valDataFolder, f'env{env_num:06d}/path_{pathNum}.p')
            data = pickle.load(open(pathFile, 'rb'))
            path = data['path_interpolated']
            
            if data['success']:
                goal_pos = geom2pix(path[0, :], size=mapSize)
                start_pos = geom2pix(path[-1, :], size=mapSize)

                # Identitfy Anchor points
                encoder_input = get_encoder_input(small_map, goal_pos, start_pos)
                # NOTE: Currently only valid for map sizes of certain multiples.
                predVal = transformer(encoder_input[None,:].float().cuda())
                predClass = predVal[0, :, :].max(1)[1]

                predProb = F.softmax(predVal[0, :, :], dim=1)
                possAnchor = [hashTable[i] for i, label in enumerate(predClass) if label==1]

                # Generate Patch Maps
                patch_map = np.zeros_like(small_map)
                map_size = small_map.shape
                for pos in possAnchor:
                    goal_start_x = max(0, pos[0]- receptive_field//2)
                    goal_start_y = max(0, pos[1]- receptive_field//2)
                    goal_end_x = min(map_size[1], pos[0]+ receptive_field//2)
                    goal_end_y = min(map_size[0], pos[1]+ receptive_field//2)
                    patch_map[goal_start_y:goal_end_y, goal_start_x:goal_end_x] = 1.0
                cost = np.linalg.norm(np.diff(path, axis=0), axis=1).sum()
                _, t, v, s = get_path(path[0, :], path[-1, :], small_map, patch_map, args.plannerType, cost)
                pathSuccess.append(s)
                pathTime.append(t)
                pathVertices.append(v)
            else:
                pathSuccess.append(False)
                pathTime.append(0)
                pathVertices.append(0)

    pathData = {'Time':pathTime, 'Success':pathSuccess, 'Vertices':pathVertices}
    pickle.dump(pathData, open(osp.join(modelFolder, f'eval_val_plan_{args.plannerType}_{start:06d}.p'), 'wb'))
    print(sum(pathSuccess))