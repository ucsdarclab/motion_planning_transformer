'''A script for generating patches
'''

import torch
import torch.nn as nn
import torch.nn.functional as F
import skimage.io
import skimage.morphology as skim
import numpy as np
import pickle

from os import path as osp
import argparse
import json

try:
    from ompl import base as ob
    from ompl import geometric as og
    from ompl import util as ou
except ImportError:
    raise ImportError("Container does not have OMPL installed")

from transformer import Models
from utils import geom2pix, ValidityChecker

res = 0.05
length = 24


def pix2geom(pos, res=0.05, length=24):
    """
    Converts pixel co-ordinates to geometrical positions. 
    :param pos: The (x,y) pixel co-ordinates.
    :param res: The distance represented by each pixel.
    :param length: The length of the map in meters.
    :returns (float, float): The associated eucledian co-ordinates.
    """
    return (pos[0]*res, length-pos[1]*res)


def get_encoder_input(InputMap, goal_pos, start_pos):
    '''
    Returns the input map appended with the goal, and start position encoded.
    :param InputMap: The grayscale map
    :param goal_pos: The goal pos of the robot on the costmap.
    :param start_pos: The start pos of the robot on the costmap.
    :returns np.array: The map concatentated with the encoded start and goal pose.
    '''
    map_size = InputMap.shape
    assert len(map_size) == 2, "This only works for 2D maps"
    
    context_map = np.zeros(map_size)
    goal_start_x = max(0, goal_pos[0]- receptive_field//2)
    goal_start_y = max(0, goal_pos[1]- receptive_field//2)
    goal_end_x = min( map_size[0], goal_pos[0]+ receptive_field//2)
    goal_end_y = min( map_size[1], goal_pos[1]+ receptive_field//2)
    context_map[goal_start_x:goal_end_x, goal_start_y:goal_end_y] = 1.0
    # Mark start region
    start_start_x = max(0, start_pos[0]- receptive_field//2)
    start_start_y = max(0, start_pos[1]- receptive_field//2)
    start_end_x = min( map_size[0], start_pos[0]+ receptive_field//2)
    start_end_y = min( map_size[1], start_pos[1]+ receptive_field//2)
    context_map[start_start_x:start_end_x, start_start_y:start_end_y] = -1.0
    return torch.as_tensor(np.concatenate((InputMap[None, :], context_map[None, :])))


receptive_field = 32
hashTable = [(20*r+4, 20*c+4) for c in range(24) for r in range(24)]



# Planning parameters
space = ob.RealVectorStateSpace(2)
bounds = ob.RealVectorBounds(2)
bounds.setLow(0.0)
bounds.setHigh(length)
space.setBounds(bounds)
si = ob.SpaceInformation(space)


def get_path(start, goal, input_map, patch_map):
    '''
    Plan a path given the start, goal and patch_map.
    :param start: 
    :param goal:
    :param patch_map:
    returns bool: Returns True if a path was planned successfully.
    '''

    # Tried importance sampling, but seems like it makes not much improvement 
    # over rejection sampling.
    ValidityCheckerObj = ValidityChecker(si, input_map, patch_map)
    si.setStateValidityChecker(ValidityCheckerObj)

    StartState = ob.State(space)
    # import pdb;pdb.set_trace()
    StartState[0] = start[0]
    StartState[1] = start[1]

    GoalState = ob.State(space)
    GoalState[0] = goal[0]
    GoalState[1] = goal[1]

    success = False
    ss = og.SimpleSetup(si)

    # Set the start and goal States:
    ss.setStartAndGoalStates(StartState, GoalState, 0.1)

    planner = og.RRTstar(si)
    ss.setPlanner(planner)

    time = 1
    solved = ss.solve(time)
    while not ss.haveExactSolutionPath():
        solved = ss.solve(2)
        time += 2
        if time>10:
            break
    if ss.haveExactSolutionPath():
        success = True
        print("Found Solution")

    return success

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
    encoder_input = get_encoder_input(input_map, goal_pos, start_pos)
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
        goal_end_x = min(map_size[0], pos[0]+ receptive_field//2)
        goal_end_y = min(map_size[1], pos[1]+ receptive_field//2)
        patch_map[goal_start_y:goal_end_y, goal_start_x:goal_end_x] = 1.0
    return patch_map

device='cuda' if torch.cuda.is_available() else 'cpu'

if __name__=="__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--modelFolder', help='Directory where model_params.json exists', required=True)
    parser.add_argument('--valDataFolder', help='Directory where training data exists', required=True)
    # parser.add_argument('--envNum', help='Environment number to validate model', required=True)
    parser.add_argument('--start', help='Start of environment number', required=True, type=int)
    parser.add_argument('--epoch', help='Model epoch number to test', required=True, type=int)
    parser.add_argument('--numPaths', help='Number of start and goal pairs for each env', default=1, type=int)

    args = parser.parse_args()

    modelFolder = args.modelFolder
    modelFile = osp.join(modelFolder, f'model_params.json')
    assert osp.isfile(modelFile), f"Cannot find the model_params.json file in {modelFolder}"

    # env_num = args.envNum
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
    # /root/data/maze/val
    valDataFolder = args.valDataFolder
    # Only do evaluation
    transformer.eval()
    # Get path data
    PathSuccess = []
    for env_num in range(start, start+20):
        temp_map =  osp.join(valDataFolder, f'env{env_num:06d}/map_{env_num}.png')
        small_map = skimage.io.imread(temp_map, as_gray=True)

        for pathNum in range(args.numPaths):
        # pathNum = 0
            pathFile = osp.join(valDataFolder, f'env{env_num:06d}/path_{pathNum}.p')
            data = pickle.load(open(pathFile, 'rb'))
            path = data['path_interpolated']

            if data['success']:
                goal_pos = geom2pix(path[0, :])
                start_pos = geom2pix(path[-1, :])

                # Identitfy Anchor points
                encoder_input = get_encoder_input(small_map, goal_pos, start_pos)
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
                    goal_end_x = min(map_size[0], pos[0]+ receptive_field//2)
                    goal_end_y = min(map_size[1], pos[1]+ receptive_field//2)
                    patch_map[goal_start_y:goal_end_y, goal_start_x:goal_end_x] = 1.0

                PathSuccess.append(get_path(path[0, :], path[-1, :], small_map, patch_map))
            else:
                PathSuccess.append(False)

    pickle.dump(PathSuccess, open(osp.join(modelFolder, f'eval_val_plan_{start:06d}.p'), 'wb'))
    print(sum(PathSuccess))