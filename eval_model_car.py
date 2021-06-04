'''A script for evaluating the Car Model.
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

from functools import partial

try:
    from ompl import base as ob
    # from ompl import geometric as og
    from ompl import control as oc

    from ompl import util as ou
except ImportError:
    raise ImportError("Container does not have OMPL installed")

from transformer import Models
from utils import geom2pix
from math import sin, cos, tan, pi
from pathlib import Path

from dataLoader import get_encoder_input
res = 0.05
dist_resl = res
length = 24
robot_radius = 0.2
carLength = 0.3


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
hashTable = [(20*r+4, 20*c+4) for c in range(24) for r in range(24)]



# Planning parameters
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
ss = oc.SimpleSetup(cspace)
si = ob.SpaceInformation(space)

def kinematicCarODE(q, u, qdot):
    theta = q[2]
    
    qdot[0] = u[0] * cos(theta)
    qdot[1] = u[0] * sin(theta)
    qdot[2] = u[0] * tan(u[1]) / carLength

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

def get_path_sst(start, goal, input_map, patch_map):
    '''
    Plan a path using SST, but invert the start and goal location.
    :param start: The starting position on map co-ordinates.
    :param goal: The goal position on the map co-ordinates.
    :param input_map: The input map 
    :param patch_map: The patch map
    :returns 
    '''
    success, time, _, path = get_path(start, goal, input_map, patch_map, use_valid_sampler=True)
    return path, time, [], success


def get_path(start, goal, input_map, patch_map, step_time=0.1, max_time=300, use_valid_sampler=False):
    '''
    Plan a path given the start, goal and patch_map.
    :param start: 
    :param goal:
    :param patch_map:
    returns bool: Returns True if a path was planned successfully.
    '''
    # Tried importance sampling, but seems like it makes not much improvement 
    # over rejection sampling.

    StartState = ob.State(space)
    # import pdb;pdb.set_trace()
    StartState().setX(start[0])
    StartState().setY(start[1]) 
    StartState().setYaw(start[2]) 

    GoalState = ob.State(space)
    GoalState().setX(goal[0])
    GoalState().setY(goal[1])
    GoalState().setYaw(goal[2]) 

    success = False
    ss = oc.SimpleSetup(cspace)
    # setup validity checker
    ValidityCheckerObj = ValidityChecker(si, input_map, patch_map)
    def isStateValid(spaceInformation, state):
        return spaceInformation.satisfiesBounds(state) and ValidityCheckerObj.isValid(state)
    
    class MyValidStateSampler(ob.ValidStateSampler):
     def __init__(self, si):
        super(MyValidStateSampler, self).__init__(si)
        self.name_ = "my sampler"
        self.rng_ = ou.RNG()
        self.size = input_map.shape
        # Dilate image for collision checking
        InvertMap = np.abs(1-input_map)
        InvertMapDilate = skim.dilation(InvertMap, skim.disk((robot_radius+0.1)/res))
        MapDilate = abs(1-InvertMapDilate)
        if patch_map is None:
            self.MaskMapDilate = MapDilate>0.5
        else:
            self.MaskMapDilate = np.logical_and(MapDilate, patch_map)
    
     def sample(self, state):
        x, y = self.rng_.uniformReal(0, 24), self.rng_.uniformReal(0, 24)
        pix_dim = geom2pix([x, y], size=self.size)
        while pix_dim[0] < 0 or pix_dim[0] >= self.size[0] or pix_dim[1] < 0 or pix_dim[1] >= self.size[1] or self.MaskMapDilate[pix_dim[1], pix_dim[0]] > 0.5:
            x, y = self.rng_.uniformReal(0, 24), self.rng_.uniformReal(0, 24)
            pix_dim = geom2pix([x, y], size=self.size)
        state().setX(x)
        state().setY(y)
        state().setYaw(rng_.uniformReal(-pi, pi))
        return True
    
    def allocMyValidStateSampler(si):
        return MyValidStateSampler(si)
    if patch_map is not None and use_valid_sampler:
        si.setValidStateSamplerAllocator(ob.ValidStateSamplerAllocator(allocMyValidStateSampler))

    validityChecker = ob.StateValidityCheckerFn(partial(isStateValid, ss.getSpaceInformation()))
    ss.setStateValidityChecker(validityChecker)
    
    # Set the start and goal States:
    ss.setStartAndGoalStates(StartState, GoalState, 1.0)

    ode = oc.ODE(kinematicCarODE)
    odeSolver = oc.ODEBasicSolver(ss.getSpaceInformation(), ode)
    propagator = oc.ODESolver.getStatePropagator(odeSolver)
    ss.setStatePropagator(propagator)
    ss.getSpaceInformation().setPropagationStepSize(0.1)
    ss.getSpaceInformation().setMinMaxControlDuration(1, 50)

    planner = oc.SST(ss.getSpaceInformation())

    ss.setPlanner(planner)

    time = step_time
    solved = ss.solve(time)
    while not ss.haveExactSolutionPath():
        solved = ss.solve(step_time)
        time += step_time
        if time > max_time:
            break
    if ss.haveExactSolutionPath():
        success = True
        print("Found Solution")
        path = np.array([[ss.getSolutionPath().getState(i).getX(), ss.getSolutionPath().getState(i).getY(), ss.getSolutionPath().getState(i).getYaw()]
            for i in range(ss.getSolutionPath().getStateCount())
            ])
        path_quality = 0
        for i in range(len(path)-1):
            path_quality += np.linalg.norm(path[i+1, :2]-path[i, :2])
    else:
        success = False
        path_quality = np.inf
    return success, time, path_quality, path

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
    # parser.add_argument('--envNum', help='Environment number to validate model', required=True)
    parser.add_argument('--start')
    parser.add_argument('--use_sst')

    args = parser.parse_args()

    modelFolder = args.modelFolder
    modelFile = osp.join(modelFolder, f'model_params.json')
    assert osp.isfile(modelFile), f"Cannot find the model_params.json file in {modelFolder}"

    # env_num = args.envNum
    start = int(args.start)
    use_sst = args.use_sst
    if not use_sst:
        model_param = json.load(open(modelFile))
        transformer = Models.Transformer(
            **model_param
        )

        transformer.to(device)

        receptive_field=32
        # Load model parameters
        epoch = 69
        checkpoint = torch.load(osp.join(modelFolder, f'model_epoch_{epoch}.pkl'))
        transformer.load_state_dict(checkpoint['state_dict'])

        # Only do evaluation
        transformer.eval()
    # Get path data
    PathSuccess = []
    TimeSuccess = []
    QualitySuccess = []

    result_folder = '/data/results/car-mp-sst_vs' if not use_sst else '/data/results/car-sst'
    Path(result_folder).mkdir(parents=True, exist_ok=True)

    for env_num in range(start, start+10):
        temp_map =  f'/data/train2_car/env{env_num:06d}/map_{env_num}.png'
        small_map = skimage.io.imread(temp_map, as_gray=True)

        for pathNum in range(10):
            print(f"planning on env_{env_num} path_{pathNum}")
        # pathNum = 0
            pathFile = f'/data/train2_car/env{env_num:06d}/path_{pathNum}.p'
            data = pickle.load(open(pathFile, 'rb'))
            path = data['path_interpolated']

            if data['success']:
                if not use_sst:
                    goal_pos = geom2pix(path[0, :])
                    start_pos = geom2pix(path[-1, :])

                    # Identitfy Anchor points
                    encoder_input = get_encoder_input(small_map, goal_pos, start_pos)
                    # predVal = transformer(encoder_input[None,:].float().cuda())
                    with torch.no_grad():
                        predVal, embeddings = transformer(encoder_input[None,:].float().cuda())
                    predProb = F.softmax(predVal[0, :, :], dim=1)
                    predClass = (predProb[:, 1] / (predProb[:, 1]+predProb[:, 0]) > 0.4) # predClass = predVal[0, :, :].max(1)[1]
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
                    success, time, quality, planned_path = get_path(path[0, :], path[-1, :], small_map, patch_map, max_time=300, use_valid_sampler=True)

                else:
                    success, time, quality, planned_path = get_path(path[0, :], path[-1, :], small_map, patch_map=None, max_time=300)
                
                if success:
                    PathSuccess.append(planned_path)
                    TimeSuccess.append(time)
                    QualitySuccess.append(quality)

                    np.save(f'{result_folder}/PathSuccess_{start}.npy', PathSuccess)
                    np.save(f'{result_folder}/TimeSuccess_{start}.npy', TimeSuccess)
                    np.save(f'{result_folder}/QualitySuccess{start}.npy', QualitySuccess)
    # pickle.dump(PathSuccess, open(osp.join(modelFolder, f'eval_unknown_plan_{start:06d}.p'), 'wb'))
    print(len(PathSuccess))