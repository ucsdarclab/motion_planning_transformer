'''A script for generating patches
'''

import torch
import torch.nn as nn
import torch.nn.functional as F
import skimage.io
import skimage.morphology as skim
import numpy as np
from os import path as osp

from transformer import Models


res = 0.05
length = 24

def geom2pix(pos, res=0.05, size=(480, 480)):
    """
    Convert geometrical position to pixel co-ordinates. The origin is assumed to be 
    at [image_size[0]-1, 0].
    :param pos: The (x,y) geometric co-ordinates.
    :param res: The distance represented by each pixel.
    :param size: The size of the map image
    :returns (int, int): The associated pixel co-ordinates.
    """
    return (np.int(np.floor(pos[0]/res)), np.int(size[0]-1-np.floor(pos[1]/res)))

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
hashTable = [(20*r+15, 20*c+15) for c in range(23) for r in range(23)]

device='cuda' if torch.cuda.is_available() else 'cpu'

if __name__=="__main__":
    transformer = Models.Transformer(
    n_layers=2, 
    n_heads=3, 
    d_k=512, 
    d_v=256, 
    d_model=512, 
    d_inner=1024, 
    pad_idx=None,
    n_position=40*40,
    train_shape=[23, 23], # NOTE: This is hard coded value.
    dropout=0.1
    )

    # Load model parameters
    epoch = 149
    modelFolder = '/root/data/model6/'
    checkpoint = torch.load(osp.join(modelFolder, f'model_epoch_{epoch}.pkl'))
    transformer.load_state_dict(checkpoint['state_dict'])

    env_num=5
    temp_map =  f'/root/data/env{env_num}/map_{env_num}.png'
    small_map = skimage.io.imread(temp_map, as_gray=True)

    # Define the start/goal pos
    start_pos = (176, 120)
    goal_pos = (200, 400)


    # Identitfy Anchor points
    encoder_input = get_encoder_input(small_map, goal_pos, start_pos)
    predVal = transformer(encoder_input[None,:].float())
    predClass = predVal[0, :, :].max(1)[1]

    predProb = F.softmax(predVal[0, :, :], dim=1)
    possAnchor = [hashTable[i] for i, label in enumerate(predClass) if label==1]

    # Generate Patch Maps
    patch_map = np.zeros_like(small_map)
    receptive_field=32
    map_size = small_map.shape
    for pos in possAnchor:
        goal_start_x = max(0, pos[0]- receptive_field//2)
        goal_start_y = max(0, pos[1]- receptive_field//2)
        goal_end_x = min(map_size[0], pos[0]+ receptive_field//2)
        goal_end_y = min(map_size[1], pos[1]+ receptive_field//2)
        patch_map[goal_start_y:goal_end_y, goal_start_x:goal_end_x] = 1.0
    
