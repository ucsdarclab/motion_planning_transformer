'''A script for export a TorchScript model.
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
#import matplotlib.plt as pt
#import matplotlib.pyplot as plt

try:
    from ompl import base as ob
    from ompl import geometric as og
    from ompl import util as ou
except ImportError:
    raise ImportError("Container does not have OMPL installed")

from transformer import Models as tfModel
from unet import Models as unetModel
from utils import geom2pix, ValidityChecker
from dataLoader import get_encoder_input
from eval_model import get_patch

parser = argparse.ArgumentParser()
parser.add_argument('--modelFolder', help='Directory where model_params.json exists', required=True)

args = parser.parse_args()

modelFolder = args.modelFolder
modelFile = osp.join(modelFolder, f'model_params.json')
assert osp.isfile(modelFile), f"Cannot find the model_params.json file in {modelFolder}"

model_param = json.load(open(modelFile))

model = tfModel.Transformer(
            **model_param
        )

checkpoint = torch.load(osp.join('/workspace/motion_planning_transformer/final_models/point_robot',
                                 f'model_weights.pkl'))
model.load_state_dict(checkpoint['state_dict'])
'''
valDataFolder = '/home/udayk/rsch/motion_planning_transformer/test_script/example_app/build/maze4/val'
pathFile = osp.join(valDataFolder, f'env{0:06d}/path_{0}.p')
data = pickle.load(open(pathFile, 'rb'))


temp_map =  osp.join(valDataFolder, f'env{0:06d}/map_{0}.png')
small_map = skimage.io.imread(temp_map, as_gray=True)
mapSize = small_map.shape

path = data['path_interpolated']

goal_pos = geom2pix(path[0, :], size=(480, 480))
start_pos = geom2pix(path[-1, :], size=(480, 480))

patch_map, patchTime = get_patch(model, start_pos, goal_pos, small_map)
'''
#plt.imshow(patch_map)

#input = torch.rand((1,2,480,480))
#print(model(input))

sm = torch.jit.script(model)
sm.save("saved_model.pt")    