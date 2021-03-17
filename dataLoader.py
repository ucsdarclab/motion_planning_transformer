''' Define a dataloader class.
'''

from os import path as osp
from torch.utils.data import IterableDataset
from skimage import io, color

import numpy as np
import pickle
import torch

rootFolder = '/root/data'
map_size = (480, 480)
patch_size = 32
stride = 8

def geom2pix(pos, res=0.05, size=(480, 480)):
    """
    Convert geometrical position to pixel co-ordinates. The origin is assumed to be 
    at [image_size[0]-1, 0].
    :param pos: The (x,y) geometric co-ordinates.
    :param res: The distance represented by each pixel.
    :param size: The size of the map image
    :returns (int, int): The associated pixel co-ordinates.
    """
    return (np.int(size[0]-1-np.floor(pos[1]/res)), np.int(np.floor(pos[0]/res)))


num_points = (map_size[0]-patch_size)//stride
discrete_points = np.linspace(0.05*(patch_size//2), 24-0.05*(patch_size//2), num_points)
grid_2d = np.meshgrid(discrete_points, discrete_points)
grid_points = np.reshape(np.array(grid_2d), (2, -1)).T
hash_table = [geom2pix(xy) for xy in grid_points]

class PathIterDataset(IterableDataset):
    ''' Iterable dataset.
    '''
    def __init__(self, envs, samples):
        '''
        :param env: list of map environments to load data from.
        :param samples: Number of samples to load from each map.
        '''
        assert isinstance(envs, list), "List of environments required"
        self.envs = envs
        self.samples = samples
        self.mapEnvg = {}
        # Load all maps
        for env in envs:
            mapLoc = osp.join('/root/data', 'env{}'.format(env), 'map_{}.png'.format(env))
            mapEnv = io.imread(mapLoc)
            self.mapEnvg[env] = color.rgb2gray(color.rgba2rgb(mapEnv))
        
    def __iter__(self):
        '''
        Return the data iterator.
        '''
        # For multiple cores
        worker_info = torch.utils.data.get_worker_info()
        if worker_info is not None:
            assert worker_info.num_workers == len(self.envs), "Each map requires 1 worker"
            worker_id = worker_info.id
            return iter(self.getItem(i, self.envs[worker_id]) for i in range(self.samples))
        
        return iter(self.getItem(i, self.envs[0]) for i in range(self.samples))
    
    def getItem(self, i, env):
        '''
        Get the item and unroll the environment.
        :param i: The index of the sample.
        :param env: Environment from which to read the data.
        '''
        dataFile = osp.join(rootFolder, 'env{}'.format(env), 'process', 'data_{}.p'.format(i))
        data = pickle.load(open(dataFile, 'rb'))
        if data['seq'] is not None:
            samples = len(data['seq'])
            obs = np.repeat(data['map'][None, :], samples-1, axis=0)
            inputs = np.zeros((samples-1, 1, patch_size, patch_size))
            targets = np.zeros((samples-1, 1))
            for j in range(samples-1):
                cur_index = hash_table[data['seq'][j]]
                start_x, start_y = cur_index[0]-patch_size//2, cur_index[1]-patch_size//2
                goal_x, goal_y = cur_index[0]+patch_size//2, cur_index[1]+patch_size//2
                inputs[j, 0,:, :] = self.mapEnvg[env][start_x:goal_x, start_y:goal_y]
                targets[j, 0] = data['seq'][j+1]
            return {
                'obs': obs,
                'inputs': inputs,
                'targets': targets
            }
