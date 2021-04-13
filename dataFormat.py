''' Extract classes and map of the environment.
'''

from os import path as osp
from skimage import color, io
from torch.utils.data import Dataset
import pickle
import numpy as np

from utils import geom2pix

map_size = (480, 480)
patch_size = 32
stride = 8

num_points = (map_size[0]-patch_size)//stride
discrete_points = np.linspace(0.05*(patch_size//2), 24-0.05*(patch_size//2), num_points)
grid_2d = np.meshgrid(discrete_points, discrete_points)
grid_points = np.reshape(np.array(grid_2d), (2, -1)).T
hash_table = [geom2pix(xy)[::-1] for xy in grid_points]


def geom2pixMat(pos, res=0.05, size=(480, 480)):
    """
    Find the nearest index of the discrete map state.
    :param pos: The (x,y) geometric co-ordinates.
    :param res: The distance represented by each pixel.
    :param size: The size of the map image
    :returns (int, int): The associated pixel co-ordinates.
    """
    index = np.argmin(np.linalg.norm(grid_points-pos, axis=1))
    return index, hash_table[index]


class PathDataLoader(Dataset):
    '''Loads the dataset for each path, and converts to class labels.
    '''
    def __init__(self, env_list, samples, dataFolder):
        '''
        :param env_list: The list of map environments.
        :param samples: Number of samples for each env.
        :param dataFolder: Folder location
        '''
        assert isinstance(env_list, list), "Needs to be a list"
        self.total_numbers = len(env_list)*samples
        self.samples = samples
        self.env_list = env_list
        self.dataFolder = dataFolder        
        
    def __len__(self):
        return self.total_numbers
    
    def __getitem__(self, idx):
        map_index = idx//self.samples
        env_num = self.env_list[map_index]
        folderLoc = osp.join(self.dataFolder, 'env{}'.format(env_num))
        fileLoc = osp.join(folderLoc, 'map_{}.png'.format(env_num))
        mapEnv = io.imread(fileLoc)
        mapEnvg = color.rgb2gray(color.rgba2rgb(mapEnv))
        # Load trajectory
        path_num = idx%self.samples
        fileLoc2 = osp.join(folderLoc, 'path_{}.p'.format(path_num))
        dataFile = pickle.load(open(fileLoc2, 'rb'))
        path = dataFile['path_interpolated']
        if dataFile['success']:
            # Pad map:
            goal_map = np.zeros(mapEnvg.shape)
            _, goal_index = geom2pixMat(path[-1, :])
            goal_start_x = max(0, goal_index[0]- patch_size//2)
            goal_start_y = max(0, goal_index[1]- patch_size//2)
            goal_end_x = min( map_size[0], goal_index[0]+ patch_size//2)
            goal_end_y = min( map_size[1], goal_index[1]+ patch_size//2)
            goal_map[goal_start_x:goal_end_x, goal_start_y:goal_end_y] = 1.0
            input_map = np.concatenate((mapEnvg[None, :], goal_map[None,:]))
            # output_seq = [geom2pixMat(xy)[0] for xy in path]
            # Extract unique ID's
            output_seq_interpolate = [geom2pixMat(xy)[0] for xy in path]
            output_seq = []
            for pi_i in output_seq_interpolate:
                if pi_i not in output_seq:
                    output_seq.append(pi_i)

            # Terminate the sequence with the goal class.
            output_seq.append(num_points**2)
            return {'map':input_map, 'seq':output_seq}
        return {'map':None, 'seq':None}

import sys
from tqdm import tqdm
if __name__ == "__main__":
    env = int(sys.argv[1])
    envs = [env]
    rootFolder = '/root/data'
    customDataset = PathDataLoader(envs, 10000, rootFolder)
    for i in tqdm(range(10000)):
        with open(osp.join(rootFolder, 'env{}'.format(envs[0]), 'process', 'data_{}.p'.format(i)), 'wb') as f:
            pickle.dump(customDataset[i], f)