''' Generate the all the shards for the dataset.
'''

import torch
import numpy as np
import skimage.io
import pickle

from os import path as osp
from skimage import color

import sys
from tqdm import tqdm

import webdataset as wds

from utils import geom2pix

map_size = (480, 480)
patch_size = 32
stride = 8

num_points = (map_size[0]-patch_size)//stride + 1
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


if __name__ == "__main__":
    shard_num, start, samples = int(sys.argv[1]), int(sys.argv[2]), int(sys.argv[3])
    rootFolder = '/root/data/'
    dataType = 'train'

    sink = wds.TarWriter(osp.join(rootFolder, dataType, f'{dataType}_{shard_num:04d}.tar'))
    # Form the paths as shards from each dataset.
    count = 0
    for env in [1, 2, 3, 4, 5]:
        mapLoc = osp.join(rootFolder, f'env{env}', f'map_{env}.png')
        mapEnvg = skimage.io.imread(mapLoc, as_gray=True)
        for i in tqdm(range(start, start+samples)):
            dataFile = osp.join(rootFolder, f'env{env}', f'path_{i}.p')
            dataFile = pickle.load(open(dataFile, 'rb'))
            path = dataFile['path_interpolated']

            if dataFile['success']:
                goal_map = np.zeros(mapEnvg.shape)
                _, goal_index = geom2pixMat(path[-1, :])
                
                goal_start_x = max(0, goal_index[0]- patch_size//2)
                goal_start_y = max(0, goal_index[1]- patch_size//2)
                goal_end_x = min( map_size[0], goal_index[0]+ patch_size//2)
                goal_end_y = min( map_size[1], goal_index[1]+ patch_size//2)
                goal_map[goal_start_x:goal_end_x, goal_start_y:goal_end_y] = 1.0
                
                output_seq_interpolate = [geom2pixMat(xy)[0] for xy in path]
                # Select unique sequences
                output_seq = []
                for seq in output_seq_interpolate:
                    if seq not in output_seq:
                        output_seq.append(seq)
                output_seq.append(num_points**2)

                for j, seq in enumerate(output_seq[:-1]):
                    cur_index = hash_table[seq]
                    start_x, start_y = cur_index[0]-patch_size//2, cur_index[1]-patch_size//2
                    goal_x, goal_y = cur_index[0]+patch_size//2, cur_index[1]+patch_size//2
                    inputs = mapEnvg[start_x:goal_x, start_y:goal_y]
                    target = output_seq[j+1]
                    sink.write(
                        {
                            '__key__':f'samples_{count:08d}',
                            'goal_map.png': goal_map,
                            'input_patch.png': inputs,
                            'cur_index.cls':seq,
                            'map.cls': env,
                            'target.cls':target
                        }
                    )
                    count+=1

    sink.close()