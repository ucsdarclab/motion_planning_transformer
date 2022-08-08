'''A script for generating patches
'''
import skimage.io
import skimage.morphology as skim
import numpy as np
import pickle

from os import path as osp
import argparse

from eval_model import get_path

if __name__=="__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        '--plannerType', 
        help='The underlying sampler to use', 
        required=True, 
        choices=['rrtstar', 'informedrrtstar']
    )
    parser.add_argument('--valDataFolder', help='Directory where training data exists', required=True)
    parser.add_argument('--start', help='Start of environment number', required=True, type=int)
    parser.add_argument('--samples', help='Number of envs', required=True, type=int)
    parser.add_argument('--numPaths', help='Number of start and goal pairs for each env', default=1, type=int)

    args = parser.parse_args()

    start = args.start
    samples = args.samples

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

            if data['success']:
                cost = np.linalg.norm(np.diff(path, axis=0), axis=1).sum()
                _, t, v, s = get_path(path[0, :], path[-1, :], small_map, None, args.plannerType, cost)
                pathSuccess.append(s)
                pathTime.append(t)
                pathVertices.append(v)
            else:
                pathSuccess.append(False)
                pathTime.append(0)
                pathVertices.append(0)

    pathData = {'Time':pathTime, 'Success':pathSuccess, 'Vertices':pathVertices}
    pickle.dump(pathData, open(osp.join(valDataFolder, f'eval_val_plan_{args.plannerType}_{start:06d}.p'), 'wb'))
    print(sum(pathSuccess))