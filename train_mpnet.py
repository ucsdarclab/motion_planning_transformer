'''A script to train an MPNet model
'''

import numpy as np
import pickle

import torch
import torch.optim as optim

import json
import argparse

from torch.utils.data import DataLoader
import torch.nn as nn
import torch.nn.functional as F
from tqdm import tqdm
from os import path as osp

from mpnet import Models
from dataLoader import PathSeqDataLoader, PaddedSequenceMPnet

from torch.utils.tensorboard import SummaryWriter

# TODO: Write the code for performing training for one epoch
def train_epoch(model, trainingData, optimizer, device):
    '''
    Train the model for 1-epoch with the given training data.
    '''
    model.train()
    total_loss = 0
    # Train for a single epoch.
    for batch in tqdm(trainingData, mininterval=2):
        optimizer.zero_grad()
        encoder_input = batch['map'].to(device)
        encoder_val = model.get_environment_encoding(encoder_input)
        loss = 0
        for i, l in enumerate(batch['length']):
            nInputs = torch.hstack([encoder_val[i, :].repeat(l, 1), batch['inputs'][i, :l, :].to(device)])
            predTarget = mpnet(nInputs)
            loss += F.mse_loss(predTarget, batch['targets'][i, :l, :].to(device))
        loss.backward()
        optimizer.step()
        total_loss +=loss.item()
    return total_loss

# TODO: Write the code for performing evalution of validation data.
def eval_epoch(model, validationData, device):
    '''
    Evaluation for a single epoch.
    :param model: The MPNet Model to be trained.
    :param validataionData: The set of validation data.
    :param device: cpu/cuda to be used.
    '''

    total_loss = 0.0
    total_n_correct = 0.0
    with torch.no_grad():
        for batch in tqdm(validationData, mininterval=2):
            encoder_input = batch['map'].to(device)
            encoder_val = model.get_environment_encoding(encoder_input)
            loss = 0
            for i, l in enumerate(batch['length']):
                nInputs = torch.hstack([encoder_val[i, :].repeat(l, 1), batch['inputs'][i, :l, :].to(device)])
                predTarget = mpnet(nInputs)
                loss += F.mse_loss(predTarget, batch['targets'][i, :l, :].to(device))
            total_loss +=loss.item()
    return total_loss


def check_data_folders(folder):
    '''
    Checks if the folder is formatted properly for training.
    The folder need to have a 'train' and 'val' folder
    :param folder: The folder to test
    '''
    assert osp.isdir(osp.join(folder, 'train')), "Cannot find trainining data"
    assert osp.isdir(osp.join(folder, 'val')), "Cannot find validation data"

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--batchSize', help="Batch size per GPU", required=True, type=int)
    parser.add_argument('--mazeDir', help="Directory with training and validation data for Maze", default=None)
    parser.add_argument('--forestDir', help="Directory with training and validation data for Random Forest", default=None)
    parser.add_argument('--fileDir', help="Directory to save training Data")
    args = parser.parse_args()

    maze=False
    if args.mazeDir is not None:
        check_data_folders(args.mazeDir)
        maze=True
    forest=False
    if args.forestDir is not None:
        check_data_folders(args.forestDir)
        forest=True

    assert forest or maze, "Need to provide data folder for atleast one kind of environment"
    dataFolder = args.mazeDir if not(maze and forest) and maze else args.forestDir

    print(f"Using data from {dataFolder}")

    batch_size = args.batchSize
    device = 'cpu'
    if torch.cuda.is_available():
        print("Using GPU....")
        device = torch.device('cuda')

    if torch.cuda.device_count() > 1:
        batch_size = batch_size * torch.cuda.device_count()
    print(f"Total batch size : {batch_size}")

    torch_seed = np.random.randint(low=0, high=1000)
    torch.manual_seed(torch_seed)
    
    model_args = dict(
        AE_input_size=[1, 480, 480],
        state_size=2
    )
    
    mpnet = Models.MPNet(**model_args)

    if torch.cuda.device_count() > 1:
        print("Using ", torch.cuda.device_count(), "GPUs")
        mpnet = nn.DataParallel(mpnet)
    mpnet.to(device=device)

    # Define the optimizer    
    optimizer = optim.Adagrad(mpnet.parameters(), lr=1e-4)

    # TODO: Update the dat loader for MPNet format
    # # Training with Mixed samples
    # if maze and forest:
    #     from toolz.itertoolz import partition
    #     trainDataset= PathMixedDataLoader(
    #         envListForest=list(range(10)),
    #         dataFolderForest=osp.join(args.forestDir, 'train'),
    #         envListMaze=list(range(10)),
    #         dataFolderMaze=osp.join(args.mazeDir, 'train')
    #     )
    #     allTrainingData = trainDataset.indexDictForest + trainDataset.indexDictMaze
    #     batch_sampler_train = list(partition(batch_size, allTrainingData))
    #     trainingData = DataLoader(trainDataset, num_workers=15, batch_sampler=batch_sampler_train, collate_fn=PaddedSequence)

    #     valDataset = PathMixedDataLoader(
    #         envListForest=list(range(10)),
    #         dataFolderForest=osp.join(args.forestDir, 'val'),
    #         envListMaze=list(range(10)),
    #         dataFolderMaze=osp.join(args.mazeDir, 'val')
    #     )
    #     allValData = valDataset.indexDictForest+valDataset.indexDictMaze
    #     batch_sampler_val = list(partition(batch_size, allValData))
    #     validationData = DataLoader(valDataset, num_workers=5, batch_sampler=batch_sampler_val, collate_fn=PaddedSequence)
    # else:        
    trainDataset = PathSeqDataLoader(
        env_list=list(range(1750)),
        dataFolder=osp.join(dataFolder, 'train'),
        worldMapBounds=[480*0.05, 480*0.05]
    )
    trainingData = DataLoader(trainDataset, num_workers=15, collate_fn=PaddedSequenceMPnet, batch_size=batch_size)

    # Validation Data
    valDataset = PathSeqDataLoader(
        env_list=list(range(2500)),
        dataFolder=osp.join(dataFolder, 'val'),
        worldMapBounds=[480*0.05, 480*0.05]
    )
    validationData = DataLoader(valDataset, num_workers=5, collate_fn=PaddedSequenceMPnet, batch_size=batch_size)

    # Increase number of epochs.
    n_epochs = 70
    results = {}
    train_loss = []
    val_loss = []
    trainDataFolder  = args.fileDir
    # Save the model parameters as .json file
    json.dump(
        model_args, 
        open(osp.join(trainDataFolder, 'model_params.json'), 'w'),
        sort_keys=True,
        indent=4
    )
    writer = SummaryWriter(log_dir=trainDataFolder)
    for n in range(n_epochs):
        train_total_loss = train_epoch(mpnet, trainingData, optimizer, device)
        val_total_loss = eval_epoch(mpnet, validationData, device)
        print(f"Epoch {n} Loss: {train_total_loss}")
        print(f"Epoch {n} Loss: {val_total_loss}")

        # Log data.
        train_loss.append(train_total_loss)
        val_loss.append(val_total_loss)

        if (n+1)%5==0:
            if isinstance(mpnet , nn.DataParallel):
                state_dict = mpnet.module.state_dict()
            else:
                state_dict = mpnet.state_dict()
            states = {
                'state_dict': state_dict,
                'optimizer': optimizer.state_dict(),
                'torch_seed': torch_seed
            }
            torch.save(states, osp.join(trainDataFolder, 'model_epoch_{}.pkl'.format(n)))
        
        pickle.dump(
            {
                'trainLoss': train_loss, 
                'valLoss':val_loss, 
            }, 
            open(osp.join(trainDataFolder, 'progress.pkl'), 'wb')
            )
        writer.add_scalar('Loss/train', train_total_loss, n)
        writer.add_scalar('Loss/test', val_total_loss, n)