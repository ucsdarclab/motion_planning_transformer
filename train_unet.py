'''A script to train a simple UNet model
'''

import numpy as np
import pickle

import torch
import torch.optim as optim

import io
import skimage.io

from torchvision.transforms import Normalize
from torch.utils.data import DataLoader
import torch.nn as nn
import torch.nn.functional as F
from tqdm import tqdm
from os import path as osp

import torch.optim as optim

from unet import Models
from dataLoader import PathPatchDataLoader, PaddedSequenceUnet, PathMixedDataLoader
from utils import png_decoder, cls_decoder
import webdataset as wds
from einops import rearrange

from torch.utils.tensorboard import SummaryWriter

def dice_coeff(predMask, target, reduce_batch_first=False, epsilon=1e-6):
    # Average of Dice coefficient for all batches, or for a single mask
    assert predMask.size() == target.size()
    if predMask.dim() == 2 and reduce_batch_first:
        raise ValueError(f'Dice: asked to reduce batch but got tensor without batch dimension (shape {predMask.shape})')

    if predMask.dim() == 2 or reduce_batch_first:
        inter = torch.dot(predMask.flatten(), target.flatten())
        sets_sum = torch.sum(predMask) + torch.sum(target)
        if sets_sum.item() == 0:
            sets_sum = 2 * inter

        return (2 * inter + epsilon) / (sets_sum + epsilon)
    else:
        # compute and average metric for each batch element
        dice = 0
        for i in range(predMask.shape[0]):
            dice += dice_coeff(predMask[i, ...], target[i, ...])
        return dice / predMask.shape[0]

def dice_loss(predMask, target):
    # Dice loss (objective to minimize) between 0 and 1
    assert predMask.size() == target.size()
    return 1 - dice_coeff(predMask, target, reduce_batch_first=True)

def cal_performance_unet(predVals, trueLabels):
    '''
    Calculate the error between the true labels and predicted values.
    :param predVals: The output of the network.
    :param trueLabels: The true label of the pixel.
    returns loss: the loss of the given data.
    '''
    dl = dice_loss(F.softmax(predVals, dim=1).float(), rearrange(F.one_hot(trueLabels, num_classes=2), 'b h w c-> b c h w').float())
    loss = F.cross_entropy(predVals, trueLabels) + dl
    return loss, dl


def train_epoch(model, trainingData, optimizer, scheduler, device):
    '''
    Train the model for 1-epoch with data from wds
    '''
    model.train()
    total_loss = 0
    total_overlap = 0
    # Train for a single epoch.
    for batch in tqdm(trainingData, mininterval=2):        
        optimizer.zero_grad()
        encoder_input = batch['map'].float().to(device)
        predVal = model(encoder_input)

        # Calculate the cross-entropy loss
        loss, dl = cal_performance_unet(
            predVal, batch['mask'].to(device)
        ) 
        loss.backward()
        optimizer.step()
        total_loss +=loss.item()
        total_overlap += dl.item()
    scheduler.step()
    return total_loss, total_overlap


def eval_epoch(model, validationData, device):
    '''
    Evaluation for a single epoch.
        :param model: The Transformer Model to be trained.
    :param validataionData: The set of validation data.
    :param device: cpu/cuda to be used.
    '''

    model.eval()
    total_loss = 0.0
    total_overlap = 0.0
    with torch.no_grad():
        for batch in tqdm(validationData, mininterval=2):

            encoder_input = batch['map'].float().to(device)
            predVal = model(encoder_input)

            loss, dl = cal_performance_unet(
                predVal, 
                batch['mask'].to(device), 
            )

            total_loss +=loss.item()
            total_overlap += (1-dl)
    return total_loss, total_overlap


import sys
import json

if __name__ == "__main__":
    batch_size = int(sys.argv[1])
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
        n_channels=2,
        n_classes=2
    )
    
    unet = Models.UNet(**model_args)
    if torch.cuda.device_count() > 1:
        print("Using ", torch.cuda.device_count(), "GPUs")
        unet = nn.DataParallel(unet)
    unet.to(device=device)

    # Define the optimizer
    optimizer = optim.Adam(filter(lambda p: p.requires_grad, unet.parameters()), lr=1e-4)
    exp_lr_scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=30, gamma=0.1)

    # Training with Mixed samples
    trainDataset = PathPatchDataLoader(
        env_list=list(range(1750)),
        dataFolder='/root/data/forest/train'
    )
    trainingData = DataLoader(trainDataset, num_workers=15, collate_fn=PaddedSequenceUnet, batch_size=batch_size)

    # Validation Data
    valDataset = PathPatchDataLoader(
        env_list=list(range(1000)), 
        dataFolder='/root/data/forest/val'
    )
    validationData = DataLoader(valDataset, num_workers=5, collate_fn=PaddedSequenceUnet, batch_size=batch_size)

    # Increase number of epochs.
    n_epochs = 70
    results = {}
    train_loss = []
    val_loss = []
    train_n_overlap_list = []
    val_n_overlap_list = []
    trainDataFolder  = '/root/data/unet/model2'
    # Save the model parameters as .json file
    json.dump(
        model_args, 
        open(osp.join(trainDataFolder, 'model_params.json'), 'w'),
        sort_keys=True,
        indent=4
    )
    writer = SummaryWriter(log_dir=trainDataFolder)
    for n in range(n_epochs):
        train_total_loss, train_n_overlap = train_epoch(unet, trainingData, optimizer, exp_lr_scheduler, device)
        val_total_loss, val_n_overlap = eval_epoch(unet, validationData, device)
        print(f"Epoch {n} Loss: {train_total_loss}")
        print(f"Epoch {n} Loss: {val_total_loss}")
        print(f"Epoch {n} Overlap {val_n_overlap/len(valDataset)}")

        # Log data.
        train_loss.append(train_total_loss)
        val_loss.append(val_total_loss)
        train_n_overlap_list.append(train_n_overlap)
        val_n_overlap_list.append(val_n_overlap)

        if (n+1)%5==0:
            if isinstance(unet, nn.DataParallel):
                state_dict = unet.module.state_dict()
            else:
                state_dict = unet.state_dict()
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
                'trainNOverlap':train_n_overlap_list, 
                'valOverlap':val_n_overlap_list
            }, 
            open(osp.join(trainDataFolder, 'progress.pkl'), 'wb')
            )
        writer.add_scalar('Loss/train', train_total_loss, n)
        writer.add_scalar('Loss/test', val_total_loss, n)
        writer.add_scalar('Overlap/train', train_n_overlap/len(trainDataset), n)
        writer.add_scalar('Overlap/test', val_n_overlap/len(valDataset), n)