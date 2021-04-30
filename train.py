'''A script to train a simple model
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

from transformer import Models, Optim
from dataLoader import PathDataLoader, PaddedSequence
from utils import png_decoder, cls_decoder
import webdataset as wds

from torch.utils.tensorboard import SummaryWriter


def focal_loss(predVals, trueLabels, gamma, eps=1e-8):
    '''
    A function to calculate the focal loss as mentioned in 
    https://arxiv.org/pdf/1708.02002.pdf
    :param predVals: The output of the final linear layer.
    :param trueLabels: The true labels
    :param gamma: The hyperparameter of the loss function
    :param eps: A scalar value to enforce numerical stability.
    :returns float: The loss value
    '''
    input_soft = F.softmax(predVals, dim=1) + eps
    target_one_hot = torch.zeros((trueLabels.shape[0], 2), device=trueLabels.device)
    target_one_hot.scatter_(1, trueLabels.unsqueeze(1), 1.0)

    weight = torch.pow(-input_soft + 1., gamma)
    focal = -weight*torch.log(input_soft)
    loss = torch.sum(target_one_hot*focal, dim=1).sum()
    return loss

def cal_performance(predVals, anchorPoints, trueLabels, lengths):
    '''
    Return the loss and number of correct predictions.
    :param predVals: the output of the final linear layer.
    :param anchorPoints: The anchor points of interest
    :param trueLabels: The expected clas of the corresponding anchor points.
    :param lengths: The legths of each of sequence in the batch
    :returns (loss, n_correct): The loss of the model and number of avg predictions.
    '''
    n_correct = 0
    total_loss = 0
    num = 0
    for predVal, anchorPoint, trueLabel, length in zip(predVals, anchorPoints, trueLabels, lengths):
        predVal = predVal.index_select(0, anchorPoint[:length])
        # loss = F.cross_entropy(predVal, trueLabel[:length])
        loss = focal_loss(predVal, trueLabel[:length], gamma=2)
        total_loss += loss
        classPred = predVal.max(1)[1]
        n_correct +=classPred.eq(trueLabel[:length]).sum().item()/length
        num+=1
    total_loss = total_loss/num
    return total_loss, n_correct

def train_epoch(model, trainingData, optimizer, device):
    '''
    Train the model for 1-epoch with data from wds
    '''
    model.train()
    total_loss = 0
    total_n_correct = 0
    # Train for a single epoch.
    for batch in tqdm(trainingData, mininterval=2):
        
        optimizer.zero_grad()
        encoder_input = batch['map'].float().to(device)
        predVal = model(encoder_input)

        # Calculate the cross-entropy loss
        loss, n_correct = cal_performance(
            predVal, batch['anchor'].to(device), 
            batch['labels'].to(device), 
            batch['length'].to(device)
        )
        loss.backward()
        optimizer.step_and_update_lr()
        total_loss +=loss.item()
        total_n_correct += n_correct
    return total_loss, total_n_correct


def eval_epoch(model, validationData, device):
    '''
    Evaluation for a single epoch.
        :param model: The Transformer Model to be trained.
    :param validataionData: The set of validation data.
    :param device: cpu/cuda to be used.
    '''

    model.eval()
    total_loss = 0.0
    total_n_correct = 0.0
    with torch.no_grad():
        for batch in tqdm(validationData, mininterval=2):

            encoder_input = batch['map'].float().to(device)
            predVal = model(encoder_input)

            loss, n_correct = cal_performance(
                predVal, 
                batch['anchor'].to(device), 
                batch['labels'].to(device),
                batch['length'].to(device)
            )

            total_loss +=loss.item()
            total_n_correct += n_correct
    return total_loss, total_n_correct


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
        n_layers=6, 
        n_heads=3, 
        d_k=512, 
        d_v=256, 
        d_model=512, 
        d_inner=1024, 
        pad_idx=None,
        n_position=40*40, 
        dropout=0.1,
        train_shape=[24, 24],
    )
    
    transformer = Models.Transformer(**model_args)

    if torch.cuda.device_count() > 1:
        print("Using ", torch.cuda.device_count(), "GPUs")
        transformer = nn.DataParallel(transformer)
    transformer.to(device=device)

    # Define the optimizer
    # TODO: What does these parameters do ???
    optimizer = Optim.ScheduledOptim(
        optim.Adam(transformer.parameters(), betas=(0.9, 0.98), eps=1e-9),
        lr_mul = 0.5,
        d_model = 256,
        n_warmup_steps = 12800
    )

    # Training Data
    # shard_num = 0
    trainDataset = PathDataLoader(list(range(900)), samples=25, dataFolder='/root/data/maze/train')
    trainingData = DataLoader(trainDataset, num_workers=10, shuffle=True, batch_size=batch_size, collate_fn=PaddedSequence)

    # Validation Data
    valDataset = PathDataLoader(list(range(900, 1000)), samples=25, dataFolder='/root/data/maze/val')
    validationData = DataLoader(valDataset, num_workers=5, shuffle=True, batch_size=batch_size, collate_fn=PaddedSequence)

    # Increase number of epochs.
    n_epochs = 50
    results = {}
    train_loss = []
    val_loss = []
    train_n_correct_list = []
    val_n_correct_list = []
    trainDataFolder  = '/root/data/model15'
    # Save the model parameters as .json file
    json.dump(
        model_args, 
        open(osp.join(trainDataFolder, 'model_params.json'), 'w'),
        sort_keys=True,
        indent=4
    )
    writer = SummaryWriter(log_dir=trainDataFolder)
    for n in range(n_epochs):
        train_total_loss, train_n_correct = train_epoch(transformer, trainingData, optimizer, device)
        val_total_loss, val_n_correct = eval_epoch(transformer, validationData, device)
        print(f"Epoch {n} Loss: {train_total_loss}")
        print(f"Epoch {n} Loss: {val_total_loss}")
        print(f"Epoch {n} Accuracy {val_n_correct/len(valDataset)}")

        # Log data.
        train_loss.append(train_total_loss)
        val_loss.append(val_total_loss)
        train_n_correct_list.append(train_n_correct)
        val_n_correct_list.append(val_n_correct)

        if (n+1)%5==0:
            if isinstance(transformer, nn.DataParallel):
                state_dict = transformer.module.state_dict()
            else:
                state_dict = transformer.state_dict()
            states = {
                'state_dict': state_dict,
                'optimizer': optimizer._optimizer.state_dict(),
                'torch_seed': torch_seed
            }
            torch.save(states, osp.join(trainDataFolder, 'model_epoch_{}.pkl'.format(n)))
        
        pickle.dump(
            {
                'trainLoss': train_loss, 
                'valLoss':val_loss, 
                'trainNCorrect':train_n_correct_list, 
                'valNCorrect':val_n_correct_list
            }, 
            open(osp.join(trainDataFolder, 'progress.pkl'), 'wb')
            )
        writer.add_scalar('Loss/train', train_total_loss, n)
        writer.add_scalar('Loss/test', val_total_loss, n)
        writer.add_scalar('Accuracy/train', train_n_correct/len(trainDataset), n)
        writer.add_scalar('Accuracy/test', val_n_correct/len(valDataset), n)