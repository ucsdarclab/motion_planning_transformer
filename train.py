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
import torch.nn.functional as F
from tqdm import tqdm
from os import path as osp

from transformer import Models, Optim
from dataLoader import PathIterDataset
from utils import png_decoder, cls_decoder
import webdataset as wds

from torch.utils.tensorboard import SummaryWriter

def cost_function(predVal, trueLabel, smoothing=False):
    '''
    Calculate the cost between predicted output and true labels
    :param predVal: The output of the linear model function.
    :param trueLabel: The true class of the data.
    :param smoothing: Apply smoothing to the data.
    '''
    if smoothing:
        eps = 0.1
        n_class = predVal.size(1)

        one_hot = torch.zeros_like(predVal).scatter(1, trueLabel.view(-1, 1), 1)
        one_hot = one_hot*(1-eps) + (1-one_hot)*eps/(n_class-1)
        log_prob = F.log_softmax(predVal, dim=1)
        return -(one_hot*log_prob).sum(dim=1).mean()
    return F.cross_entropy(predVal, trueLabel)


def cal_performance(predVal, trueLabel, smoothing=False):
    '''
    Return the loss and number of correct predictions.
    :param predVal: the output of the final linear layer.
    :param trueLabel: The expected class.
    :param smoothing: If True, do calculate smooth label loss.
    :returns (loss, n_correct): The loss of the model and number of correct predictions.
    '''
    loss = cost_function(predVal, trueLabel, smoothing)

    predVal = predVal.max(1)[1]
    n_correct = predVal.eq(trueLabel).sum().item()
    return loss, n_correct


batch_size = 256
gs_norm = Normalize([0]*batch_size, [255.0]*batch_size)
train_num_samples = 2e5
train_num_batches = int(train_num_samples//batch_size)

from itertools import islice

maps = [skimage.io.imread(f'/root/data/env{env}/map_{env}.png', as_gray=True) for env in [1, 2, 3, 4, 5]]

def train_wds_epoch(model, trainingData, optimizer, device):
    '''
    Train the model for 1-epoch with data from wds
    '''
    model.train()
    total_loss = 0
    total_n_correct = 0
    # Train for a single epoch.
    for batch in tqdm(islice(trainingData, train_num_batches), mininterval=2, total=train_num_batches):
        obs_goal, map_list, input_patches, targets = batch
        # Normalize inputs
        obs_goal = gs_norm(obs_goal.float())
        input_patches = gs_norm(input_patches.float())

        optimizer.zero_grad()
        # When map has to be stacked.
        batch_maps  = torch.zeros_like(obs_goal)
        for i in range(targets.shape[0]):
            batch_maps[i, :, :] = torch.as_tensor(maps[map_list[i]-1])

        batch_maps = torch.cat([batch_maps[:, None, :], obs_goal[:, None, :]], dim=1)
        batch_maps = batch_maps.to(device)
        batch_inputs = (input_patches[:, None, :]).to(device)
        pred = model(batch_maps, batch_inputs)
        batch_targets = targets.long().to(device)

        # Calculate the cross-entropy loss
        loss, n_correct = cal_performance(pred, batch_targets, smoothing=True)

        loss.backward()
        optimizer.step_and_update_lr()
        total_loss +=loss.item()
        total_n_correct += n_correct
    return total_loss, total_n_correct

val_num_samples = 4000
val_num_batches = int(val_num_samples//batch_size)

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
        for batch in tqdm(islice(validationData, val_num_batches), mininterval=2, total=val_num_batches):
            obs_goal, map_list, input_patches, targets = batch
            
            obs_goal = gs_norm(obs_goal.float())
            input_patches = gs_norm(input_patches.float())

            # Stack maps.
            batch_maps  = torch.zeros_like(obs_goal)
            for i in range(targets.shape[0]):
                batch_maps[i, :, :] = torch.as_tensor(maps[map_list[i]-1])
            batch_maps = torch.cat([batch_maps[:, None, :], obs_goal[:, None, :]], dim=1)
            batch_maps = batch_maps.to(device)

            batch_inputs = (input_patches[:, None, :]).to(device)
            pred = model(batch_maps, batch_inputs)

            batch_targets = targets.long().to(device)
            # NOTE : Need not do label smoothing for evaluation
            loss, n_correct = cal_performance(pred, batch_targets, smoothing=False)

            total_loss +=loss.item()
            total_n_correct += n_correct
    return total_loss, total_n_correct


def check_data():
    '''
    Check if all data is properly formatted.
    '''
    shard_num=0
    shard_file = f'/root/data/train/train_{shard_num:04d}.tar'
    dataset = wds.WebDataset(shard_file).decode(
        png_decoder, 
        cls_decoder)

    for data_i in tqdm(dataset, total=200):
        if 'target.cls' not in data_i.keys():
            print(data_i['__key__'])


# if __name__ == "__main__":
#     check_data()

# if False:
if __name__ == "__main__":
    device = 'cpu'
    if torch.cuda.is_available():
        print("Using GPU....")
        device = torch.device('cuda')

    torch_seed = np.random.randint(low=0, high=1000)
    torch.manual_seed(torch_seed)

    map_size = (480, 480)
    patch_size = 32
    stride = 8
    num_points = (map_size[0]-patch_size)//stride + 1
    transformer = Models.Transformer(
        map_res=0.05,
        map_size=map_size,
        patch_size=patch_size,
        n_layers=2,
        n_heads=3,
        d_k=64,
        d_v=64,
        d_model=256,
        d_inner=1024,
        pad_idx=None,
        dropout=0.1,
        n_classes=(num_points)**2 
    ).to(device=device)

    # Define the optimizer
    # TODO: What does these parameters do ???
    optimizer = Optim.ScheduledOptim(
        optim.Adam(transformer.parameters(), betas=(0.9, 0.98), eps=1e-9),
        lr_mul = 0.5,
        d_model = 256,
        n_warmup_steps = 4000
    )

    # Training Data
    # shard_num = 0
    total_shards=50
    # Select random shards
    shard_list = np.random.choice(range(100), size=total_shards, replace=False)
    shard_files = [f'/root/data/train2/train2_{shard_num:04d}.tar' for shard_num in shard_list]
    dataset = wds.Dataset(shard_files).decode(
        png_decoder, 
        cls_decoder).shuffle(100).to_tuple('goal_map.png', 'map.cls', 'input_patch.png', 'target.cls') 

    trainingData = DataLoader(dataset, num_workers=25, batch_size=batch_size)

    # Validation Data
    val_shard_files = [f'/root/data/val/val/val_{shard_num:04d}.tar' for shard_num in range(1)]
    valDataset = wds.Dataset(val_shard_files).decode(
        png_decoder,
        cls_decoder).shuffle(100).to_tuple('goal_map.png', 'map.cls', 'input_patch.png', 'target.cls')

    validationData = DataLoader(valDataset, num_workers=1, batch_size=batch_size)

    # Increase number of epochs.
    n_epochs = 100
    results = {}
    train_loss = []
    val_loss = []
    train_n_correct_list = []
    val_n_correct_list = []
    trainDataFolder  = '/root/data/model3'
    writer = SummaryWriter(log_dir=trainDataFolder)
    for n in range(n_epochs):
        train_total_loss, train_n_correct = train_wds_epoch(transformer, trainingData, optimizer, device)
        val_total_loss, val_n_correct = eval_epoch(transformer, validationData, device)
        print(f"Epoch {n} Loss: {train_total_loss}")
        print(f"Epoch {n} Loss: {val_total_loss}")
        print(f"Epoch {n} Accuracy {val_n_correct/(batch_size*val_num_batches)}")

        # Log data.
        train_loss.append(train_total_loss)
        val_loss.append(val_total_loss)
        train_n_correct_list.append(train_n_correct)
        val_n_correct_list.append(val_n_correct)

        if (n+1)%5==0:
            states = {
                'state_dict': transformer.state_dict(),
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
        writer.add_scalar('Accuracy/train', train_n_correct/(batch_size*train_num_batches), n)
        writer.add_scalar('Accuracy/test', val_n_correct/(batch_size*val_num_batches), n)