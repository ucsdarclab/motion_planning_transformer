''' A script to profile the network.
'''

import numpy as np
import pickle

import torch
import torch.autograd.profiler as profiler

from transformer import Models


def png_decoder(key, value):
    '''
    PNG decoder with gray images.
    :param key:
    :param value:
    '''
    if not key.endswith(".png"):
        return None
    assert isinstance(value, bytes)
    return skimage.io.imread(io.BytesIO(value), as_gray=True)


def cls_decoder(key, value):
    '''
    Converts class represented as bytes to integers.
    :param key:
    :param value:
    :returns the decoded value
    '''
    if not key.endswith(".cls"):
        return None
    assert isinstance(value, bytes)
    return int(value)


import skimage.io
maps = [skimage.io.imread(f'/root/data/env{env}/map_{env}.png', as_gray=True) for env in [1, 2, 3, 4, 5]]

if __name__ =="__main__":
    if torch.cuda.is_available():
        print("Using GPU....")
        device = torch.device('cuda')

    map_size = (480, 480)
    patch_size = 32
    stride = 8


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
        n_classes=((map_size[0]-patch_size)//stride)**2
    ).to(device=device)


    batch_size = 128
    encode_input = torch.rand(batch_size, 2, 480, 480).to(device)
    decoder_input = torch.rand(batch_size, 1, 32, 32).to(device)

    # warm-up
    transformer(encode_input, decoder_input)

    with profiler.profile(with_stack=True, profile_memory=True) as prof:
        pred = transformer(encode_input, decoder_input)
    
    print(prof.key_averages(group_by_stack_n=5).table(sort_by='self_cpu_time_total', row_limit=5))