''' Set up the MPNet model
'''
import torch
import torch.nn as nn
from mpnet.Layers import MLP, Encoder

class MPNet(nn.Module):
    ''' A Motion Planning Network for 2D environment
    '''
    def __init__(
        self,
        AE_input_size,
        state_size,
    ):
        '''
        Initialize the model
        :param AE_input_size: The input size of the autoencoder
        :param AE_output_size: The output size of the autoencoder
        :param output_size: The size of the output for the network
        :param mlp_input_size: The size of the mlp input size.
        '''
        super(MPNet, self).__init__()
        self.encoder = Encoder()

        # For accepting different input shapes
        x = self.encoder(torch.autograd.Variable(torch.rand([1] + AE_input_size)))    
        print(x.shape[-1])
        self.mlp = MLP(x.shape[-1] + state_size*2, state_size)

    def get_environment_encoding(self, obs):
        '''
        Returns the environment encoding.
        :param map: The map for the current planning problem
        '''
        return self.encoder(obs)

    def forward(self, x):
        '''
        Forward step of MPNet
        :param x: The current and goal state concatenated.
        '''
        return self.mlp(x)