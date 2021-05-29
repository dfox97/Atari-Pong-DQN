import os
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import numpy as np
#https://towardsdatascience.com/tutorial-double-deep-q-learning-with-dueling-network-architectures-4c1b3fb7f756
class DeepQNetwork(nn.Module):
    def __init__(self, lr, n_actions, name, input_dims, chkpt_dir):
        super(DeepQNetwork, self).__init__()
        """
        Creation of deepQNetwork.
        The DQN (Deep Q-Network) algorithm was developed by DeepMind in 2015

        lr=Learning rate
        n_acitons=number of actions
        name=name of checkpoint file
        input_dims=Input dimensions of environment. 
        chkpt_dir=checkpoing directory.

        uses the nn.Module to inherit subclass. helps take and receive tensor inputs and outputs. Can call things like zero_grad. forward function, nn.Linear ect..

        conv = convolution layers for cnn network, image processing
        fc=full connected layers 
        optimizer=helps shape and mold model to be as accurate as possible. RMSprop is a gradient based optimization technique
        loss=calculates the difference between the network output and its expected output. MSE = Mean squared error.



        """
        #device init , gpu if available otherwise use cpu.
        self.device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
        
        self.checkpoint_dir = chkpt_dir#checkpoint directory to save and load file
        self.checkpoint_file = os.path.join(self.checkpoint_dir, name)#checkpoint file
        #print(input_dims)
        #input dims=num channels in our input image should be 4x1 frames  because turns greyscale 1 channel since greyscale if color it would be 4x3 for 3 rgb colors.
        self.conv1 = nn.Conv2d(input_dims[0], 32, 8, stride=4)#first conv layer 4 input dimentions so 4 frames of pong, 32filters output channels,8x8kernal, stride 4
        self.conv2 = nn.Conv2d(32, 64, 4, stride=2)#second cnn layer 64,4x4 with stride 2
        self.conv3 = nn.Conv2d(64, 64, 3, stride=1)#final conv3 layer is 64 with 3x3 and stride is equal 1. #Which is flattened out for the 2 fully connected layers in the network.


        #fully connecter layer input dimensions
        fc_InputDims = self.calc_convOutputDims(input_dims)#figure out the fully connected inputs for fc1.
        #print(fc_input_dims)#3136

        self.fc1 = nn.Linear(fc_InputDims, 512) #512 neuron outputs
        self.fc2 = nn.Linear(512, n_actions)#final outputs are possible actions. 
        #print(n_actions)#6 actions

        #optimizer RMS used in deep learning paper.
        #optimizes params of our network
        self.optimizer = optim.RMSprop(self.parameters(), lr=lr)
        #Means squared error to workout loss for backpropagation 
        self.loss = nn.MSELoss() 
        
        self.to(self.device)#init to device pytorch funtion.


    def forward(self, state):
        """
        state of the env as input and state can be a batch size of one by input dim or a batch size of 32 by inputs dims. 

        pass state through conv layers and use a activation relu through each layer apart from last fully connected as its not needed.

        we reshape the final conv layer becuase we want to pass in the something that has a shape batch size by whatever num of input features, into the fully connected layer.
        [0] element is batch size
        -1 is telling the function that it wants to flatten the rest of the functions 
        
        the final output of the deep neural network is the action values of the given state. Note no activation function used for the final fc layer.

        """

        conv1 = F.relu(self.conv1(state))
        conv2 = F.relu(self.conv2(conv1))
        conv3 = F.relu(self.conv3(conv2))
        # conv3 shape is batchsize x num filters x Height x Width (of final conv image not the input images)
        conv_state = conv3.view(conv3.size()[0], -1)
        # conv_state shape is BS x (n_filters * H * W)
        flatten1 = F.relu(self.fc1(conv_state))
        actions = self.fc2(flatten1)

        return actions
    
    def calc_convOutputDims(self, input_dims):
        """
        find the input dimensions for the fully connected layer (4x84x84), taking total input dimensions
        return the product of the last layer size.
        """
        state = torch.zeros(1, *input_dims)#state matrix as batch size one with input dims.
        x = self.conv1(state)#pass into conv layers.
        x = self.conv2(x)
        x = self.conv3(x)
        return int(np.prod(x.size()))#best to return int float not needed.

    def save_checkpoint(self):
        """
        create checkpoint and saves file.
        """
        print('Saving checkpoint ...')
        torch.save(self.state_dict(), self.checkpoint_file)

    def load_checkpoint(self):
        """
        load checkpoint file.
        """
        print('Loading checkpoint ...')
        self.load_state_dict(torch.load(self.checkpoint_file))
