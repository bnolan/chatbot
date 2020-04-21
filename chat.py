import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.datasets as dsets
from torch.autograd import Variable
import numpy as np
import random
from torch.utils.data import DataLoader, Dataset

import json
import nltk
import difflib
from nltk.tokenize import RegexpTokenizer
from nltk.stem.porter import *

import gen_tensor

class Net(nn.Module):
    def __init__(self, input_size, hidden_size, num_classes):
        super(Net, self).__init__()
        self.fc1 = nn.Linear(input_size, hidden_size) 
        self.fc2 = nn.Linear(hidden_size, hidden_size)
        self.fc3 = nn.Linear(hidden_size, num_classes)
    
    def forward(self, x):
        out = F.relu(self.fc1(x))
        # out = F.relu(self.fc2(x))
        # out = F.relu(self.fc3(x))
        # out = self.fc1(x)
        out = self.fc2(out)
        out = self.fc3(out)
        # return F.log_softmax(out, dim=1)
        return self.sigmoid(out) # final activation function
        # return o
        
    def sigmoid(self, s):
        return 1 / (1 + torch.exp(-s))

    def saveWeights(self, model):
        # we will use the PyTorch internal storage functions
        torch.save(model, "mary")
        # you can reload model with all the weights and so forth with:
        # torch.load("NN")
    

wayne = torch.load("mary")
#print(wayne)

LENGTH = 4

print("You are now chatting with mary...")

text = ''
while text != 'quit':
  text = input("> ")  # Python 3
  inputs = gen_tensor.encode(text) # 
  X = torch.tensor([inputs], dtype=torch.float)
  # print(X)

  output = wayne.forward(X)
  # print(output)
  print('< ' + gen_tensor.decode(output.data[0]))
