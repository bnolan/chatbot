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

stemmer = PorterStemmer()

with open("ogen.txt") as f:
  words = f.readlines()
words = ' '.join(words)
words = words.replace(',', ' ')
words = words.replace('.', ' ')
words = words.lower()

tokens = nltk.word_tokenize(words)
singles = [stemmer.stem(t) for t in tokens]

with open("basic.txt") as f:
  words = f.readlines()
words = ' '.join(words)
words = words.replace(',', ' ')
words = words.replace('.', ' ')
words = words.lower()
t2 = nltk.word_tokenize(words)
s2 = [stemmer.stem(t) for t in t2]

tokens = list(set(singles + s2))
tokens.sort()


NUM_TOKENS = len(tokens)
print(NUM_TOKENS)

LENGTH = 8
inputs = []
outputs = []


with open('data_tolokers.json') as f:
  data = json.load(f)

tokenizer = RegexpTokenizer(r'\w+')

priorInput = False

for d in data:
  dialog = d["dialog"]

  if len(inputs) > 200:
    continue

  for f in dialog:
    if f["sender_class"] == "Bot":
      continue
    else:
      # print(f)

      text = f["text"].lower()
      dialog_tokens = tokenizer.tokenize(text)
      singles = [stemmer.stem(t) for t in dialog_tokens]

      print(text)

      transcode = []

      for t in singles:
        matches = difflib.get_close_matches(t, tokens, 3, 0.95)
        # print(matches)

        if len(matches) > 0:
          transcode.append(matches[0])
        else:
          transcode.append("*")
    
      transcoded = ' '.join(transcode)
      print(' > ' + transcoded)

      tokenIndices = []
      
      for t in transcode:
        # print("x" + t + "x ")
        try:
          tokenIndices.append(tokens.index(t) / NUM_TOKENS)
        except ValueError:
          tokenIndices.append(0.0)
      #[tokens.index(t) for t in transcoded]

      tokenIndices = tokenIndices[0:LENGTH]
      tokenIndices += [0.0] * (LENGTH - len(tokenIndices))
      print(tokenIndices)

      if priorInput:
        inputs.append(priorInput)
        outputs.append(tokenIndices)
      
      priorInput = tokenIndices




input_size = LENGTH
hidden_size = 300
num_classes = LENGTH
num_epochs = 1000
batch_size = 10

training_data = torch.FloatTensor(inputs)
training_label = torch.FloatTensor(outputs)

class MyDataset(Dataset):
    def __init__(self, datas, labels):
        self.datas = datas
        self.labels = labels

    def __getitem__(self, index):
        data, target = self.datas[index], self.labels[index] 
        return data, target

    def __len__(self):
        return len(self.datas)

train_dataloader = DataLoader(MyDataset(training_data, training_label), batch_size=batch_size)

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
        torch.save(model, "wayneo")
        # you can reload model with all the weights and so forth with:
        # torch.load("NN")
    
net = Net(input_size, hidden_size, num_classes)
    
input_size = LENGTH
hidden_sizes = [128, 64]
output_size = LENGTH

model = nn.Sequential(nn.Linear(input_size, hidden_sizes[0]),
                      nn.ReLU(),
                      nn.Linear(hidden_sizes[0], hidden_sizes[1]),
                      nn.ReLU(),
                      nn.Linear(hidden_sizes[1], output_size),
                      nn.Softmax(dim=1))


# criterion = nn.NLLLoss()  
# criterion = nn.NLLLoss()
criterion = nn.CrossEntropyLoss()

learning_rate = 0.001
optimizer = torch.optim.SGD(net.parameters(), lr=learning_rate)

X = torch.tensor(inputs, dtype=torch.float)
print(X)

#Output
# y = torch.Tensor([[0],[1],[1]])
y = torch.tensor(outputs, dtype=torch.long)
print(y)

for epoch in range(num_epochs):
  for data, labels in train_dataloader:  
      # Convert torch tensor to Variable
      data = Variable(data)
      labels = Variable(labels)
      
      # Forward + Backward + Optimize
      optimizer.zero_grad()  # zero the gradient buffer
      outputs = net(data)
      # outputs = model(data)
      # loss = criterion(outputs, labels.long())
      loss = criterion(outputs, torch.max(labels, 1)[1])
      loss.backward()
      optimizer.step()

  print ('Epoch [%d/%d], Loss: %.4f'
    %(epoch+1, num_epochs, loss))

net.saveWeights(net)


# epochs = 1000
# for e in range(epochs):
#   for data, labels in train_dataloader:  
#     running_loss = 0
#     # Flatten MNIST images into a 784 long vector
#     # images = images.view(images.shape[0], -1)

#     # Training pass
#     optimizer.zero_grad()
    
#     output = model(X)
#     # loss = criterion(output, y)
#     loss = criterion(output, torch.max(y, 1)[1])

#     loss.backward()
#     optimizer.step()
    
#     running_loss += loss.item()

#     print(f"Training loss: {running_loss}") # /len(trainloader)}")

# # Train the Model
# for epoch in range(num_epochs):
#     for data, labels in train_dataloader:  
#         # Convert torch tensor to Variable
#         data = Variable(data)
#         labels = Variable(labels)
        
#         # Forward + Backward + Optimize
#         optimizer.zero_grad()  # zero the gradient buffer
#         outputs = net(data)

#         print('wtf')
#         print(outputs)
#         print(labels)

#         loss = criterion(outputs, labels.data[0].item())
#         loss.backward()
#         optimizer.step()
#         print ('Epoch [%d/%d], Loss: %.4f' 
#                    %(epoch+1, num_epochs, loss.data[0]))


