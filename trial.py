## neural network in pytorch
import torch
import torch.nn as nn
import torch.nn.functional as F
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

LENGTH = 2
inputs = []
outputs = []


with open('data_tolokers.json') as f:
  data = json.load(f)

tokenizer = RegexpTokenizer(r'\w+')

priorInput = False

for d in data:
  dialog = d["dialog"]

  if len(inputs) > 50:
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

#Input array
# X = torch.Tensor([[0,0,0,0],[0,0,1,1],[1,1,1,1]])
X = torch.tensor(inputs, dtype=torch.float)
print(X)

#Output
# y = torch.Tensor([[0],[1],[1]])
y = torch.tensor(outputs, dtype=torch.float)
print(y)

class Neural_Network(nn.Module):
    def __init__(self, ):
        super(Neural_Network, self).__init__()
        # parameters
        # TODO: parameters can be parameterized instead of declaring them here
        self.inputSize = LENGTH
        self.outputSize = LENGTH
        self.hiddenSize = 600
        
        # weights
        self.W1 = torch.randn(self.inputSize, self.hiddenSize) # 2 X 3 tensor
        self.W2 = torch.randn(self.hiddenSize, self.outputSize) # 3 X 1 tensor
        
    # def forward(self, x):
    #     out = self.W1(x)
    #     # out = F.relu(self.fc1(x))
    #     out = self.W2(out)
    #     # out = self.fc3(out)
    #     return F.log_softmax(out, dim=1)

    def forward(self, X):
        self.z = torch.matmul(X, self.W1) # 3 X 3 ".dot" does not broadcast in PyTorch
        self.z2 = self.sigmoid(self.z) # activation function
        self.z3 = torch.matmul(self.z2, self.W2)
        print(self.z3)
        o = self.sigmoid(self.z3) # final activation function
        return o
        
    def sigmoid(self, s):
        return 1 / (1 + torch.exp(-s))
    
    def sigmoidPrime(self, s):
        # derivative of sigmoid
        return s * (1 - s)
    
    def backward(self, X, y, o):
        self.o_error = y - o # error in output
        self.o_delta = self.o_error * self.sigmoidPrime(o) # derivative of sig to error
        self.z2_error = torch.matmul(self.o_delta, torch.t(self.W2))
        self.z2_delta = self.z2_error * self.sigmoidPrime(self.z2)
        self.W1 += torch.matmul(torch.t(X), self.z2_delta)
        self.W2 += torch.matmul(torch.t(self.z2), self.o_delta)
        
    def train(self, X, y):
        # forward + backward pass for training
        o = self.forward(X)
        self.backward(X, y, o)
        
    def saveWeights(self, model):
        # we will use the PyTorch internal storage functions
        torch.save(model, "NN")
        # you can reload model with all the weights and so forth with:
        # torch.load("NN")
        
    def predict(self):
        print ("Predicted data based on trained weights: ")
        print ("Input (scaled): \n" + str(xPredicted))
        print ("Output: \n" + str(self.forward(xPredicted)))


NN = Neural_Network()
for i in range(5000):  # trains the NN 1,000 times
    print ("#" + str(i) + " Loss: " + str(torch.mean((y - NN(X))**2).detach().item()))  # mean sum squared loss
    NN.train(X, y)
NN.saveWeights(NN)

prediction = inputs[0]
xPredicted = torch.tensor(prediction, dtype=torch.float) # 1 X 2 tensor
NN.predict()

