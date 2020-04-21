## neural network in pytorch
import torch
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

print(len(tokens))

NUM_TOKENS = len(tokens)
LENGTH = 8
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
          tokenIndices.append(0)
      #[tokens.index(t) for t in transcoded]

      tokenIndices = tokenIndices[0:LENGTH]
      tokenIndices += [0] * (LENGTH - len(tokenIndices))
      print(tokenIndices)

      if priorInput:
        inputs.append(priorInput)
        outputs.append(tokenIndices)
      
      priorInput = tokenIndices

#Input array
# X = torch.Tensor([[0,0,0,0],[0,0,1,1],[1,1,1,1]])
X = torch.Tensor(inputs)
print(X)

#Output
# y = torch.Tensor([[0],[1],[1]])
y = torch.Tensor(outputs)
print(y)

#Sigmoid Function
def sigmoid (x):
  return 1/(1 + torch.exp(-x))

#Derivative of Sigmoid Function
def derivatives_sigmoid(x):
  return x * (1 - x)

#Variable initialization
epoch=35000 #Setting training iterations
lr=0.2 #Setting learning rate
inputlayer_neurons = LENGTH #number of features in data set
hiddenlayer_neurons = 1800 #number of hidden layers neurons
output_neurons = LENGTH #number of neurons at output layer

#weight and bias initialization
wh=torch.randn(inputlayer_neurons, hiddenlayer_neurons).type(torch.FloatTensor)
bh=torch.randn(1, hiddenlayer_neurons).type(torch.FloatTensor)
wout=torch.randn(hiddenlayer_neurons, output_neurons)
bout=torch.randn(1, output_neurons)

for i in range(epoch):
  #Forward Propogation
  hidden_layer_input1 = torch.mm(X, wh)
  hidden_layer_input = hidden_layer_input1 + bh
  hidden_layer_activations = sigmoid(hidden_layer_input)
 
  output_layer_input1 = torch.mm(hidden_layer_activations, wout)
  output_layer_input = output_layer_input1 + bout
  output = sigmoid(output_layer_input1)

  #Backpropagation
  E = y-output
  slope_output_layer = derivatives_sigmoid(output)
  slope_hidden_layer = derivatives_sigmoid(hidden_layer_activations)
  d_output = E * slope_output_layer
  Error_at_hidden_layer = torch.mm(d_output, wout.t())
  d_hiddenlayer = Error_at_hidden_layer * slope_hidden_layer
  wout += torch.mm(hidden_layer_activations.t(), d_output) *lr
  bout += d_output.sum() *lr
  wh += torch.mm(X.t(), d_hiddenlayer) *lr
  bh += d_output.sum() *lr
 
print('actual :\n', y, '\n')
print('predicted :\n', output)