import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.datasets as dsets
from torch.autograd import Variable
import numpy as np
import random
from torch.utils.data import DataLoader, Dataset
import math

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

LENGTH = 4
inputs = []
outputs = []

with open('data_tolokers.json') as f:
  data = json.load(f)

tokenizer = RegexpTokenizer(r'\w+')

def encode(text):
  text = text.lower()
  dialog_tokens = tokenizer.tokenize(text)
  singles = [stemmer.stem(t) for t in dialog_tokens]

  # print(text)

  transcode = []

  for t in singles:
    matches = difflib.get_close_matches(t, tokens, 3, 0.95)
    # print(matches)

    if len(matches) > 0:
      transcode.append(matches[0])
    else:
      transcode.append("*")

  transcoded = ' '.join(transcode)
  # print(' > ' + transcoded)

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

  return tokenIndices

def decode (array):
  # print("decode")
  # print(array)

  output = []

  for a in array:
    index = a.item()
    index = index * NUM_TOKENS
    index = math.floor(round(index, 0))

    output.append(tokens[index])

  return ' '.join(output)
    #print(index)