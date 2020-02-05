from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import json
import h5py
import numpy as np
import random
import os
import torch
import torch.utils.data as data
from torch.utils.data import Dataset
from torch.utils.data import DataLoader
from torch.utils.data import random_split

import base64
import csv
import sys
import torch

class TempDataset(Dataset):
    def __init__(self):
        pass

    def __len__(self):
        return 100
    
    def __getitem__(self, index):
        return index, index**2, index**3

dataset = TempDataset()
train, valid, test = random_split(dataset, [80, 10, 10])
print(len(train))
print(len(valid))
print(len(test))

dataloader = DataLoader(train, batch_size=10)

for batch in dataloader:
    a,b,c = batch
    print("A", a)
    print("B", b)
    print("C", c)
#     print(batch)
    # break
#     # X, y = batch
#     # print("X", X)
#     # print("Y", y)
