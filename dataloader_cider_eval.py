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

import base64
import csv
import sys
from tqdm import tqdm
import torch

from nltk.tokenize import word_tokenize

punctuations = ["''", "'", "``", "`", "(", ")", "{", "}", ".", "?", "!", ",", ":", "-", "--", "...", ";"]

class HybridLoader:
    """
    Modiying this to read a tsv file and store a dictionary    
    """
    def __init__(self, filepath, ext):
        if ext not in ['acc', 'fc', 'box', 'width', 'height']:
            assert False, "Incorrect extension"
        
        self.id2dict = {}
        cached_file = filepath[:-4] + "__" + ext +"__cached"
        if os.path.exists(cached_file):
            print("Loading saved dict ", cached_file)
            self.id2dict = torch.load(cached_file)
            return 

        assert False, "Saved models not found"

    def get(self, key):
        return self.id2dict[int(key)]

class CiderDataset(Dataset):
    def __init__(self, acc_path, captions_path, cider_values_path, talk_file):
        self.image_features = HybridLoader(acc_path, 'acc')
        self.cider_vals = np.array(json.load(open('coco_cider_scores.json', 'r'))['CIDEr'])
        self.captions = np.array(json.load(open('coco_generated_captions.json', 'r'))) # List of {image_id:, caption:}
        ix_to_word = json.load(open(talk_file, 'r'))['ix_to_word']
        self.word2idx = {}
        for key in ix_to_word:
            self.word2idx[ix_to_word[key]] = key
        assert 'UNK' in self.word2idx
    
    def __len__(self):
        return len(self.captions)
    
    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()

        cider_vals = self.cider_vals[idx]
        captions = self.captions[idx]
        
        image_features = []
        for caption_entry in captions:
            image_id = caption_entry['image_id']
            image_features.append(self.image_features.get(image_id))
        image_features = torch.Tensor(image_features)

        final_captions = [] # Max size is 15 
        for i in range(len(captions)):
            caption = captions[i]['caption'].lower().strip()
            tokens = word_tokenize(caption)
            tokens = [token for token in tokens if token not in punctuations]  
            indexed_caption = [w if w in self.word2idx else 'UNK' for w in tokens][:16]
            final_captions.append(indexed_caption)

        final_captions = torch.Tensor(final_captions)
        return (image_features, final_captions, cider_vals)

# dataloader = DataLoader(transformed_dataset, batch_size=32, shuffle=True, num_workers=4)



    
