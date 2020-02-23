import json
import random
import os

import torch
from torch.utils.data import Dataset
import numpy as np
import _pickle as cPickle

from pytorch_pretrained_bert.tokenization import BertTokenizer
import sys
import pdb
import pickle 

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

        assert False, "Saved models not found " + cached_file

    def get(self, key):
        return self.id2dict[int(key)]

class CiderDataset(Dataset):
    def __init__(
        self,
        captions,
        images_path,
        tokenizer,
        is_eval = False,
        padding_index = 0,
        max_seq_length = 20,
        max_region_num = 37,
    ):
        # Cider values are sequential. Get index i in cider_vals, same index in captions to get image_id

        self.image_features = HybridLoader(images_path, 'acc')
        self.image_boxes = HybridLoader(images_path, 'box')
        self.image_height = HybridLoader(images_path, 'height')
        self.image_width = HybridLoader(images_path, 'width')
        # self.captions = np.array(json.load(open(captions_path, 'r'))) # List of {image_id:, caption:}
        self.captions = captions 
        # if isinstance(self.cider_vals, list):
        #     self.cider_vals = np.array(self.cider_vals)
        # else:
        #     self.cider_vals = np.array(self.cider_vals['CIDEr'])

        self._tokenizer = tokenizer
        self.num_labels = 1
        
        self._padding_index = padding_index
        self._max_region_num = max_region_num
        self._max_seq_length = max_seq_length
        self._is_eval = is_eval

    def tokenize(self, caption):
        """Tokenizes the captions.

        This will add caption_tokens in each entry of the dataset.
        -1 represents nil, and should be treated as padding_idx in embedding.
        """
        
        sentence_tokens = self._tokenizer.tokenize(caption)
        sentence_tokens = ["[CLS]"] + sentence_tokens + ["[SEP]"]

        tokens = [self._tokenizer.vocab.get(w, self._tokenizer.vocab["[UNK]"]) for w in sentence_tokens]
        tokens = tokens[:self._max_seq_length]

        segment_ids = [0] * len(tokens)
        input_mask = [1] * len(tokens)

        if len(tokens) < self._max_seq_length:
            # Note here we pad in front of the sentence
            padding = [self._padding_index] * (self._max_seq_length - len(tokens))
            tokens = tokens + padding
            input_mask += padding
            segment_ids += padding

        assert len(tokens) == self._max_seq_length
        return tokens, input_mask, segment_ids

    def tensorize(self, tokens, input_mask, segment_ids):
        tokens = torch.from_numpy(np.array(tokens))
        input_mask = torch.from_numpy(np.array(input_mask))
        segment_ids = torch.from_numpy(np.array(segment_ids))
        return tokens, input_mask, segment_ids
    
    def preprocess_features(self, features):
        features = features.reshape(-1, 2048)
        num_boxes = features.shape[0]

        g_feat = np.sum(features, axis=0) / num_boxes
        num_boxes = num_boxes + 1

        features = np.concatenate(
            [np.expand_dims(g_feat, axis=0), features], axis=0
        )
        return features

    def preprocess_boxes(self, boxes, image_h, image_w):
        # N * 4 => Normalized (N + 1)* 5

        boxes = boxes.reshape(-1, 4)
        image_location = np.zeros((boxes.shape[0], 5), dtype=np.float32)
        image_location[:, :4] = boxes
        image_location[:, 4] = (
            (image_location[:, 3] - image_location[:, 1])
            * (image_location[:, 2] - image_location[:, 0])
            / (float(image_w) * float(image_h))
        )

        image_location[:, 0] = image_location[:, 0] / float(image_w)
        image_location[:, 1] = image_location[:, 1] / float(image_h)
        image_location[:, 2] = image_location[:, 2] / float(image_w)
        image_location[:, 3] = image_location[:, 3] / float(image_h)

        g_location = np.array([0, 0, 1, 1, 1])
        image_location = np.concatenate(
            [np.expand_dims(g_location, axis=0), image_location], axis=0
        )
        return image_location

    def __getitem__(self, index):
        # print(self.captions)
        # y = self.cider_vals[index]
        caption_entry = self.captions[index]
        
        image_id = caption_entry['image_id']
        caption_raw = caption_entry['caption'].lower().strip()

        features = self.image_features.get(image_id)
        boxes = self.image_boxes.get(image_id)
        image_h = self.image_height.get(image_id)
        image_w = self.image_width.get(image_id)

        features = self.preprocess_features(features)
        boxes = self.preprocess_boxes(boxes, image_h, image_w)

        num_boxes = boxes.shape[0]

        mix_num_boxes = min(int(num_boxes), self._max_region_num)
        mix_boxes_pad = np.zeros((self._max_region_num, 5))
        mix_features_pad = np.zeros((self._max_region_num, 2048))

        image_mask = [1] * (int(mix_num_boxes))
        while len(image_mask) < self._max_region_num:
            image_mask.append(0)

        mix_boxes_pad[:mix_num_boxes] = boxes[:mix_num_boxes]
        mix_features_pad[:mix_num_boxes] = features[:mix_num_boxes]

        features = torch.tensor(mix_features_pad).float()
        image_mask = torch.tensor(image_mask).long()
        spatials = torch.tensor(mix_boxes_pad).float()

        token, input_mask, segment_ids = self.tokenize(caption_raw)
        token, input_mask, segment_ids = self.tensorize(token, input_mask, segment_ids)
        
        co_attention_mask = torch.zeros((self._max_region_num, self._max_seq_length))
        target = 0

        if self._is_eval:
            return (features, spatials, image_mask, token, target, input_mask, segment_ids, co_attention_mask, image_id, caption_raw)
        else:
            return (features, spatials, image_mask, token, target, input_mask, segment_ids, co_attention_mask, image_id)

    def __len__(self):
        return len(self.captions)
