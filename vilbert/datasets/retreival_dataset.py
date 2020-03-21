import json
from typing import Any, Dict, List
import random
import os

import torch
from torch.utils.data import Dataset
import numpy as np
import _pickle as cPickle

from pytorch_pretrained_bert.tokenization import BertTokenizer
from ._image_features_reader import ImageFeaturesH5Reader
import jsonlines
import sys
import pdb
import pickle 

def assert_eq(real, expected):
    assert real == expected, "%s (true) vs %s (expected)" % (real, expected)

def _load_annotations(annotations_jsonpath, task):

    with jsonlines.open(annotations_jsonpath) as reader:

        # Build an index which maps image id with a list of caption annotations.
        entries = []
        imgid2entry = {}
        count = 0

        for annotation in reader:
            if task == 'RetrievalCOCO':
                image_id = annotation['id']
            elif task == 'RetrievalFlickr30k':
                image_id = int(annotation['img_path'].split('.')[0])
            imgid2entry[image_id] = []
            for sentences in annotation['sentences']:
                entries.append({"caption": sentences, 'image_id':image_id})
                imgid2entry[image_id].append(count)
                count += 1

    return entries, imgid2entry


class RetreivalDataset(Dataset):
    def __init__(
        self,
        task: str,
        dataroot: str,
        annotations_jsonpath: str,
        split: str,
        image_features_reader: ImageFeaturesH5Reader,
        gt_image_features_reader: ImageFeaturesH5Reader,
        tokenizer: BertTokenizer,
        padding_index: int = 0,
        max_seq_length: int = 20,
        max_region_num: int = 37,
    ):
        # All the keys in `self._entries` would be present in `self._image_features_reader`

        self._entries, self.imgid2entry = _load_annotations(annotations_jsonpath, task)
        self.image_id_list = [*self.imgid2entry]

        self._image_features_reader = image_features_reader
        self._tokenizer = tokenizer
        self.num_labels = 1
        self._split = split
        self._padding_index = padding_index
        self._max_region_num = max_region_num
        self._max_seq_length = max_seq_length

        if self._split == 'train':
            image_info = cPickle.load(open(os.path.join(dataroot, 'hard_negative.pkl'), 'rb'))
            for key, value in image_info.items():
                setattr(self, key, value)
            self.train_imgId2pool = {imageId:i for i, imageId in enumerate(self.train_image_list)}

        cache_path = os.path.join(dataroot, "cache", task + '_' + split + '_' + str(max_seq_length)+'.pkl')

        if not os.path.exists(cache_path):
            self.tokenize()
            self.tensorize()
            cPickle.dump(self._entries, open(cache_path, 'wb'))
        else:
            print('loading entries from %s' %(cache_path))
            self._entries = cPickle.load(open(cache_path, "rb"))

    def tokenize(self):
        """Tokenizes the captions.

        This will add caption_tokens in each entry of the dataset.
        -1 represents nil, and should be treated as padding_idx in embedding.
        """
        for entry in self._entries:
            sentence_tokens = self._tokenizer.tokenize(entry["caption"])
            sentence_tokens = ["[CLS]"] + sentence_tokens + ["[SEP]"]

            tokens = [
                self._tokenizer.vocab.get(w, self._tokenizer.vocab["[UNK]"])
                for w in sentence_tokens
            ]
            tokens = tokens[:self._max_seq_length]
            segment_ids = [0] * len(tokens)
            input_mask = [1] * len(tokens)

            if len(tokens) < self._max_seq_length:
                # Note here we pad in front of the sentence
                padding = [self._padding_index] * (self._max_seq_length - len(tokens))
                tokens = tokens + padding
                input_mask += padding
                segment_ids += padding

            assert_eq(len(tokens), self._max_seq_length)
            entry["token"] = tokens
            entry["input_mask"] = input_mask
            entry["segment_ids"] = segment_ids

    def tensorize(self):

        for entry in self._entries:
            token = torch.from_numpy(np.array(entry["token"]))
            entry["token"] = token

            input_mask = torch.from_numpy(np.array(entry["input_mask"]))
            entry["input_mask"] = input_mask

            segment_ids = torch.from_numpy(np.array(entry["segment_ids"]))
            entry["segment_ids"] = segment_ids


    def __getitem__(self, index):
        entry = self._entries[index]
        image_id = entry["image_id"]

        features, num_boxes, boxes, _ = self._image_features_reader[image_id]
        
        mix_num_boxes = min(int(num_boxes), self._max_region_num)
        mix_boxes_pad = np.zeros((self._max_region_num, 5))
        mix_features_pad = np.zeros((self._max_region_num, 2048))

        image_mask = [1] * (int(mix_num_boxes))
        while len(image_mask) < self._max_region_num:
            image_mask.append(0)

        mix_boxes_pad[:mix_num_boxes] = boxes[:mix_num_boxes]
        mix_features_pad[:mix_num_boxes] = features[:mix_num_boxes]

        features1 = torch.tensor(mix_features_pad).float()
        image_mask1 = torch.tensor(image_mask).long()
        spatials1 = torch.tensor(mix_boxes_pad).float()

        caption1 = entry["token"]
        input_mask1 = entry["input_mask"]
        segment_ids1 = entry["segment_ids"]
        # negative samples.
        # 1: correct one, 2: random caption wrong, 3: random image wrong. 4: hard image wrong.
        
        while True:
            # sample a random image:
            img_id2 = random.choice(self.image_id_list)
            if img_id2 != image_id: break

        entry2 = self._entries[random.choice(self.imgid2entry[img_id2])]

        features2 = features1
        image_mask2 = image_mask1
        spatials2 = spatials1
        caption2 = entry2["token"]
        input_mask2 = entry2["input_mask"]
        segment_ids2 = entry2["segment_ids"]        

        # random image wrong
        while True:
            # sample a random image:
            img_id3 = random.choice(self.image_id_list)
            if img_id3 != image_id: break        

        features3, num_boxes3, boxes3, _ = self._image_features_reader[img_id3]
        image_mask3 = [1] * (int(num_boxes3))

        mix_num_boxes3 = min(int(num_boxes3), self._max_region_num)
        mix_boxes_pad3 = np.zeros((self._max_region_num, 5))
        mix_features_pad3 = np.zeros((self._max_region_num, 2048))

        while len(image_mask3) < self._max_region_num:
            image_mask3.append(0)        


        mix_boxes_pad[:mix_num_boxes3] = boxes3[:mix_num_boxes3]
        mix_features_pad[:mix_num_boxes3] = features3[:mix_num_boxes3]

        features3 = torch.tensor(mix_features_pad).float()
        image_mask3 = torch.tensor(image_mask3).long()
        spatials3 = torch.tensor(mix_boxes_pad).float()

        caption3 = caption1
        input_mask3 = input_mask1
        segment_ids3 = segment_ids1


        if self._split == 'train':
            # random hard caption.
            rand_img_id_pool = self.train_hard_pool[self.train_imgId2pool[image_id]]
            pool_img_idx = int(rand_img_id_pool[np.random.randint(1, len(rand_img_id_pool))])
            img_id4 = self.train_image_list[pool_img_idx]
        else:
            while True:
                # sample a random image:
                img_id4 = random.choice(self.image_id_list)
                if img_id4 != image_id: break

        entry4 = self._entries[random.choice(self.imgid2entry[img_id4])]

        features4 = features1
        image_mask4 = image_mask1
        spatials4 = spatials1
        caption4 = entry4["token"]
        input_mask4 = entry4["input_mask"]
        segment_ids4 = entry4["segment_ids"]

        features = torch.stack([features1, features2, features3, features4], dim=0)
        spatials = torch.stack([spatials1, spatials2, spatials3, spatials4], dim=0)
        image_mask = torch.stack([image_mask1, image_mask2, image_mask3, image_mask4], dim=0)
        caption = torch.stack([caption1, caption2, caption3, caption4], dim=0)
        input_mask = torch.stack([input_mask1, input_mask2, input_mask3, input_mask4], dim=0)
        segment_ids = torch.stack([segment_ids1, segment_ids2, segment_ids3, segment_ids4], dim=0)
        co_attention_mask = torch.zeros((4, self._max_region_num, self._max_seq_length))
        target = 0

        return features, spatials, image_mask, caption, target, input_mask, segment_ids, co_attention_mask, image_id

    def __len__(self):
        return len(self._entries)

def _load_annotationsVal(annotations_jsonpath, task):

    with jsonlines.open(annotations_jsonpath) as reader:

        # Build an index which maps image id with a list of caption annotations.
        image_entries = {}
        caption_entries = []

        for annotation in reader:
            if task == 'RetrievalCOCO':
                image_id = annotation['id']
            elif task == 'RetrievalFlickr30k':
                image_id = int(annotation['img_path'].split('.')[0])

            image_entries[image_id] = 1

            for sentences in annotation['sentences']:
                caption_entries.append({"caption": sentences, 'image_id':image_id})

    image_entries = [*image_entries]

    return image_entries, caption_entries

class RetreivalDatasetVal(Dataset):
    def __init__(
        self,
        task: str,
        dataroot: str,
        annotations_jsonpath: str,
        split: str,
        image_features_reader: ImageFeaturesH5Reader,
        gt_image_features_reader: ImageFeaturesH5Reader,
        tokenizer: BertTokenizer,
        padding_index: int = 0,
        max_seq_length: int = 20,
        max_region_num: int = 101,
    ):
        # All the keys in `self._entries` would be present in `self._image_features_reader`

        self._image_entries, self._caption_entries = _load_annotationsVal(annotations_jsonpath, task)
        self._image_features_reader = image_features_reader
        self._tokenizer = tokenizer

        self._split = split
        self._padding_index = padding_index
        self._max_region_num = max_region_num
        self._max_seq_length = max_seq_length
        self.num_labels = 1

        # cache file path data/cache/train_ques
        # cap_cache_path = "data/cocoRetreival/cache/val_cap.pkl"
        # if not os.path.exists(cap_cache_path):
        self.tokenize()
        self.tensorize()
            # cPickle.dump(self._entries, open(cap_cache_path, 'wb'))
        # else:
            # print('loading entries from %s' %(cap_cache_path))
            # self._entries = cPickle.load(open(cap_cache_path, "rb"))
# 
        self.features_all = np.zeros((1000, self._max_region_num, 2048))
        self.spatials_all = np.zeros((1000, self._max_region_num, 5))
        self.image_mask_all = np.zeros((1000, self._max_region_num))

        for i, image_id in enumerate(self._image_entries):
            features, num_boxes, boxes, _ = self._image_features_reader[image_id]

            mix_num_boxes = min(int(num_boxes), self._max_region_num)
            mix_boxes_pad = np.zeros((self._max_region_num, 5))
            mix_features_pad = np.zeros((self._max_region_num, 2048))

            image_mask = [1] * (int(mix_num_boxes))
            while len(image_mask) < self._max_region_num:
                image_mask.append(0)

            mix_boxes_pad[:mix_num_boxes] = boxes[:mix_num_boxes]
            mix_features_pad[:mix_num_boxes] = features[:mix_num_boxes]

            self.features_all[i] = mix_features_pad
            self.image_mask_all[i] = np.array(image_mask)
            self.spatials_all[i] = mix_boxes_pad

            sys.stdout.write('%d/%d\r' % (i, len(self._image_entries)))
            sys.stdout.flush()

        self.features_all = torch.Tensor(self.features_all).float()
        self.image_mask_all = torch.Tensor(self.image_mask_all).long()
        self.spatials_all = torch.Tensor(self.spatials_all).float()

    def tokenize(self):
        """Tokenizes the captions.

        This will add caption_tokens in each entry of the dataset.
        -1 represents nil, and should be treated as padding_idx in embedding.
        """
        for entry in self._caption_entries:
            sentence_tokens = self._tokenizer.tokenize(entry["caption"])
            sentence_tokens = ["[CLS]"] + sentence_tokens + ["[SEP]"]

            tokens = [
                self._tokenizer.vocab.get(w, self._tokenizer.vocab["[UNK]"])
                for w in sentence_tokens
            ]
            tokens = tokens[:self._max_seq_length]
            segment_ids = [0] * len(tokens)
            input_mask = [1] * len(tokens)

            if len(tokens) < self._max_seq_length:
                # Note here we pad in front of the sentence
                padding = [self._padding_index] * (self._max_seq_length - len(tokens))
                tokens = tokens + padding
                input_mask += padding
                segment_ids += padding

            assert_eq(len(tokens), self._max_seq_length)
            entry["token"] = tokens
            entry["input_mask"] = input_mask
            entry["segment_ids"] = segment_ids

    def tensorize(self):
        for entry in self._caption_entries:
            token = torch.from_numpy(np.array(entry["token"])).long()
            entry["token"] = token

            input_mask = torch.from_numpy(np.array(entry["input_mask"]))
            entry["input_mask"] = input_mask

            segment_ids = torch.from_numpy(np.array(entry["segment_ids"])).long()
            entry["segment_ids"] = segment_ids

    def __getitem__(self, index):

        # we iterate through every caption here.
        caption_idx = int(index / 2)
        image_idx = index % 2

        if image_idx == 0:
            image_entries = self._image_entries[:500]
            features_all = self.features_all[:500]
            spatials_all = self.spatials_all[:500]
            image_mask_all = self.image_mask_all[:500]

        else:
            image_entries = self._image_entries[500:]
            features_all = self.features_all[500:]
            spatials_all = self.spatials_all[500:]
            image_mask_all = self.image_mask_all[500:]

        entry = self._caption_entries[caption_idx]
        caption = entry["token"]
        input_mask = entry["input_mask"]
        segment_ids = entry["segment_ids"]

        target_all = torch.zeros(500)
        for i, image_id in enumerate(image_entries):
            if image_id == entry["image_id"]:
                target_all[i] = 1

        return features_all, spatials_all, image_mask_all, caption, input_mask, segment_ids, target_all, caption_idx, image_idx

    def __len__(self):
        return len(self._caption_entries) * 2


class HybridLoader:
    """
    Modiying this to read a tsv file and store a dictionary    
    """
    def __init__(self, filepath, ext):
        if ext not in ['acc', 'fc', 'box', 'width', 'height']:
            assert False, "Incorrect extension"
        
        self.id2dict = {}
        cached_file = filepath[:-4] + "__" + ext +"__cached"

        # if ext == 'acc': # Very temporary bad hack
        #     cached_file = cached_file + '.pkl'
        #     self.id2dict = pickle.load(open(cached_file, 'rb'), encoding="latin1")
        #     return

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
        captions_path: str,
        images_path: str,
        cider_values_path: str,
        tokenizer: BertTokenizer,
        is_eval: bool = False,
        padding_index: int = 0,
        max_seq_length: int = 20,
        max_region_num: int = 37,
    ):
        # Cider values are sequential. Get index i in cider_vals, same index in captions to get image_id

        self.image_features = HybridLoader(images_path, 'acc')
        self.image_boxes = HybridLoader(images_path, 'box')
        self.image_height = HybridLoader(images_path, 'height')
        self.image_width = HybridLoader(images_path, 'width')
        self.captions = np.array(json.load(open(captions_path, 'r'))) # List of {image_id:, caption:}
        self.cider_vals = json.load(open(cider_values_path, 'r'))
        if isinstance(self.cider_vals, list):
            self.cider_vals = np.array(self.cider_vals)
        else:
            self.cider_vals = np.array(self.cider_vals['CIDEr'])

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

        assert_eq(len(tokens), self._max_seq_length)
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

        y = self.cider_vals[index]
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
            return (features, spatials, image_mask, token, target, input_mask, segment_ids, co_attention_mask, image_id, y, caption_raw)
        else:
            return (features, spatials, image_mask, token, target, input_mask, segment_ids, co_attention_mask, image_id, y)

    def __len__(self):
        return len(self.captions)
