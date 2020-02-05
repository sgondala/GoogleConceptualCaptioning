from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import misc.utils as utils
from torch.nn.utils.rnn import PackedSequence, pack_padded_sequence, pad_packed_sequence

class CiderPredictor(nn.Module):
    def __init__(self, embed_size=16304, embed_dimension=512, lstm_dimension=512, image_feature_dimension=2048, pretrained=None):
        super(CiderPredictor, self).__init__()
        self.image_feature_dimension = image_feature_dimension
        self.embeddings = nn.Embedding(embed_size, embed_dimension)
        if pretrained is not None:
            self.embeddings = nn.Embedding.from_pretrained(pretrained)
        self.bilstm = nn.LSTM(lstm_dimension, lstm_dimension, batch_first=True, bidirectional=True)
        self.attention_linear = nn.Linear(2 * embed_dimension, image_feature_dimension)
        self.attention_layer_2 = nn.Linear(image_feature_dimension, 2 * embed_dimension)
        self.final_layer = nn.Linear(2*embed_dimension, 1)

    def forward(self, image_regions, captions): # N * 36 * 2048, N * 14
        captions_after_embedding = self.embeddings(captions)
        print("Captions shape ", captions_after_embedding.shape)
        batch_size = captions.shape[0]
        _, (hn, _) = self.bilstm(captions_after_embedding) #hn = 2*embedding_size
        print("hn shape ", hn.shape)
        hn = hn.permute(1,0,2)
        print("hm shape ", hn.shape)
        hn = hn.reshape((batch_size, -1))
        print("Hn shape ", hn.shape)
        linear_out = self.attention_linear(hn)
        print("Linear out ", linear_out.shape)
        relu = nn.ReLU()
        linear_out = relu(linear_out)
        linear_out = linear_out.unsqueeze(-1) # N * 2048 * 1
        print("IM reg ", image_regions.shape)
        print("LI out shape ", linear_out.shape)
        attention_out = torch.bmm(image_regions, linear_out) # N * 36 * 1
        attention_out = attention_out.squeeze(-1) # N * 36 
        softmax = nn.Softmax(dim = 1)
        weights = softmax(attention_out).unsqueeze(-1) # N * 36 * 1
        attention_out = torch.sum(image_regions * weights, axis = 1) # N * 2048
        print("attention out shape ", attention_out.shape)
        attention_out = self.attention_layer_2(attention_out) # N * (2*embed_dimension)
        attention_out = relu(attention_out)
        fusion_out = attention_out * hn # N * (2*embed_dimension)
        print("fusion out shape ", fusion_out.shape)
        output = self.final_layer(fusion_out) # N * 1
        output = relu(output)
        return output
