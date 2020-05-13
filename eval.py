from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import json
import numpy as np

import time
import os
from six.moves import cPickle

import opts
import models
from dataloader import *
from dataloaderraw import *
import eval_utils
import argparse
import misc.utils as utils
import torch

# Reproducibility
np.random.seed(42)
torch.manual_seed(42)
torch.cuda.manual_seed_all(42)
torch.backends.cudnn.benchmark = False
torch.backends.cudnn.deterministic = True

# Input arguments and options
parser = argparse.ArgumentParser()
# Input paths
parser.add_argument('--model', type=str, default='',
                help='path to model to evaluate')
parser.add_argument('--cnn_model', type=str,  default='resnet101',
                help='resnet101, resnet152')
parser.add_argument('--infos_path', type=str, default='',
                help='path to infos to evaluate')
parser.add_argument('--out_file_path', type=str, default = '')

opts.add_eval_options(parser)
opts.add_diversity_opts(parser)
opt = parser.parse_args()

assert len(opt.out_file_path) != 0

print(opt)

# Load infos
with open(opt.infos_path, 'rb') as f:
    infos = utils.pickle_load(f)

# override and collect parameters
replace = ['input_fc_dir', 'input_att_dir', 'input_box_dir', 'input_label_h5', 'input_json', 'batch_size', 'id']
ignore = ['start_from']

for k in vars(infos['opt']).keys():
    if k in replace:
        setattr(opt, k, getattr(opt, k) or getattr(infos['opt'], k, ''))
    elif k not in ignore:
        if not k in vars(opt):
            vars(opt).update({k: vars(infos['opt'])[k]}) # copy over options from model

vocab = infos['vocab'] # ix -> word mapping

# Setup the model
opt.vocab = vocab
model = models.setup(opt)
del opt.vocab
model.load_state_dict(torch.load(opt.model))
model.cuda()
model.eval()
crit = utils.LanguageModelCriterion()

print(model.parameters)
# assert False
# Create the Data Loader instance
# print(opt.image_folder)
#if len(opt.image_folder) == 0:
print(opt)
loader = DataLoader(opt)
#else:
#  assert False
#else:
#  loader = DataLoaderRaw({'folder_path': opt.image_folder, 
#                            'coco_json': opt.coco_json,
#                            'batch_size': opt.batch_size,
#                            'cnn_model': opt.cnn_model})

# When eval using provided pretrained model, the vocab may be different from what you have in your cocotalk.json
# So make sure to use the vocab in infos file.
loader.ix_to_word = infos['vocab']

# Set sample options
loss, split_predictions, lang_stats = eval_utils.eval_split(model, crit, loader, vars(opt))

print('loss: ', loss)
if lang_stats:
  print(lang_stats)

if opt.post_processing == 1:
    for i in range(len(split_predictions)):
      split_predictions[i]['caption'] = utils.post_processing_for_conceptual(split_predictions[i]['caption'])

if opt.dump_json == 1:
    # dump the json
    # json.dump(split_predictions, open('vis/vis.json', 'w'))
    print("Dumping results")
    json.dump(split_predictions, open(opt.out_file_path + '_' + opt.split + '.json', 'w'))
