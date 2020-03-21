from dataloader_cider_eval import CiderDataset
from torch.utils.data import DataLoader
from models.cider_predictor import CiderPredictor
import torch
import torch.nn as nn
import torch.optim as optim
from tqdm import tqdm
import argparse
from torch.utils.tensorboard import SummaryWriter
from torch.utils.data import random_split
import numpy as np
import json

# Reproducibility
np.random.seed(42)
torch.manual_seed(42)
torch.cuda.manual_seed_all(42)
torch.backends.cudnn.benchmark = False
torch.backends.cudnn.deterministic = True

writer = SummaryWriter()

if torch.cuda.is_available():
    device = torch.device('cuda', 0)
else:
    device = torch.device('cpu')

# dataset = CiderDataset('/srv/share2/sgondala/tmp/trainval_36/trainval_resnet101_faster_rcnn_genome_36.tsv', 'data/coco_generated_captions.json', 'data/coco_cider_scores.json', 'data/cocotalk_with_cc_vocab.json')

# Create image_id to domain map
nocaps_val_image_info_actual = open('data/nocaps_val_image_info_actual.json', 'r')
nocaps_val_image_info_actual = json.load(nocaps_val_image_info_actual)

nocaps_val_image_info = open('data/nocaps_val_image_info.json', 'r')
nocaps_val_image_info = json.load(nocaps_val_image_info)

open_image_id_to_actual_id = {}
for entry in nocaps_val_image_info['images']:
    open_image_id_to_actual_id[entry['open_images_id']] = entry['id']


image_id_to_domain = {}
for image_entry in nocaps_val_image_info_actual:
    image_id = open_image_id_to_actual_id[image_entry['open_image_id']]
    domain = str(image_entry['domain'])
    image_id_to_domain[int(image_id)] = str(domain)

# Done 

dataset = CiderDataset('/srv/share2/sgondala/tmp/trainval_36/nocaps_val_vg_detector_features_adaptive.h5', 'data/nocaps_conceptual_base_with_coco_finetune_1000_images_only_ascii.json', 'data/nocaps_cider_scores.json', 'data/nocapstalk_with_cc_vocab.json', True)

#train, val, test = random_split(dataset, [65000, 5000, 10000])

#train_dataloader = DataLoader(train, batch_size=256, shuffle=True)
#val_dataloader = DataLoader(val, batch_size=256, shuffle=True)
test_dataloader = DataLoader(dataset, batch_size=256, shuffle=False)


pretrained_embeddings = torch.load('data/embedding_for_coco_only.pth')
model = CiderPredictor(pretrained=pretrained_embeddings).to(device)
model = torch.load('cider_model/model-12.pth', map_location=device)
# optimizer = optim.Adam(model.parameters(), lr=3e-3)
criterion = nn.MSELoss()

actual_values = []
predicted_values = []
image_ids_list = []

j = 0
for batch in test_dataloader:
    j += 1
    image_features, captions, lengths, y, image_ids = batch
    image_ids_list += image_ids.tolist()
    actual_values += y.tolist()
    out = model(image_features.to(device), captions.long().to(device))
    predicted_values += out.tolist()
    loss = torch.sqrt(criterion(out, y.to(device)))
    # writer.add_scalar('Nocaps Test loss', loss, j)
    
out_file = {}
out_file['actual_values'] = actual_values
out_file['predicted_values'] = predicted_values
print("Total ", np.corrcoef(np.array(actual_values), np.array(predicted_values)))
json.dump(out_file, open('cider_for_nocaps_predicted.json', 'w'))

in_domain_indices = [i for i in range(len(image_ids_list)) if image_id_to_domain[image_ids_list[i]] == 'in-domain'] 
near_domain_indices = [i for i in range(len(image_ids_list)) if image_id_to_domain[image_ids_list[i]] == 'near-domain']
out_domain_indices = [i for i in range(len(image_ids_list)) if image_id_to_domain[image_ids_list[i]] == 'out-domain']

print("In ", len(in_domain_indices))
print("Near ", len(near_domain_indices))
print("Out ", len(out_domain_indices))

actual_values = np.array(actual_values)
predicted_values = np.array(predicted_values)


indomain_actual = actual_values[in_domain_indices]
indomain_predicted = predicted_values[in_domain_indices]
print("In domain vals ", np.corrcoef(indomain_actual, indomain_predicted))

neardomain_actual = actual_values[near_domain_indices]
neardomain_predicted = predicted_values[near_domain_indices]
print("Near domain vals ", np.corrcoef(neardomain_actual, neardomain_predicted))

outdomain_actual = actual_values[out_domain_indices]
outdomain_predicted = predicted_values[out_domain_indices]
print("Out domain vals ", np.corrcoef(outdomain_actual, outdomain_predicted))
