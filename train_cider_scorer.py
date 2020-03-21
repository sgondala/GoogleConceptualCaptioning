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

dataset = CiderDataset('/srv/share2/sgondala/tmp/trainval_36/trainval_resnet101_faster_rcnn_genome_36.tsv', 'data/coco_caps_all_images_2_models.json', 'data/coco_caps_all_images_2_models_cider_scores.json', 'data/cocotalk_with_cc_vocab.json')

total_length = len(dataset)

# train, val, test = random_split(dataset, [65000, 5000, 10000])
train, val, test = random_split(dataset, [total_length - 30000, 10000, 20000]) 

train_dataloader = DataLoader(train, batch_size=256, shuffle=True)
val_dataloader = DataLoader(val, batch_size=256, shuffle=True)
test_dataloader = DataLoader(test, batch_size=256, shuffle=False)

pretrained_embeddings = torch.load('data/embedding_for_coco_only.pth')
model = CiderPredictor(pretrained=pretrained_embeddings).to(device)
optimizer = optim.Adam(model.parameters(), lr=3e-5)
criterion = nn.MSELoss()

i = 0
j = 0
for epoch in range(50):
    print("Epoch ", epoch)
    model.train()
    for batch in train_dataloader:
        i += 1
        image_features, captions, lengths, y, _ = batch
        out = model(image_features.to(device), captions.long().to(device))    
        loss = torch.sqrt(criterion(out, y.to(device)))
        writer.add_scalar('Train loss', loss, i)
        loss.backward()
        nn.utils.clip_grad_norm_(model.parameters(), 2.0)
        optimizer.step()
        optimizer.zero_grad()
        model.zero_grad()
    model.eval()
    for batch in val_dataloader:
        j += 1
        image_features, captions, lengths, y, _ = batch
        out = model(image_features.to(device), captions.long().to(device))
        loss = torch.sqrt(criterion(out, y.to(device)))
        writer.add_scalar('Val loss', loss, j)
    torch.save(model, 'cider_model/model-' + str(epoch) + '.pth')
