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

dataset = CiderDataset('/srv/share2/sgondala/tmp/trainval_36/trainval_resnet101_faster_rcnn_genome_36.tsv', 'data/coco_generated_captions.json', 'data/coco_cider_scores.json', 'data/cocotalk_with_cc_vocab.json')
train, val, test = random_split(dataset, [65000, 5000, 10000])

train_dataloader = DataLoader(train, batch_size=256, shuffle=True)
val_dataloader = DataLoader(val, batch_size=256, shuffle=True)
test_dataloader = DataLoader(test, batch_size=256, shuffle=False)

pretrained_embeddings = torch.load('data/embedding_for_coco_only.pth')
model = CiderPredictor(pretrained=pretrained_embeddings).to(device)
model = torch.load('checkpoints/cider_model_3e4_50epochs/model-49.pth')
# optimizer = optim.Adam(model.parameters(), lr=3e-3)
criterion = nn.MSELoss()

j = 0
for batch in test_dataloader:
    j += 1
    image_features, captions, lengths, y = batch
    out = model(image_features.to(device), captions.long().to(device))
    loss = torch.sqrt(criterion(out, y.to(device)))
    writer.add_scalar('Test loss', loss, j)