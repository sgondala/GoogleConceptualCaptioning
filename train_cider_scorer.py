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

# Reproducibility
np.random.seed(42)
torch.manual_seed(42)
torch.cuda.manual_seed_all(42)
torch.backends.cudnn.benchmark = False
torch.backends.cudnn.deterministic = True

if torch.cuda.is_available():
    device = torch.device('cuda', 0)
else:
    device = torch.device('cpu')

dataset = CiderDataset('/srv/share2/sgondala/tmp/trainval_36/trainval_resnet101_faster_rcnn_genome_36.tsv', 'data/coco_generated_captions.json', 'data/coco_cider_scores.json', 'data/cocotalk_with_cc_vocab.json')
train, val, test = random_split(dataset, [60000, 10000, 10000])

train_dataloader = DataLoader(train, batch_size=32, shuffle=True)
val_dataloader = DataLoader(val, batch_size=32, shuffle=True)
test_dataloader = DataLoader(test, batch_size=32, shuffle=False)

pretrained_embeddings = torch.load('data/embedding_for_coco_only.pth')
model = CiderPredictor(pretrained=pretrained_embeddings).to(device)
optimizer = optim.Adam(model.parameters(), lr=3e-3)
criterion = nn.MSELoss()

for batch in train_dataloader:
    image_features, captions, lengths, y = batch
    out = model(image_features.to(device), captions.long().to(device))    
    loss = torch.sqrt(criterion(out, y))
    print(loss)
    loss.backward()
    nn.utils.clip_grad_norm_(model.parameters(), 2.0)
    optimizer.step()
    optimizer.zero_grad()
    model.zero_grad()