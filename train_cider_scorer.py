from dataloader_cider_eval import CiderDataset
from torch.utils.data import DataLoader
from models.cider_predictor import CiderPredictor
import torch


a = CiderDataset('/srv/share2/sgondala/tmp/trainval_36/trainval_resnet101_faster_rcnn_genome_36.tsv', 'data/coco_generated_captions.json', 'data/coco_cider_scores.json', 'data/cocotalk_with_cc_vocab.json')
dataloader = DataLoader(a, batch_size=32, shuffle=True)

pretrained_embeddings = torch.load('data/embedding_for_coco_only.pth')
model = CiderPredictor(pretrained=pretrained_embeddings)
for batch in dataloader:
    image_features, captions, lengths, y = batch
    out = model(image_features, captions.long())
    print(out)
    break
