import numpy as np
import torch
from torch.utils.data import DataLoader
from nltk import word_tokenize
from CiderDataset import CiderDataset

def clips_cider_scores(cider_scores):
    # Convert 0 - 2 to 0 and 1
    scores = np.minimum(cider_scores, 2)
    scores = scores / 2.0
    return scores

def sigmoid(x):
  return 1 / (1 + np.exp(-x))

def get_single_cider_scores(cider_dataset, model, is_classification_model, classification_threshold):
    val_dataloader = DataLoader(cider_dataset, batch_size=10,shuffle=False)
    rewards = []
    
    model.eval()
    for batch in val_dataloader:
        features, spatials, image_mask, captions, _, input_mask, segment_ids, co_attention_mask, _ = batch
        _, vil_logit, _, _, _, _, _ = \
            model(captions.cuda(), features.cuda(), spatials.cuda(), segment_ids.cuda(), input_mask.cuda(), image_mask.cuda(), co_attention_mask.cuda())
        rewards += vil_logit.squeeze(-1).tolist()
    
    if is_classification_model:
        return (sigmoid(np.array(rewards)) < classification_threshold).astype(float)
    else:
        return np.array(rewards)

def get_self_critical_cider_reward_using_model(cider_dataset, model, captions_greedy, captions_gen, opt, length_of_output, is_classification_model, classification_threshold):
    # Assuming captions is a mix of both
    cider_dataset.captions = captions_greedy
    cider_greedy = get_single_cider_scores(cider_dataset, model, is_classification_model, classification_threshold)
    average_greedy_cider = cider_greedy.mean()
    # if not is_classification_model:
    #     cider_greedy = clips_cider_scores(cider_greedy)

    cider_dataset.captions = captions_gen
    cider_gen = get_single_cider_scores(cider_dataset, model, is_classification_model, classification_threshold)
    average_gen_cider = cider_gen.mean()
    # if not is_classification_model:
    #     cider_gen = clips_cider_scores(cider_gen)

    assert len(cider_gen) == len(cider_greedy)
    scores = (cider_gen - cider_greedy) * opt.cider_reward_weight

    # Scores = [10,20,30] - (3)
    # Reward = [[10,10,10...., 10],[20,20,.... ,20],[30,30,...30]] - 3*20
    rewards = np.repeat(scores[:, np.newaxis], length_of_output, 1)
    return rewards, average_greedy_cider, average_gen_cider