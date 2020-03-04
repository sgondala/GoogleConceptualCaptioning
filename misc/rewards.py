from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import numpy as np
import time
from collections import OrderedDict
import torch

import sys
sys.path.append("cider")
from pyciderevalcap.ciderD.ciderD import CiderD
sys.path.append("coco-caption")
from pycocoevalcap.bleu.bleu import Bleu
from torch.utils.data import DataLoader
from nltk import word_tokenize

CiderD_scorer = None
Bleu_scorer = None
#CiderD_scorer = CiderD(df='corpus')

def init_scorer(cached_tokens):
    global CiderD_scorer
    CiderD_scorer = CiderD_scorer or CiderD(df=cached_tokens)
    global Bleu_scorer
    Bleu_scorer = Bleu_scorer or Bleu(4)

def array_to_str(arr):
    out = ''
    for i in range(len(arr)):
        out += str(arr[i]) + ' '
        if arr[i] == 0:
            break
    return out.strip()

def get_self_critical_reward(greedy_res, data_gts, gen_result, opt):
    batch_size = gen_result.size(0)# batch_size = sample_size * seq_per_img
    seq_per_img = batch_size // len(data_gts)

    res = OrderedDict()
    
    gen_result = gen_result.data.cpu().numpy()
    greedy_res = greedy_res.data.cpu().numpy()
    for i in range(batch_size):
        res[i] = [array_to_str(gen_result[i])]
    for i in range(batch_size):
        res[batch_size + i] = [array_to_str(greedy_res[i])]

    gts = OrderedDict()
    for i in range(len(data_gts)):
        gts[i] = [array_to_str(data_gts[i][j]) for j in range(len(data_gts[i]))]

    res_ = [{'image_id':i, 'caption': res[i]} for i in range(2 * batch_size)]
    res__ = {i: res[i] for i in range(2 * batch_size)}
    gts = {i: gts[i % batch_size // seq_per_img] for i in range(2 * batch_size)}
    if opt.cider_reward_weight > 0:
        _, cider_scores = CiderD_scorer.compute_score(gts, res_)
        print('Cider scores:', _)
    else:
        cider_scores = 0
    if opt.bleu_reward_weight > 0:
        _, bleu_scores = Bleu_scorer.compute_score(gts, res__)
        bleu_scores = np.array(bleu_scores[3])
        print('Bleu scores:', _[3])
    else:
        bleu_scores = 0
    scores = opt.cider_reward_weight * cider_scores + opt.bleu_reward_weight * bleu_scores

    scores = scores[:batch_size] - scores[batch_size:]

    rewards = np.repeat(scores[:, np.newaxis], gen_result.shape[1], 1)

    return rewards

def get_log_probability(caption, language_model_tokenizer, language_model):
    # TODO: Batch?
    # input_ids = torch.tensor(language_model_tokenizer.encode(caption, add_special_tokens=True)).unsqueeze(0) # Batch size 1
    tokenize_input = language_model_tokenizer.tokenize(caption)
    
    # TODO: Check start vs end of token
    #50256 is the token_id for <|endoftext|>
    tensor_input = torch.tensor([[50256] + language_model_tokenizer.convert_tokens_to_ids(tokenize_input)])
    with torch.no_grad():
        outputs = language_model(tensor_input, labels=tensor_input)
        _, logits = outputs[:2]
    
    lp = 0.0
    
    for i in range(len(tokenize_input)):
        masked_index = i
        predicted_score = logits[0, masked_index]
        predicted_prob = torch.nn.functional.softmax(predicted_score)
        lp += np.log(predicted_prob[language_model_tokenizer.convert_tokens_to_ids([tokenize_input[i]])[0]])
    
    return lp

def get_slor_score(unigram_prob_dict, logprob, caption):
    uni = 0.0
    caption = caption.strip('.')
    for w in word_tokenize(caption):
        if not w.lower()[-1].isalpha():
            w = w.lower()[:-1]
        if w.lower() not in unigram_prob_dict.keys():
            return -1
        uni += np.log(unigram_prob_dict[w.lower()])
    n = len(caption.split())
    return ((logprob - uni) / n)

def get_single_cider_scores(cider_dataset, model):
    val_dataloader = DataLoader(cider_dataset, batch_size=10,shuffle=False)
    rewards = []
    
    model.eval()
    for batch in val_dataloader:
        features, spatials, image_mask, captions, _, input_mask, segment_ids, co_attention_mask, _ = batch
        _, vil_logit, _, _, _, _, _ = \
            model(captions.cuda(), features.cuda(), spatials.cuda(), segment_ids.cuda(), input_mask.cuda(), image_mask.cuda(), co_attention_mask.cuda())
        rewards += vil_logit.squeeze(-1).tolist()
    
    return np.array(rewards)

def get_self_critical_cider_reward_using_model(cider_dataset, model, captions_greedy, captions_gen, opt, length_of_output):
    
    from CiderDataset import CiderDataset
    
    # Assuming captions is a mix of both
    cider_dataset.captions = captions_greedy
    cider_greedy = get_single_cider_scores(cider_dataset, model)

    cider_dataset.captions = captions_gen
    cider_gen = get_single_cider_scores(cider_dataset, model)

    assert len(cider_gen) == len(cider_greedy)
    scores = (cider_gen - cider_greedy) * opt.cider_reward_weight

    rewards = np.repeat(scores[:, np.newaxis], length_of_output, 1)
    return rewards

def get_self_critical_slor_reward_using_model(language_model, language_model_tokenizer, captions_greedy, captions_gen, unigram_prob_dict, length_of_output):
    gen_slor_scores = []
    greedy_slor_scores = []

    for entry in captions_gen:
        caption = entry['caption']
        log_prob_of_sentence = get_log_probability(caption, language_model_tokenizer, language_model)
        slor_score = get_slor_score(unigram_prob_dict, log_prob_of_sentence, caption)
        gen_slor_scores += slor_score
    
    for entry in captions_greedy:
        caption = entry['caption']
        log_prob_of_sentence = get_log_probability(caption, language_model_tokenizer, language_model)
        slor_score = get_slor_score(unigram_prob_dict, log_prob_of_sentence, caption)
        greedy_slor_scores += slor_score
    
    gen_slor_scores = np.array(gen_slor_scores)
    greedy_slor_scores = np.array(greedy_slor_scores)
    
    scores = gen_slor_scores - greedy_slor_scores
    rewards = np.repeat(scores[:, np.newaxis], length_of_output, 1)
    return rewards


def get_scores(data_gts, gen_result, opt):
    batch_size = gen_result.size(0)# batch_size = sample_size * seq_per_img
    seq_per_img = batch_size // len(data_gts)

    res = OrderedDict()
    
    gen_result = gen_result.data.cpu().numpy()
    for i in range(batch_size):
        res[i] = [array_to_str(gen_result[i])]

    gts = OrderedDict()
    for i in range(len(data_gts)):
        gts[i] = [array_to_str(data_gts[i][j]) for j in range(len(data_gts[i]))]

    res_ = [{'image_id':i, 'caption': res[i]} for i in range(batch_size)]
    res__ = {i: res[i] for i in range(batch_size)}
    gts = {i: gts[i // seq_per_img] for i in range(batch_size)}
    if opt.cider_reward_weight > 0:
        _, cider_scores = CiderD_scorer.compute_score(gts, res_)
        print('Cider scores:', _)
    else:
        cider_scores = 0
    if opt.bleu_reward_weight > 0:
        _, bleu_scores = Bleu_scorer.compute_score(gts, res__)
        bleu_scores = np.array(bleu_scores[3])
        print('Bleu scores:', _[3])
    else:
        bleu_scores = 0

    scores = opt.cider_reward_weight * cider_scores + opt.bleu_reward_weight * bleu_scores

    return scores
