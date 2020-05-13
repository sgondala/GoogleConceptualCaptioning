from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import torch
import torch.nn as nn
import torch.nn.functional as F

import numpy as np
import json
from json import encoder
import random
import string
import time
import os
import sys
import misc.utils as utils
import eval_multi
from nocaps.evalai import NocapsEvaluator

bad_endings = ['a','an','the','in','for','at','of','with','before','after','on','upon','near','to','is','are','am']
bad_endings += ['the']

def count_bad(sen):
    sen = sen.split(' ')
    if sen[-1] in bad_endings:
        return 1
    else:
        return 0

# def language_eval(dataset, preds, preds_n, eval_kwargs, cache_file_key):
#     model_id = eval_kwargs.get('id_language_eval', '')
#     eval_oracle = eval_kwargs.get('eval_oracle', 0)
    
#     import sys
#     sys.path.append("coco-caption")
#     annFile = 'coco-caption/annotations/captions_all2014.json'

#     from pycocotools.coco import COCO
#     from pycocoevalcap.eval import COCOEvalCap

#     if not os.path.isdir('eval_results'):
#         os.mkdir('eval_results')
#     cache_path = os.path.join('eval_results/', '.cache_'+ model_id + '_' + cache_file_key + '.json')

#     coco = COCO(annFile)
#     valids = coco.getImgIds() # 123287 - All coco data

#     preds_filt = [p for p in preds if p['image_id'] in valids]
#     print("Annotations file name " + cache_path)
#     print('Using %d/%d predictions' % (len(preds_filt), len(preds)))
#     json.dump(preds_filt, open(cache_path, 'w')) # serialize to temporary json file. Sigh, COCO API...

#     cocoRes = coco.loadRes(cache_path)
#     cocoEval = COCOEvalCap(coco, cocoRes)
#     cocoEval.params['image_id'] = cocoRes.getImgIds()
#     cocoEval.evaluate()

#     # create output dictionary
#     out = {}
#     for metric, score in cocoEval.eval.items():
#         out[metric] = score
    
#     out['CIDErArary'] = cocoEval.evalArray['CIDEr']
#     # if len(out['CIDErArary']) != len(preds):
#     #     final_dict_to_dump = {}
#     #     final_dict_to_dump['actual_preds'] = preds.tolist() if isinstance(preds, np.ndarray) else preds
#     #     final_dict_to_dump['filtered_preds'] = preds_filt.tolist() if isinstance(preds_filt, np.ndarray) else preds_filt
#     #     final_dict_to_dump['CIDErArary'] = out['CIDErArary'].tolist() if isinstance(out['CIDErArary'], np.ndarray) else out['CIDErArary']
#     #     final_dict_to_dump['valids'] = valids.tolist() if isinstance(valids, np.ndarray) else valids
#     #     filename_dump = 'eval_results/.cache_'+ model_id + '_' + cache_file_key + '_final_dump.json'
#     #     json.dump(final_dict_to_dump, open(filename_dump, 'w'))
#     #     print("Dumped to file ", filename_dump)
#     #     assert False

#     return out

def language_eval_for_nocaps(predictions):
    evaluator = NocapsEvaluator('val')
    evaluation_metrics = evaluator.evaluate(predictions)

    # for metric_name in evaluation_metrics:
    for metric_name in ['CIDEr']:    
        print(f"\t{metric_name}:")
        for domain in evaluation_metrics[metric_name]:
            print(f"\t\t{domain}:", evaluation_metrics[metric_name][domain])
    
    out = {}
    out['CIDEr'] = evaluation_metrics['CIDEr']['entire']
    out['CIDErArary'] = []
    return out

def language_eval_for_coco(predictions, cache_file_key):
    candName = 'eval_results/.cache_'+ cache_file_key + '.json'
    resultFile = 'eval_results/.cache_'+ cache_file_key + '_cider.json'

    json.dump(predictions, open(candName, 'w')) # serialize to temporary json file. Sigh, COCO API...
    
    bash_command = 'bash compute_cider_score.sh ' + candName + ' ' + resultFile 
    os.system(bash_command)
    CIDEr_array = json.load(open(resultFile))['CIDEr']
    out['CIDEr'] = np.array(CIDEr_array).mean()
    out['CIDErArary'] = CIDEr_array
    return out

def eval_split(model, crit, loader, eval_kwargs={}):
    verbose = eval_kwargs.get('verbose', True)
    verbose_beam = eval_kwargs.get('verbose_beam', 1)
    verbose_loss = eval_kwargs.get('verbose_loss', 1)
    num_images = eval_kwargs.get('num_images', eval_kwargs.get('val_images_use', -1))
    split = eval_kwargs.get('split', 'val')
    lang_eval = eval_kwargs.get('language_eval', 0)
    dataset = eval_kwargs.get('dataset', 'coco')
    beam_size = eval_kwargs.get('beam_size', 1)
    sample_n = eval_kwargs.get('sample_n', 1)
    sample_n_method = eval_kwargs.get('sample_n_method', 'sample')
    remove_bad_endings = eval_kwargs.get('remove_bad_endings', 0)
    nocaps_eval = eval_kwargs.get('nocaps_eval', False)
    
    print("Using nocaps ", nocaps_eval)
    os.environ["REMOVE_BAD_ENDINGS"] = str(remove_bad_endings) # Use this nasty way to make other code clean since it's a global configuration

    print("Split, Num images ", split, num_images)
    # Make sure in the evaluation mode
    model.eval()

    loader.reset_iterator(split)

    n = 0
    loss = 0
    loss_sum = 0
    loss_evals = 1e-8
    predictions = []
    n_predictions = [] # when sample_n > 1
    while True:
        data = loader.get_batch(split)
        n = n + loader.batch_size

        if data.get('labels', None) is not None and verbose_loss:
            # forward the model to get loss
            tmp = [data['fc_feats'], data['att_feats'], data['labels'], data['masks'], data['att_masks']]
            tmp = [_.cuda() if _ is not None else _ for _ in tmp]
            fc_feats, att_feats, labels, masks, att_masks = tmp

            with torch.no_grad():
                loss = crit(model(fc_feats, att_feats, labels, att_masks), labels[:,1:], masks[:,1:]).item()
            loss_sum = loss_sum + loss
            loss_evals = loss_evals + 1

        # forward the model to also get generated samples for each image
        # Only leave one feature for each image, in case duplicate sample
        tmp = [data['fc_feats'][np.arange(loader.batch_size) * loader.seq_per_img], 
            data['att_feats'][np.arange(loader.batch_size) * loader.seq_per_img],
            data['att_masks'][np.arange(loader.batch_size) * loader.seq_per_img] if data['att_masks'] is not None else None]
        tmp = [torch.Tensor(_).cuda() if _ is not None else _ for _ in tmp]
        fc_feats, att_feats, att_masks = tmp
        # forward the model to also get generated samples for each image
        with torch.no_grad():
            seq, seq_logprobs = model(fc_feats, att_feats, att_masks, opt=eval_kwargs, mode='sample')
            seq = seq.data
            entropy = - (F.softmax(seq_logprobs, dim=2) * seq_logprobs).sum(2).sum(1) / ((seq>0).float().sum(1)+1)
            perplexity = - seq_logprobs.gather(2, seq.unsqueeze(2)).squeeze(2).sum(1) / ((seq>0).float().sum(1)+1)
        
        # Print beam search
        if beam_size > 1 and verbose_beam:
            print("Printing beam search")
            for i in range(loader.batch_size):
                print('\n'.join([utils.decode_sequence(loader.get_vocab(), _['seq'].unsqueeze(0))[0] for _ in model.done_beams[i]]))
                print('--' * 10)
        sents = utils.decode_sequence(loader.get_vocab(), seq)

        for k, sent in enumerate(sents):
            # entry = {'image_id': data['infos'][k]['id'], 'caption': sent, 'perplexity': perplexity[k].item(), 'entropy': entropy[k].item()}
            try:
                entry = {'image_id': data['infos'][k]['id'], 'caption': sent}
            except:
                print("Error in ", k, " ", sent)
                continue
            if eval_kwargs.get('dump_path', 0) == 1:
                entry['file_name'] = data['infos'][k]['file_path']
            predictions.append(entry)
            # if eval_kwargs.get('dump_images', 0) == 1:
                # dump the raw image to vis/ folder
                # cmd = 'cp "' + os.path.join(eval_kwargs['image_root'], data['infos'][k]['file_path']) + '" vis/imgs/img' + str(len(predictions)) + '.jpg' # bit gross
                # print(cmd)
                # os.system(cmd)

            if verbose:
                pass
                # print('image %s: %s' %(entry['image_id'], entry['caption']))

        if sample_n > 1:
            tmp_eval_kwargs = eval_kwargs.copy()
            if sample_n_method == 'bs':
                # case 1 sample_n == beam size
                tmp_eval_kwargs.update({'beam_size': sample_n, 'group_size': 1}) # randomness from softmax
                with torch.no_grad():
                    model(fc_feats, att_feats, opt=tmp_eval_kwargs, mode='sample')
                for k in range(loader.batch_size):
                    _sents = utils.decode_sequence(loader.get_vocab(), torch.stack([model.done_beams[k][_]['seq'] for _ in range(beam_size)]))
                    for sent in _sents:
                        entry = {'image_id': data['infos'][k]['id'], 'caption': sent}
                        n_predictions.append(entry)
            # case 2 sample_max =0 temperature xx / gumbel / topk sampling
            elif sample_n_method == 'sample' or \
                 sample_n_method == 'gumbel' or \
                 sample_n_method.startswith('top'):
                if sample_n_method == 'sample':
                    tmp_sample_max = 0
                elif sample_n_method == 'gumbel':
                    tmp_sample_max = 2
                elif sample_n_method.startswith('top'):
                    tmp_sample_max = -int(sample_n_method[3:])
                tmp_eval_kwargs.update({'sample_max': tmp_sample_max, 'beam_size': 1}) # randomness from sample
                with torch.no_grad():
                    _seq, _sampleLogprobs = model(fc_feats, att_feats, att_masks, opt=tmp_eval_kwargs, mode='sample')
                _sents = utils.decode_sequence(loader.get_vocab(), _seq)
                for k, sent in enumerate(_sents):
                    entry = {'image_id': data['infos'][k // sample_n]['id'], 'caption': sent}
                    n_predictions.append(entry)
            else:
                # Use diverse beam search
                tmp_eval_kwargs.update({'beam_size': sample_n * beam_size, 'group_size': sample_n}) # randomness from softmax
                with torch.no_grad():
                    model(fc_feats, att_feats, opt=tmp_eval_kwargs, mode='sample')
                for k in range(loader.batch_size):
                    _sents = utils.decode_sequence(loader.get_vocab(), torch.stack([model.done_beams[k][_]['seq'] for _ in range(0, sample_n*beam_size, beam_size)]))
                    for sent in _sents:
                        entry = {'image_id': data['infos'][k]['id'], 'caption': sent}
                        n_predictions.append(entry)
            if verbose:
                for entry in sorted(n_predictions[-loader.batch_size * sample_n:], key=lambda x: x['image_id']):
                    print('image %s: %s' %(entry['image_id'], entry['caption']))
            
        
        ix0 = data['bounds']['it_pos_now']
        ix1 = data['bounds']['it_max']
        if num_images != -1:
            ix1 = min(ix1, num_images)
        for i in range(n - ix1):
            predictions.pop()

        if verbose:
            pass
            # print('evaluating validation preformance... %d/%d (%f)' %(ix0 - 1, ix1, loss))

        if data['bounds']['wrapped']:
            break
        if num_images >= 0 and n >= num_images:
            break

    lang_stats = None
    if lang_eval == 1:
        if not nocaps_eval:
            # lang_stats = language_eval(dataset, predictions, n_predictions, eval_kwargs, split)
            lang_stats = language_eval_for_coco(predictions, eval_kwargs.get('id_language_eval'))
        else:
            lang_stats = language_eval_for_nocaps(predictions)

    model.train()
    return loss_sum/loss_evals, predictions, lang_stats