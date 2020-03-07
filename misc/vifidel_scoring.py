from __future__ import division
import numpy as np
from pyemd import emd
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.metrics import euclidean_distances
from gensim.models import KeyedVectors
import gensim.downloader as api
from text_unidecode import unidecode
import os
import math
from sklearn.feature_extraction.text import TfidfVectorizer
from nltk import word_tokenize
import torch

def get_vifidel_score(word_embedding, word_to_index, ground_truth_annotation, caption):
    '''
    Function that computes the score given detected objects, description
    without references. The word_embedding and vocablist refer to the memcached
    word vectors. Both word_embedding and vocablist can be obtained using the
    preprocess function.
    Parameters
    ----------
    word_embedding : nn.Embedding
    word_to_index : word_to_index
    ground_truth_annotation : String with all detected objects seperated by space
    caption : Caption.
    Returns
    -------
    score : Vifidel score
    '''
    for word in word_tokenize(ground_truth_annotation):
        if word.lower() not in word_to_index:
            ground_truth_annotation = ground_truth_annotation.replace(word, '')
    for word in word_tokenize(caption):
        if word.lower() not in word_to_index:
            caption = caption.replace(word, '')

    vc = CountVectorizer(stop_words='english').fit([ground_truth_annotation, caption])
    v_obj, v_desc = vc.transform([ground_truth_annotation, caption])

    v_obj = v_obj.toarray().ravel()
    v_desc = v_desc.toarray().ravel()
    temp = vc.get_feature_names()
    indices = torch.Tensor([word_to_index[w] for w in temp]).long()
    wvoc = word_embedding(indices)

    distance_matrix = euclidean_distances(wvoc)

    if np.sum(distance_matrix) == 0.0:
        return 1

    v_obj = v_obj.astype(np.double)
    v_desc = v_desc.astype(np.double)

    if v_obj.sum() == 0:
        return 0
    if v_desc.sum() == 0:
        return 0
 
    v_obj /= v_obj.sum()
    v_desc /= v_desc.sum()
    distance_matrix = distance_matrix.astype(np.double)
    distance_matrix /= distance_matrix.max()

    score = emd(v_obj, v_desc, distance_matrix)
    
    score = math.exp(-score)

    return score

def clip_vifidel_scores(vifidel_scores):
    scores = np.minimum(vifidel_scores, 0.9)
    scores = np.maximum(scores, 0.3)
    scores -= 0.3
    scores = scores / 0.6
    return scores

def get_vifidel_rewards(greedy_captions, gen_captions, ground_truth_annotations, glove_embedding, glove_word_to_ix, length_of_output):
    gen_vifidel_scores = []
    print(gen_captions)
    for entry in gen_captions:
        image_id = entry['image_id'].item()
        caption = entry['caption']
        gen_vifidel_scores.append(get_vifidel_score(glove_embedding.cpu(), glove_word_to_ix, ground_truth_annotations.get(str(image_id), ""), caption))

    greedy_vifidel_scores = []
    for entry in greedy_captions:
        image_id = entry['image_id'].item()
        caption = entry['caption']
        greedy_vifidel_scores.append(get_vifidel_score(glove_embedding, glove_word_to_ix, ground_truth_annotations.get(str(image_id), ""), caption))
    
    print(gen_vifidel_scores)
    print(greedy_vifidel_scores)
    gen_vifidel_scores = clip_vifidel_scores(np.array(gen_vifidel_scores))
    greedy_vifidel_scores = clip_vifidel_scores(np.array(greedy_vifidel_scores))

    assert len(gen_vifidel_scores) == len(greedy_vifidel_scores)
    
    scores = gen_vifidel_scores - greedy_vifidel_scores
    rewards = np.repeat(scores[:, np.newaxis], length_of_output, 1)
    return rewards





   
