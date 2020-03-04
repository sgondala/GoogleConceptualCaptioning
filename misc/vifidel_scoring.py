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

def preprocess(wordvectors, binned, embed_type):
    '''
    Auxiliary function for storing and sorting vocabulary
    Parameters:
    ----------
    wordvectors : path to wordvectors file in word2vec format
    binned : True if wordvectors are in binned format
    embed_type: can be word2vec, glove, fasttext
    Returns:
    -------
    W : a matrix of wordvectors
    vocab_list : vocab_list where each item corresponds to the row of W
    Notes
    -----
    This function also saves W and vocab_list in the data folder.
    '''
    embed_type += '_embed'
    dat = 'data/{}.dat'.format(embed_type)
    vocab_file = 'data/{}.vocab'.format(embed_type)
    if not os.path.exists(dat):
        wv = KeyedVectors.load_word2vec_format(wordvectors, binary=binned)
        wv.init_sims(replace=False) # l2 normalizing all wvs
        wvshape = wv.vectors.shape
        # saving memmapped file and vocab for posterity
        fp = np.memmap(dat, dtype=np.double, mode='w+',
                shape=wv.vectors.shape)
        fp[:] = wv.vectors[:]
        counter = 0
        with open(vocab_file, 'w') as f:
            for _, w in sorted((voc.index, word) for word, voc in
                    wv.vocab.items()):
                print(unidecode(w),file=f)
        del fp, wv

    # freeing up precious memory
    wv = KeyedVectors.load_word2vec_format(wordvectors, binary=binned)
    W = np.memmap(dat, dtype=np.double, mode='r', shape=wv.vectors.shape)
    with open(vocab_file) as f:
        vocab_list = list(map(str.strip, f.readlines()))
        # print(vocab_list)
    # print(counter)
    return W, vocab_list

def objdesc(wvvecs, vocablist, objs, desc):
    '''
    Function that computes the score given detected objects, description
    without references. The wvvecs and vocablist refer to the memcached
    word vectors. Both wvvecs and vocablist can be obtained using the
    preprocess function.
    Parameters
    ----------
    wvvecs : memcached matrix of embeddings
    vocablist : vocab list of the embeddings
    objs : detected objects (txt file with one line for all detected
            objects)
    desc : description for evaluation (txt file)
    Returns
    -------
    score : Vifidel score
    '''

    vocabdict = {w: k for k, w in enumerate(vocablist)}
    for obj in objs.split():
        if obj.lower() not in vocabdict.keys():
            objs = objs.replace(obj, '')
    for d in desc.split():
        if d.lower() not in vocabdict.keys():
            desc = desc.replace(d, '')

    vc = CountVectorizer(stop_words='english').fit([objs, desc])
    v_obj, v_desc = vc.transform([objs, desc])

    v_obj = v_obj.toarray().ravel()
    v_desc = v_desc.toarray().ravel()
    temp = vc.get_feature_names()
    wvoc = wvvecs[[vocabdict[w] for w in temp]]

    distance_matrix = euclidean_distances(wvoc)

    if np.sum(distance_matrix) == 0.0:
        return float('inf')

    v_obj = v_obj.astype(np.double)
    v_desc = v_desc.astype(np.double)

    v_obj /= v_obj.sum()
    v_desc /= v_desc.sum()
    distance_matrix = distance_matrix.astype(np.double)
    distance_matrix /= distance_matrix.max()

    score = emd(v_obj, v_desc, distance_matrix)
    
    score = math.exp(-score)

    return score