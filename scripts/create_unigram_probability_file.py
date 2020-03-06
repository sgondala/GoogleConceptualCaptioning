from collections import Counter
from nltk import word_tokenize
import csv

captions = []
with open('/srv/datasets/conceptual_caption/DownloadConceptualCaptions/Train_GCC-training.tsv') as tsvfile:
    tsvreader = csv.reader(tsvfile, delimiter="\t")
    for line in tsvreader:
        captions.append(line[0])

vocab_dict = {}
for cap in captions:
    voc = word_tokenize(cap)
    for v in voc:
        if v not in vocab_dict:
            vocab_dict[v] = 1
        else:
            vocab_dict[v] += 1
            
vocab_freq = Counter(vocab_dict)
total = sum(vocab_freq.values())
unigram_prob = {k: v/total for k, v in vocab_freq.items()}