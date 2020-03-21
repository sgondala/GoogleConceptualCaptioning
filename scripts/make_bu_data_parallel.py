from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os
import base64
import numpy as np
import csv
import sys
import zlib
import time
import mmap
import argparse
from tqdm import tqdm
import multiprocessing as mp

parser = argparse.ArgumentParser()

# output_dir
parser.add_argument('--downloaded_feats', default='data/bu_data', help='downloaded feature directory')
parser.add_argument('--output_dir', default='data/cocobu', help='output feature files')

args = parser.parse_args()

csv.field_size_limit(sys.maxsize)


FIELDNAMES = ['image_id', 'image_w','image_h','num_boxes', 'boxes', 'features']
# infile = 'trainval_resnet101_faster_rcnn_genome_36.tsv'
infile = '1000.tsv'

#infiles = ['trainval/karpathy_test_resnet101_faster_rcnn_genome.tsv',
#          'trainval/karpathy_val_resnet101_faster_rcnn_genome.tsv',\
#          'trainval/karpathy_train_resnet101_faster_rcnn_genome.tsv.0', \
#           'trainval/karpathy_train_resnet101_faster_rcnn_genome.tsv.1']

try:
    os.makedirs(args.output_dir+'_att')
    os.makedirs(args.output_dir+'_fc')
    os.makedirs(args.output_dir+'_box')
except:
    print("Errors in making files, probably already exist")

result_list = []
def log_result(result):
    # https://stackoverflow.com/questions/8533318/multiprocessing-pool-when-to-use-apply-apply-async-or-map
    result_list.append(result)

def process_file(filename, start, end):
    print("Called process_file ", start, end)
    with open(os.path.join(args.downloaded_feats, infile), "r+b") as tsv_in_file:
        print("read file")
        reader = csv.DictReader(tsv_in_file, delimiter='\t', fieldnames = FIELDNAMES)
        linenumber = 0
        for item in reader:
            if linenumber < int(start):
                linenumber += 1
                continue
            if linenumber >= int(end):
                break
            item['image_id'] = int(item['image_id'])
            item['num_boxes'] = int(item['num_boxes'])
            for field in ['boxes', 'features']:
                item[field] = np.frombuffer(base64.decodestring(item[field]), 
                        dtype=np.float32).reshape((item['num_boxes'],-1))
            np.savez_compressed(os.path.join(args.output_dir+'_att', str(item['image_id'])), feat=item['features'])
            np.save(os.path.join(args.output_dir+'_fc', str(item['image_id'])), item['features'].mean(0))
            np.save(os.path.join(args.output_dir+'_box', str(item['image_id'])), item['boxes'])
            linenumber += 1
    print("Finished processing ", start, end)

def call_async():
    pool = mp.Pool(30)
    for i in range(30):
        pool.apply_async(process_file, args=(infile, i*33, i*33 + 33), callback = log_result)
    pool.close()
    pool.join()

call_async()    
