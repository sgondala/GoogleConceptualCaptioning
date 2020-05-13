# coding: utf-8

# In[1]:

# demo script for running CIDEr
import json
import argparse
from pydataformat.loadData import LoadData
from pyciderevalcap.eval import CIDErEvalCap as ciderEval
import sys

parser = argparse.ArgumentParser(description='Add val files')
parser.add_argument('--idf', type=str, default='coco-val-df')
parser.add_argument('--candName', type=str, default='')
parser.add_argument('--pathToData', type=str, default='data/')
parser.add_argument('--refName', type=str, default='coco_ref.json')
parser.add_argument('--resultFile', type=str, default='')
parser.add_argument('--load_from_params', action='store_true')

args = parser.parse_args()

pathToData = None
refName = None
candName = None
resultFile = None
df_mode = None

if args.load_from_params:
    config = json.loads(open('params.json', 'r').read())
    pathToData = config['pathToData']
    refName = config['refName']
    candName = config['candName']
    resultFile = config['resultFile']
    df_mode = config['idf']
else:
    pathToData = args.pathToData
    refName = args.refName
    candName = args.candName
    resultFile = args.resultFile
    df_mode = args.idf


# Print the parameters
print "Running CIDEr with the following settings"
print "*****************************"
print "Reference File:%s" % (refName)
print "Candidate File:%s" % (candName)
print "Result File:%s" % (resultFile)
print "IDF:%s" % (df_mode)
print "*****************************"

# In[2]:

# load reference and candidate sentences
loadDat = LoadData(pathToData)
gts, res = loadDat.readJson(refName, candName)


# In[3]:

# calculate cider scores
scorer = ciderEval(gts, res, df_mode)
# scores: dict of list with key = metric and value = score given to each
# candidate
scores = scorer.evaluate()


# In[7]:

# scores['CIDEr'] contains CIDEr scores in a list for each candidate
# scores['CIDErD'] contains CIDEr-D scores in a list for each candidate

with open(resultFile, 'w') as outfile:
    json.dump(scores, outfile)
