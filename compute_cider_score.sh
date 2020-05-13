#!/bin/bash

candName=$1
resultFile=$2

#source ~/anaconda3/etc/profile.d/conda.sh
#conda activate conceptual_captions

cd cider
cp ../eval_results/$candName data/
python2 cidereval.py --candName $candName --resultFile $resultFile
cp $resultFile ../eval_results/
