#!/bin/bash

candName=$1
resultFile=$2

cd coco-caption
python eval_all_scores.py --candFile ../eval_results/$candName --resultFile ../eval_results/$resultFile
