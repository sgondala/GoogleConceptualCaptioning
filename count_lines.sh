#!/bin/bash
head -2 $1 | tail -1 > $1_oneline
filesize=$(du -b $1 | cut -f -1)
linesize=$(du -b $1_oneline | cut -f -1)
rm $1_oneline
echo $(expr $filesize / $linesize)
