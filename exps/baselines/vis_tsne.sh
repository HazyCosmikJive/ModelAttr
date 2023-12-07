#!/usr/bin/env bash
# e.g.
# bash vis_tsne.sh 01_resnet18.py 01_resnet18_1206 setting1 1

# please replace the ROOT here with your local file root
ROOT="../../../ModelAttr_release"
export PYTHONPATH=$ROOT:$PYTHONPATH

CONF=$1
EXPTAG=$2
TESTTAG=$3
DEVICE=$4

python $ROOT/tools/vis_tsne.py --config $CONF --exp_tag $EXPTAG --test_tag $TESTTAG --gpu $DEVICE