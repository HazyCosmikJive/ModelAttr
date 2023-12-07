#!/usr/bin/env bash
# e.g.
# bash vis_freq.sh vis_frequency.py 01_resnet18_1206 1

# please replace the ROOT here with your local file root
ROOT="../../../ModelAttr_release"
export PYTHONPATH=$ROOT:$PYTHONPATH

CONF=$1
EXPTAG=$2
DEVICE=$3

python $ROOT/tools/vis_frequency.py --config $CONF --exp_tag $EXPTAG --gpu $DEVICE