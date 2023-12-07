#!/usr/bin/env bash
# e.g.
# bash train.sh 01_resnet18.py 01_resnet18_1206 1

# please replace the ROOT here with your local file root
ROOT="../../../ModelAttr_release"
export PYTHONPATH=$ROOT:$PYTHONPATH

CONF=$1
EXPTAG=$2
DEVICE=$3

# CUDA_VISIBLE_DEVICES=${DEVICE} 

python $ROOT/train_classifier.py --config $CONF --exp_tag $EXPTAG --gpu $DEVICE