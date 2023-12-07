#!/usr/bin/env bash
# e.g.
# bash test.sh 01_resnet18.py 01_resnet18_1206 setting1 1

# please replace the ROOT here with your local file root
ROOT="../../../ModelAttr_release"
export PYTHONPATH=$ROOT:$PYTHONPATH

CONF=$1  # config
EXPTAG=$2  # workdir
TESTTAG=$3
DEVICE=$4

python $ROOT/test_classifier.py --config $CONF --exp_tag $EXPTAG --test_tag $TESTTAG --gpu $DEVICE