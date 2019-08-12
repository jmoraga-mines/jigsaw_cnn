#!/usr/bin/env bash
# Single Node to compile

conda deactivate
conda activate brazil2
cd /scratch/sets/subt/brazil/project
srun -N 1 -p gpu -t 48:00:00 python learning_keras_inception.py > learn.500.epochs
#export LD_LIBRARY_PATH=/sw/libs/cuda/10.1/extras/CUPTI/lib64:/sw/libs/cuda/10.1/lib64
