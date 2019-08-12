# Single Node to compile
srun -N 1 -p gpu -t 144:00:00 --mem 32G  --exclusive --pty bash 

#export LD_LIBRARY_PATH=/sw/libs/cuda/10.1/extras/CUPTI/lib64:/sw/libs/cuda/10.1/lib64
