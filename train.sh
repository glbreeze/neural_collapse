#!/bin/bash

#SBATCH --job-name=nc
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=2
#SBATCH --mem=80GB
#SBATCH --time=48:00:00
#SBATCH --gres=gpu
#SBATCH --partition=a100_1,a100_2,v100,rtx8000

# job info
LOSS=$1
EPS=$2

# Singularity path
ext3_path=/scratch/$USER/python36/python36.ext3
sif_path=/scratch/work/public/singularity/cuda11.2.2-cudnn8-devel-ubuntu20.04.sif

# start running
singularity exec --nv \
--overlay ${ext3_path}:ro \
${sif_path} /bin/bash -c "
source /ext3/env.sh
python main.py --dset cifar10 --model resnet18 --wd 54 --max_epochs 800 --scheduler ms --store_pt \
     --loss ${LOSS} --eps ${EPS} --batch_size 64 --seed 2021 --exp_name wd54_ms_${LOSS}${EPS}_b64
"
