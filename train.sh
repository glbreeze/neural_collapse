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
KL_TYPE=$3
KL_WT=$4

# Singularity path
ext3_path=/scratch/$USER/overlay-25GB-500K.ext3
sif_path=/scratch/lg154/cuda11.4.2-cudnn8.2.4-devel-ubuntu20.04.3.sif

# start running
singularity exec --nv \
--overlay ${ext3_path}:ro \
${sif_path} /bin/bash -c "
source /ext3/env.sh
python main.py --dset cifar10 --model resnet18 --wd 5e-4 --scheduler ms \
     --max_epochs 500 --batch_size 128 --lr 0.05 --log_freq 5 \
     --koleo_type ${KL_TYPE} --koleo_wt ${KL_WT} \
     --loss ${LOSS} --eps ${EPS}  --seed 2021 --exp_name ${LOSS}${EPS}_KL${KL_TYPE}${KL_WT}_B128
"
