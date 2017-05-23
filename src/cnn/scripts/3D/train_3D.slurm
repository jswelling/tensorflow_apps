#!/bin/bash -x
#SBATCH -N 1
#SBATCH -t 10:00:00
#SBATCH -p GPU-shared
#SBATCH --gres=gpu:p100:1

module load tensorflow/1.1.0
source ${TENSORFLOW_ENV}/bin/activate

cd /home/richardz/src/tensorflow_apps/src/cnn
python ./train.py \
       --network_pattern 3D_cnn \
       --batch_size 100 \
       --data_dir /pylon2/sy4s8lp/richardz/fish_data/train \
       --log_dir /pylon2/sy4s8lp/richardz/tf/logs/cnn_3D_20170522/ \
       --num_epochs 0 \
       --read_threads 3 \
       --shuffle_size 100 \
       --num_examples 9000
