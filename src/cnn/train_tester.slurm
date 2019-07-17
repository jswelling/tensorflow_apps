#!/bin/bash -x 
#SBATCH -N 1
#SBATCH -t 00:30:00
#SBATCH -p GPU-small
#xSBATCH --ntasks-per-node 7 
#SBATCH --gres=gpu:p100:1

module load anaconda3
source activate ~welling/.conda/envs/tfEnv

cd /pylon5/syr54jp/brl51/git/tensorflow_apps/src/cnn
python ./train.py \
       --network_pattern outer_layer_cnn \
       --batch_size 2 \
       --data_dir /pylon5/pscstaff/welling/pylon2_rescue/fish_cubes_group1\
       --log_dir /pylon5/syr54jp/brl51/tf/logs/cnn_20190604/ \
       --num_epochs 1 \
       --read_threads 3 \
       --shuffle_size 2 \
       --num_examples 3506
