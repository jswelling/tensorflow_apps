#!/bin/bash -x
#SBATCH -N 1
#SBATCH -t 8:00:00  # running time is about 20 min per hundred epochs
#SBATCH -p GPU-small   # used to be GPU-shared, but that doesn't seem to work any more
#SBATCH --gres=gpu:p100:1
#SBATCH --ntasks-per-node 7

cd /pylon5/pscstaff/welling/git/tensorflow_apps/src/cnn

source bridges_setup.bash

python ./scan_block.py \
    --network_pattern two_layers_logits_to_binary \
    --batch_size 4 \
    --data_dir /home/welling/data/fish_cubes \
    --log_dir $WORK/tf/logs/cnn_train_${SLURM_JOB_ID} \
    --file_list /home/welling/data/fish_cubes_train_gp0.txt	\
    --verbose=False \
    --starting_snapshot $WORK/tf/logs/cnn_train_6126590cnn-35337 \
    --drop1 0.8 \
    --layers 6,48 \
    --data_block_path /pylon5/pscstaff/welling/pylon2_rescue/fish_stick_20190815/V4750_2150_04000-08999.vol \
    --data_block_dims 1024,1024,1024 \
    --data_block_offset 5083496448 \
    --scan_start 69,325,0 \
    --scan_size 41,41,41

#    --scan_size 41,41,41
#    --scan_start 128,128,0 \
#    --data_block_offset 314572800 \
