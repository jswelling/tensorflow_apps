#!/bin/bash -exf 

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
    --data_block_offset 314572800 \
    --scan_start 128,128,0 \
    --scan_size 8,8,8

#    --data_block_dims 1024,1024,4900
