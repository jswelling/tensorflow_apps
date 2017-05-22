#! /bin/bash -x

cd /Users/richard/src/tensorflow_apps/src/cnn
python ./train.py \
       --network_pattern 3D_cnn \
       --batch_size 32 \
       --data_dir /Users/richard/src/fish_data/train \
       --log_dir /Users/richard/src/tf/3D_cnn/train/ \
       --num_epochs 500 \
       --read_threads 4 \
       --shuffle_size 4 \
       --num_examples 896
