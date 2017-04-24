#! /bin/bash -x

cd /Users/richard/src/tensorflow_apps/src/cnn
python ./train.py \
       --network_pattern outer_layer_cnn \
       --batch_size 32 \
       --data_dir /Users/richard/src/fish_cube_links_subset \
       --log_dir /Users/richard/src/tf/cnn/ \
       --num_epochs 300 \
       --read_threads 4 \
       --shuffle_size 4 \
       --num_examples 1024
