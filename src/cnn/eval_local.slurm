#! /bin/bash -x

cd /Users/richard/src/tensorflow_apps/src/cnn
python ./eval.py \
       --run_once True \
       --network_pattern outer_layer_cnn \
       --data_dir /Users/richard/src/fish_cube_links_128 \
       --checkpoint_dir /Users/richard/src/tf/cnn/ \
       --log_dir /Users/richard/src/tf/cnn/eval/ \
       --batch_size 32 \
       --read_threads 1 \
       --shuffle_size 1 \
       --num_examples 128
