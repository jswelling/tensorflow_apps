#! /bin/bash -x

cd /Users/richard/src/tensorflow_apps/src/joel
python ./fully_connected_feed.py \
  --batch_size 4 \
  --train_dir /Users/richard/src/fish_cube_links_subset \
  --log_dir ./logs \
  --num_epochs 1000 \
  --read_threads 1 \
  --shuffle_size 1 \
  --n_training_examples 1024
