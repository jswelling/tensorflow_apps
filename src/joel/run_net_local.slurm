#! /bin/bash -x

cd /Users/richard/src/tensorflow_apps/src/joel
python ./fully_connected_feed.py \
  --batch_size 32 \
  --train_dir /Users/richard/src/fish_cube_links_subset \
  --log_dir /Users/richard/src/tensorflow_apps/src/joel/logs \
  --num_epochs 300 \
  --read_threads 4 \
  --shuffle_size 4 \
  --n_training_examples 1024
