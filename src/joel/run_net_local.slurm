#! /bin/bash -x

cd /Users/richard/src/tensorflow_apps/src/joel
python ./fully_connected_feed.py \
  --batch_size 3 \
  --train_dir /Users/richard/src/fish_cube_links_12 \
  --log_dir /Users/richard/src/tensorflow_apps/src/joel/logs \
  --num_epochs 120 \
  --read_threads 1 \
  --shuffle_size 4 \
  --n_training_examples 12
