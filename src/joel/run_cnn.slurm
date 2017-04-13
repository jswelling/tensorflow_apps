#!/bin/bash -x
#SBATCH -N 1
#SBATCH -t 10:00:00
#SBATCH -p GPU-shared
#SBATCH --gres=gpu:p100:1

module load cuda/8.0 tensorflow/0.12.1
source ${TENSORFLOW_ENV}/bin/activate

cd /home/richardz/src/tensorflow_apps/src/joel
python ./fully_connected_feed.py \
       --network_pattern outer_layer_cnn \
       --batch_size 100 \
       --train_dir /pylon2/pscstaff/welling/fish_cube_links \
       --log_dir /pylon2/sy4s8lp/richardz/tf/logs/cnn \
       --num_epochs 0 \
       --read_threads 3 \
       --shuffle_size 100 \
       --n_training_examples 9987
