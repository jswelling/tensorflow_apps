#!/bin/bash -x
#SBATCH -N 1
#SBATCH -t 05:00:00
#SBATCH -p GPU-shared
#SBATCH --gres=gpu:p100:1

module load cuda/8.0 tensorflow/0.12.1
source ${TENSORFLOW_ENV}/bin/activate

cd /pylon2/sy4s8lp/richardz/src/tensorflow_apps/src/joel
python ./fully_connected_feed.py \
       --network_pattern whole_ball_1_hidden \
       --batch_size 100 \
       --train_dir /pylon2/pscstaff/welling/fish_cube_links \
       --log_dir /pylon2/sy4s8lp/richardz/tf/logs/ball_net \
       --num_epochs 0 \
       --read_threads 3 \
       --shuffle_size 100 \
       --n_training_examples 9987 \
       --starting_snapshot /pylon2/sy4s8lp/richardz/tf/logs/ball_net-26999

