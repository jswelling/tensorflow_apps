#! /bin/bash -x

#SBATCH -N1 -t 240 -p GPU-shared --gres=gpu:p100:1
########SBATCH -N1 -t 120 -p GPU

module load cuda/8.0 tensorflow/0.12.1
source ${TENSORFLOW_ENV}/bin/activate

cd /Users/richard/src/tensorflow_apps/src/joel
python ./fully_connected_feed.py \
       --network_pattern whole_ball_1_hidden \
       --batch_size 100 \
       --train_dir /pylon2/pscstaff/welling/fish_cube_links \
       --log_dir /Users/richard/src/tensorflow_apps/src/joel/logs/ball_net \
       --num_epochs 270 \
       --read_threads 3 \
       --shuffle_size 100 \
       --n_training_examples 9987
