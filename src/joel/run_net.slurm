#! /bin/bash -x

#SBATCH -N1 -t 240 -p GPU-shared --gres=gpu:p100:1
########SBATCH -N1 -t 120 -p GPU 

module load cuda/8.0 tensorflow/0.12.1
source ${TENSORFLOW_ENV}/bin/activate

cd /pylon2/pscstaff/welling/fish_python
python ./fully_connected_feed.py \
  --batch_size 100 \
  --train_dir /pylon2/pscstaff/welling/fish_cube_links \
  --log_dir /pylon2/pscstaff/welling/fish_log \
  --num_epochs 270 \
  --read_threads 3 \
  --shuffle_size 100 \
  --n_training_examples 9987 \
  --starting_snapshot /pylon2/pscstaff/welling/fish_log-2999

