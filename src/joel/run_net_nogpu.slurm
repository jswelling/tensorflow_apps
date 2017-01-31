#! /bin/bash -x

#SBATCH -N1 -t 120 -p RM-shared 
##SBATCH -N1 -t 120 -p GPU-shared --gres=gpu:2
########SBATCH -N1 -t 120 -p GPU 

module load cuda/7.5 tensorflow/0.9.0_nogpu
source ${TENSORFLOW_ENV}/bin/activate

cd /pylon2/pscstaff/welling/fish_python
python ./fully_connected_feed.py \
  --batch_size 100 \
  --train_dir /pylon2/pscstaff/welling/fish_cube_links \
  --log_dir /pylon2/pscstaff/welling/fish_log \
  --num_epochs 300 \
  --read_threads 3 \
  --shuffle_size 100 \
  --n_training_examples 9987

