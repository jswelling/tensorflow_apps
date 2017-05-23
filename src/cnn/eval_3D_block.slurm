#!/bin/bash -x
##SBATCH -N 1
##SBATCH -t 00:10:00
##SBATCH -p GPU-shared
##SBATCH --gres=gpu:p100:1

module load tensorflow/1.1.0_nogpu
source ${TENSORFLOW_ENV}/bin/activate

cd /home/richardz/src/tensorflow_apps/src/cnn
python ./eval.py \
       --run_once True \
       --network_pattern 3D_cnn \
       --data_dir /pylon2/sy4s8lp/richardz/fish_data/eval_block \
       --checkpoint_dir /pylon2/sy4s8lp/richardz/tf/logs/cnn_3D_20170522 \
       --log_dir /pylon2/sy4s8lp/richardz/tf/logs/cnn_eval_3D_block_20170522/ \
       --batch_size 141 \
       --read_threads 16 \
       --shuffle_size 10 \
       --num_examples 463
