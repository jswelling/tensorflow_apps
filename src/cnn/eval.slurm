#!/bin/bash -x
##SBATCH -N 1
##SBATCH -t 00:10:00
##SBATCH -p GPU-shared
##SBATCH --gres=gpu:p100:1

module load cuda/8.0 tensorflow/0.12.1
source ${TENSORFLOW_ENV}/bin/activate

cd /home/richardz/src/tensorflow_apps/src/cnn
python ./eval.py \
       --run_once True \
       --network_pattern outer_layer_cnn \
       --data_dir /pylon2/sy4s8lp/richardz/fish_data/eval_block \
       --checkpoint_dir /pylon2/sy4s8lp/richardz/tf/logs/cnn_20170424 \
       --log_dir /pylon2/sy4s8lp/richardz/tf/logs/cnn_eval_block/ \
       --batch_size 141 \
       --read_threads 16 \
       --shuffle_size 10 \
       --num_examples 463
