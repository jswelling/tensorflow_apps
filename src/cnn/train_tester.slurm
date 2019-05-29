#!/bin/bash -x 
#SBATCH -N 1
#SBATCH -t 00:30:00
#SBATCH -p GPU-small 
#SBATCH --gres=gpu:volta16:1

module load tensorflow/1.8_py3_gpu
source activate
export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:/opt/intel/compilers_and_libraries/linux/mkl/lib/intel64_lin

cd /home/minyuef/tensorflow_apps/src/cnn
python ./train.py \
       --network_pattern outer_layer_cnn \
       --batch_size 2 \
       --data_dir /pylon5/pscstaff/welling/pylon2_rescue/fish_cubes_group1\
       --log_dir /pylon5/syr54jp/minyuef/tf/logs/cnn_20190523/ \
       --num_epochs 1 \
       --read_threads 3 \
       --shuffle_size 2 \
       --num_examples 3506
