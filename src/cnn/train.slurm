#!/bin/bash -x
#SBATCH -N 1
#SBATCH -t 2:00:00  # running time is about 1 hour per hundred epochs
#SBATCH -p GPU-small   # used to be GPU-shared, but that doesn't seem to work any more
#SBATCH --gres=gpu:p100:1
#SBATCH --ntasks-per-node 7

myname=`whoami`
groupid=syr54jp
WORK=/pylon5/${groupid}/${myname}
#srcdir=${HOME}/tensorflow_apps/src/cnn
srcdir=/pylon5/pscstaff/welling/git/tensorflow_apps/src/cnn
datadir=/pylon5/pscstaff/welling/pylon2_rescue/fish_cube_links

file_list=${datadir}/../fish_cube_files_all.txt
nepochs=100

nexamples=`cat $file_list | wc -l`

module list
module load anaconda3
module list
source activate /pylon5/pscstaff/welling/conda/envs/tfEnv

cd $srcdir

python ./train.py \
       --network_pattern outer_layer_cnn \
       --batch_size 40 \
       --data_dir $datadir \
       --file_list $file_list \
       --log_dir $WORK/tf/logs/cnn_train_${SLURM_JOB_ID} \
       --num_epochs $nepochs \
       --shuffle_size $nexamples \
       --num_examples $nexamples
