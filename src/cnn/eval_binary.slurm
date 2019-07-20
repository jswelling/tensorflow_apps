#!/bin/bash -x
#SBATCH -N 1
#SBATCH -t 1:00:00  # running time is about 20 min per hundred epochs
#SBATCH -p GPU-small   # used to be GPU-shared, but that doesn't seem to work any more
#SBATCH --gres=gpu:p100:1
#SBATCH --ntasks-per-node 7

myname=`whoami`
groupid=syr54jp
WORK=/pylon5/${groupid}/${myname}
#srcdir=${HOME}/tensorflow_apps/src/cnn
srcdir=/pylon5/pscstaff/welling/git/tensorflow_apps/src/cnn
datadir=/pylon5/pscstaff/welling/pylon2_rescue/fish_cube_links

file_list=${datadir}/../fish_cube_files_test_2.txt

nexamples=`cat $file_list | wc -l`

module load anaconda3
source activate /pylon5/pscstaff/welling/conda/envs/tfEnv

cd $srcdir

export PYTHONPATH=/pylon5/pscstaff/welling/git/tensorflow_apps/src:$PYTHONPATH
export LD_PRELOAD="/usr/lib64/libtcmalloc_minimal.so.4"

python ./eval_binary.py \
       --network_pattern two_layers_logits_to_binary \
       --batch_size 180 \
       --read_threads 3 \
       --data_dir $datadir \
       --file_list $file_list \
       --log_dir $WORK/tf/logs/cnn_train_${SLURM_JOB_ID} \
       --shuffle_size $nexamples \
       --num_examples $nexamples \
       --random_rotation \
       --starting_snapshot $WORK/tf/logs/cnn_train_6045675cnn-4404
