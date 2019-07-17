#!/bin/bash -x
#SBATCH -N 1
#SBATCH -t 5:00
#SBATCH -p GPU-small
#SBATCH --gres=gpu:p100:1
#SBATCH --ntasks-per-node 7

myname=`whoami`
groupid=syr54jp
WORK=/pylon5/${groupid}/${myname}
#srcdir=${HOME}/tensorflow_apps/src/cnn
srcdir=/pylon5/pscstaff/welling/git/tensorflow_apps/src/cnn

# The checkpoint name should be a full path like somestuff/cnn_train_5909338cnn-99
# This corresponds to a full path as given to the Saver.save() which created the
# checkpoint plus the epoch number when it was written.

cd $srcdir
python ./inspect_checkpoint.py \
	--checkpoint_filename $WORK/tf/logs/cnn_train_5909338cnn-99
