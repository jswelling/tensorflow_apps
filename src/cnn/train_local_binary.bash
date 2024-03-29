#!/bin/bash -x 

#source activate py3Env

export LD_PRELOAD="/usr/lib/libtcmalloc_minimal.so.4"

cd ${HOME}/git/tensorflow_apps/src/cnn

python ./train_binaryclassifier.py \
	--network_pattern two_layers_logits_to_binary \
	--batch_size 4 \
	--data_dir /home/welling/data/fish_cubes \
	--log_dir ${HOME}/git/tensorflow_apps/log/train_local_log \
	--num_epochs 5 \
    --shuffle_size 4 \
	--num_examples 9 \
	--file_list /home/welling/data/fish_cubes_train_gp0.txt	\
	--verbose=False \
	--random_rotation \
	--optimizer SGD \
	--drop1=0.9 \
	--drop2=0.9

	#--reset_global_step \
	#--snapshot_load='cnn,classifier' \
	#--hold_constant='classifier' \
	#--starting_snapshot=${HOME}/sshfshook/logs/cnn_train_6045675cnn-4404 \
	#--starting_snapshot=${HOME}/git/tensorflow_apps/log/train_local_logcnn-7 \
	#--check_numerics=True \
								
