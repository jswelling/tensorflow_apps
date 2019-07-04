#!/bin/bash -x 

#source activate py3Env

cd ${HOME}/git/tensorflow_apps/src/cnn
python ./train_binaryclassifier.py \
	--network_pattern outer_layer_logits_to_binary \
	--batch_size 4 \
	--data_dir /home/welling/data/fish_cubes \
	--log_dir ${HOME}/git/tensorflow_apps/log/train_local_log \
	--num_epochs 5 \
    --shuffle_size 4 \
	--num_examples 9 \
	--file_list /home/welling/data/fish_cubes_train_gp0.txt	\
	--verbose=False \
	--check_numerics=True \
	--starting_snapshot=${HOME}/git/tensorflow_apps/log/train_local_logcnn-7 \
	--snapshot_load='cnn' \
	--hold_constant='cnn' \
	--reset_global_step
	#--starting_snapshot=${HOME}/sshfshook/logs/cnn_train_5951751cnn-4439 \
		
