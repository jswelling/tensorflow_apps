#!/bin/bash -x 

#source activate py3Env

export LD_PRELOAD="/usr/lib/libtcmalloc_minimal.so.4"

cd ${HOME}/git/tensorflow_apps/src/cnn

python ./scan_block.py \
	--network_pattern two_layers_logits_to_binary \
    --batch_size 4 \
	--data_dir /home/welling/data/fish_cubes \
	--log_dir ${HOME}/git/tensorflow_apps/log/train_local_log \
	--file_list /home/welling/data/fish_cubes_train_gp0.txt	\
	--verbose=False \
	--starting_snapshot ${HOME}/git/tensorflow_apps/log/train_local_logcnn-8 \
	--drop1 0.9 \
	--drop2 0.9

								