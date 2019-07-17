
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os
import re

from six.moves import urllib
from six.moves import xrange  # pylint: disable=redefined-builtin
import tensorflow as tf
from tensorflow.python.framework import dtypes

#tf.enable_eager_execution()

# initiate constant 
N_BALL_SAMPS = 71709
OUTERMOST_SPHERE_SHAPE = [49, 97]
#AUTOTUNE = tf.data.experimental.AUTOTUNE

# loop over all the file in the directory and return a shuffled list of file indices 
def get_data_pairs(train_dir, fake_data=False, shuffle=True, num_epochs=None,
                    num_expected_examples=None):

    print('get_data_queues: num_epochs =', num_epochs, type(num_epochs))

    yamlRE = re.compile(r'.+_.+_[0123456789]+\.yaml')
    featureFList = []
    labelFList = []
    idx_list = []
    if not fake_data:
        for fn in os.listdir(train_dir):
            if yamlRE.match(fn):
                words = fn[:-5].split('_')
                base = words[0]
                idx = int(words[2])
                idx_list.append(idx)
    print("1",idx_list)
    idx_list = np.array(idx_list)
    print("2",idx_list,idx_list[0])
    np.random.shuffle(idx_list)
    print("3",idx_list,type(idx_list))
    

    featureFList = list(map(map_image, idx_list))
    labelFList = list(map(map_label, idx_list))
    print("image list", len(featureFList),featureFList, "label list",len(labelFList),labelFList)
                

    assert len(featureFList) == num_expected_examples, ('Found %s examples, expected %d'
                                                   % (len(featureFList),
                                                      num_expected_examples))
    assert len(labelFList) == num_expected_examples, ('Found %s examples, expected %d'
                                                   % (len(labelFList),
                                                      num_expected_examples))
    assert len(labelFList) == len(featureFList), ('Found %s labels, expected %s'
                                                   % (len(labelFList),
                                                      len(featureFList)))

    print('get_data_queues: len(featureFList) =', len(featureFList))
    print('get_data_queues: featureList[:5] =', featureFList[:5])
    print('get_data_queues: labelList[:5] =', labelFList[:5])


    ds = tf.data.Dataset.from_tensor_slices((featureFList, labelFList))
    ds = ds.shuffle(num_expected_examples).repeat(num_epochs)
    return ds 
    
def load_and_preprocess_image(image_path, label_path):
    # read and preprocess feature file
    fString = tf.read_file(image_path, name='featureReadFile')
    fVals = tf.to_float(tf.reshape(tf.decode_raw(fString,
                                     dtypes.float64,
                                     name='featureDecode'),
                                   [N_BALL_SAMPS]),
                        name='featureToFloat')
    
    # return label based on image type 
    if 'empty' in image_path: 
        lVals = 0 
    else:
        lVals = 1

    print('read_pair_of_files: fVals, lVals =', fVals, lVals)

    return fVals, lVals


def input_pipeline(train_dir, batch_size, fake_data=False, num_epochs=None,
                   read_threads=1, shuffle_size=100, num_expected_examples=None):
    ds = get_data_pairs(train_dir, shuffle= True, num_epochs=num_epochs,num_expected_examples=num_expected_examples)
    image_label_ds = ds.map(load_and_preprocess_image)
    image_label_ds = image_label_ds.shuffle(buffer_size=num_expected_examples)
    image_label_ds = image_label_ds.repeat()
    image_label_ds = image_label_ds.batch(batch_size)
    # `prefetch` lets the dataset fetch batches, in the background while the model is training.
    #image_label_ds = image_label_ds.prefetch(buffer_size=AUTOTUNE)
    #keras_ds = image_label_ds.map(return_pair)
    iter = image_label_ds.make_one_shot_iterator()
    image_batch, label_batch = iter.get_next()
    return image_batch, label_batch 
    
