from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os
import re

from six.moves import urllib
from six.moves import xrange  # pylint: disable=redefined-builtin
import tensorflow as tf
from tensorflow.python.framework import dtypes

tf.enable_eager_execution()

# initiate constant 
N_BALL_SAMPS = 71709
OUTERMOST_SPHERE_SHAPE = [49, 97]

def get_data_pairs(train_dir, fake_data=False, shuffle=True, num_epochs=None,
                    num_expected_examples=None):

    print('get_data_queues: num_epochs =', num_epochs, type(num_epochs))

    yamlRE = re.compile(r'.+_.+_[0123456789]+\.yaml')
    featureFList = []
    labelFList = []
    if not fake_data:
        for fn in os.listdir(train_dir):
            if yamlRE.match(fn):
                words = fn[:-5].split('_')
                base = words[0]
                idx = int(words[2])
                featureFName = os.path.join(train_dir,
                                            '%s_rotBallSamp_%d.doubles' % (base, idx))
                labelFName = os.path.join(train_dir,
                                          '%s_rotEdgeSamp_%d.doubles' % (base, idx))

                featureFList.append('%s' % (featureFName))
                labelFList.append('%s' % (labelFName))

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
    # read and preprocess label file
    lString = tf.read_file(label_path, name='labelReadFile')
    lVals = tf.to_float(tf.decode_raw(lString, dtypes.float64,
                                      name='labelDecode'),
                        name='labelToFloat')
    # normalize label
    nRows, nCols = OUTERMOST_SPHERE_SHAPE
    nOuterSphere = nRows * nCols
    lVals = tf.cond(tf.less(tf.reduce_max(lVals), 1.0e-12),
                    lambda: tf.constant(1.0/float(nOuterSphere),
                                        dtype=dtypes.float32,
                                        shape=[nOuterSphere]),
                    lambda: tf.nn.l2_normalize(lVals, 0))
    lVals = tf.reshape(lVals, OUTERMOST_SPHERE_SHAPE)

    print('read_pair_of_files: fVals, lVals =', fVals, lVals)

    return fVals, lVals

ds = get_data_pairs(train_dir, shuffle= True, num_epochs=num_epochs,num_expected_examples=num_expected_examples)
image_label_ds = ds.map(load_and_preprocess_image)
image_label_ds