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

def get_data_pairs(train_dir, fake_data=False, num_epochs=None,
                    num_expected_examples=None, seed=None):

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

    print('get_data_pairs: num_epochs =', num_epochs, type(num_epochs))
    print('get_data_pairs: len(featureFList) =', len(featureFList))
    print('get_data_pairs: featureList[:5] =', featureFList[:5])
    print('get_data_pairs: labelList[:5] =', labelFList[:5])

    with tf.control_dependencies([tf.print('get_data_pairs: New SEED: ', seed)]):
        ds = tf.data.Dataset.from_tensor_slices((featureFList, labelFList))
        ds = ds.shuffle(num_expected_examples, seed=seed)

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

    return image_path, label_path, fVals, lVals

def return_pair(image,label):
    return (image,label)

def input_pipeline(train_dir, batch_size, fake_data=False, num_epochs=None,
                   read_threads=1, shuffle_size=100, num_expected_examples=None,
                   seed=None):
    ds = get_data_pairs(train_dir,
                        num_epochs=num_epochs,
                        num_expected_examples=num_expected_examples,
                        seed=seed
                        )
    image_label_ds = ds.map(load_and_preprocess_image)
    image_label_ds = image_label_ds.shuffle(buffer_size=num_expected_examples)
    #image_label_ds = image_label_ds.repeat()
    image_label_ds = image_label_ds.batch(batch_size)
    # `prefetch` lets the dataset fetch batches, in the background while the model is training.
    #image_label_ds = image_label_ds.prefetch(buffer_size=AUTOTUNE)
    #keras_ds = image_label_ds.map(return_pair)
    iter = image_label_ds.make_initializable_iterator()
    return iter
    
