# Copyright 2015 The TensorFlow Authors. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================

"""Functions for downloading and reading MNIST data."""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os
import re

from six.moves import urllib
from six.moves import xrange  # pylint: disable=redefined-builtin
import tensorflow as tf
from tensorflow.python.framework import dtypes

N_BALL_SAMPS = 71709
OUTERMOST_SPHERE_SHAPE = [49, 97]

ball_shells = {0: (0, 0),
               1: (0.5, 2),
               2: (1.0, 3),
               3: (1.5, 4),
               4: (2.0, 5),
               5: (2.5, 6),
               6: (3.0, 7),
               7: (3.5, 8),
               8: (4.0, 10),
               9: (4.5, 11),
               10: (5.0, 12),
               11: (5.5, 13),
               12: (6.0, 14),
               13: (6.5, 15),
               14: (7.0, 16),
               15: (7.5, 18),
               16: (8.0, 19),
               17: (8.5, 20),
               18: (9.0, 21),
               19: (9.5, 22),
               20: (10.0, 23),
               21: (10.5, 24),
               22: (11.0, 26),
               23: (11.5, 27),
               24: (12.0, 28),
               25: (12.5, 29),
               26: (13.0, 30),
               27: (13.5, 31),
               28: (14.0, 32),
               29: (14.5, 34),
               30: (15.0, 35),
               31: (15.5, 36),
               32: (16.0, 37),
               33: (16.5, 38),
               34: (17.0, 39),
               35: (17.5, 40),
               36: (18.0, 42),
               37: (18.5, 43),
               38: (19.0, 44),
               39: (19.5, 45),
               40: (20.0, 46),
               41: (20.5, 47),
               42: (21.0, 48)
               }

def get_sphere_shapes():
    '''Return a list of tuples (nRows, nCols) of the shells in increasing order.
    '''

    layers = ball_shells.keys()
    layers.sort()

    def getDimensions(shell_idx):
        r, l = ball_shells[shell_idx]
        return (l + 1, 2 * l + 1)

    return list(map(getDimensions, layers))

def get_data_queues(train_dir, fake_data=False, shuffle=True, num_epochs=None,
                    num_expected_examples=None):

    print('get_data_queues: num_epochs =', num_epochs, type(num_epochs))

    yamlRE = re.compile(r'.+_.+_[0123456789]+\.yaml')
    recList = []
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

                recList.append('%s,%s' % (featureFName, labelFName))

    assert len(recList) == num_expected_examples, ('Found %s examples, expected %d'
                                                   % (len(recList),
                                                      num_expected_examples))

    print('get_data_queues: len(recList) =', len(recList))
    print('get_data_queues: recList[:10] =', recList[:10])


    featureNameT, labelNameT = tf.decode_csv(recList, [[""], [""]],
                                           name='decodeCSV')
    namePairQ = tf.train.slice_input_producer([featureNameT, labelNameT],
                                              shuffle=shuffle,
                                              num_epochs=num_epochs)
    return namePairQ


def read_pair_of_files(namePairQ):
    print('read_pair_of_files: namePairQ[0] =', namePairQ[0])
    print('read_pair_of_files: namePairQ[1] =', namePairQ[1])

    fString = tf.read_file(namePairQ[0], name='featureReadFile')
    fVals = tf.to_float(tf.reshape(tf.decode_raw(fString,
                                     dtypes.float64,
                                     name='featureDecode'),
                                   [N_BALL_SAMPS]),
                        name='featureToFloat')
    lString = tf.read_file(namePairQ[1], name='labelReadFile')
    lVals = tf.to_float(tf.decode_raw(lString, dtypes.float64,
                                      name='labelDecode'),
                        name='labelToFloat')
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


def input_pipeline(train_dir, batch_size, fake_data=False, num_epochs=None,
                   read_threads=1, shuffle_size=100, num_expected_examples=None):
    namePairQ= get_data_queues(train_dir, shuffle=True, num_epochs=num_epochs,
                               num_expected_examples=num_expected_examples)
    flPairList = [read_pair_of_files(namePairQ) for _ in range(read_threads)]

    print('flPairList:', flPairList[:10])

    # min_after_dequeue defines how big a buffer we will randomly sample
    #   from -- bigger means better shuffling but slower start up and more
    #   memory used.
    # capacity must be larger than min_after_dequeue and the amount larger
    #   determines the maximum we will prefetch.  Recommendation:
    #   min_after_dequeue + (num_threads + a small safety margin) * batch_size
#     min_after_dequeue = 10000
    min_after_dequeue = shuffle_size
    capacity = min_after_dequeue + 3 * batch_size
    example_batch, label_batch = tf.train.shuffle_batch_join(flPairList,
                                                        batch_size=batch_size,
                                                        capacity=capacity,
                                                        min_after_dequeue=min_after_dequeue)
    return example_batch, label_batch
