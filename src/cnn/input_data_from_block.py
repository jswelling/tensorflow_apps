from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os
import re

from six.moves import urllib
from six.moves import xrange  # pylint: disable=redefined-builtin
import tensorflow as tf
from tensorflow.python.framework import dtypes
import numpy as np
from util import parse_int_list
from constants import *

#tf.enable_eager_execution()

#AUTOTUNE = tf.data.experimental.AUTOTUNE

#module-specific command line flags
flags = tf.app.flags
FLAGS = flags.FLAGS

LABEL_TRUE = tf.constant([0.0, 1.0])
LABEL_FALSE = tf.constant([1.0, 0.0])

def generate_offsets():
    x_start, y_start, z_start = parse_int_list(FLAGS.scan_start, 3,
                                               low_bound=[0, 0, 0])
    x_wid, y_wid, z_wid = parse_int_list(FLAGS.scan_size, 3,
                                         low_bound=[1, 1, 1])
    for k in range(z_start, z_start+z_wid):
        for j in range(y_start, y_start+y_wid):
            for i in range(x_start, x_start+x_wid):
                yield (i, j, k)

def get_loc_iterator(data_dir, batch_size):
    ds = tf.data.Dataset.from_generator(generate_offsets,
                                        (tf.int32, tf.int32, tf.int32))
    return ds.batch(batch_size).make_initializable_iterator()


def get_subblock_edge_len():
    """
    Each subblock is a cube with this edge length
    """
    return 2 * RAD_PIXELS + 1


def get_full_block(data_block_offset):
    """
    Return an op (usually a tf.constant) giving the full input data block
    """
    # # Make some data
    # fish_xsz = 100
    # fish_ysz = 101
    # fish_zsz = 102
    # fish_stick = np.zeros([fish_xsz, fish_ysz, fish_zsz], dtype=np.int)
    # for k in range(fish_zsz):
    #     for j in range(fish_ysz):
    #         for i in range(fish_xsz):
    #             fish_stick[i, j, k] = i + 1000 * j + 1000000 * k
    # fish_stick_flat = fish_stick.flatten('F')
    assert FLAGS.data_block_path is not None, '--data_block_path is missing'
    assert FLAGS.data_block_dims is not None, '--data_block_dims is missing'
    fish_xsz, fish_ysz, fish_zsz = [int(k.strip()) 
                                    for k in FLAGS.data_block_dims.split(',')]
    assert fish_xsz > 0 and fish_ysz > 0 and fish_zsz > 0, 'bad block volume'
    print('data_block_dims: %d %d %d' % (fish_xsz, fish_ysz, fish_zsz))
    print('reading from <%s>' % FLAGS.data_block_path)
    with open(FLAGS.data_block_path, 'rb') as f:
        f.seek(data_block_offset)
        fish_stick_flat = np.fromfile(f, count=fish_xsz*fish_ysz*fish_zsz,
                                      dtype=np.uint8)
    print('fish_stick_flat: %s' % fish_stick_flat)
    tf_fish_stick = tf.constant(fish_stick_flat, dtypes.uint8, 
                                shape=[fish_zsz, fish_ysz, fish_xsz])
    print('Created fish stick constant: %s' % tf_fish_stick)
    return tf_fish_stick





def data_fnames_from_yaml_fname(yaml_path):
    dir_path, fn = os.path.split(yaml_path)
    words = fn[:-5].split('_')
    base = words[0]
    idx = int(words[2])
    if FLAGS.random_rotation:
        feature_fn = os.path.join(dir_path,'%s_ballSamp_%d.doubles' % (base, idx))
        label_fn = os.path.join(dir_path,'%s_edgeSamp_%d.doubles' % (base, idx))
    else:
        feature_fn = os.path.join(dir_path,'%s_rotBallSamp_%d.doubles' % (base, idx))
        label_fn = os.path.join(dir_path,'%s_rotEdgeSamp_%d.doubles' % (base, idx))
    return feature_fn, label_fn
    

def get_data_pairs(train_dir, file_list, fake_data=False,
                    num_expected_examples=None, seed=None):

    yamlRE = re.compile(r'.+_.+_[0123456789]+\.yaml')
    
    feature_flist = []
    label_flist = []
    
    if not fake_data:
        with open(file_list, 'rU') as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                yaml_path = line if os.path.isabs(line) else os.path.join(train_dir, line)
                feature_fn, label_fn = data_fnames_from_yaml_fname(yaml_path)
                feature_flist.append('%s' % (feature_fn))
                label_flist.append('%s' % (label_fn))

    assert len(feature_flist) == num_expected_examples, ('Found %s examples, expected %d'
                                                         % (len(feature_flist),
                                                            num_expected_examples))
    assert len(label_flist) == num_expected_examples, ('Found %s examples, expected %d'
                                                       % (len(label_flist),
                                                          num_expected_examples))
    assert len(label_flist) == len(feature_flist), ('Found %s labels, expected %s'
                                                    % (len(label_flist),
                                                       len(feature_flist)))

    print('get_data_pairs: len(feature_flist) =', len(feature_flist))
    print('get_data_pairs: feature_flist[:5] =', feature_flist[:5])
    print('get_data_pairs: label_flist[:5] =', label_flist[:5])

    with tf.control_dependencies([tf.print('get_data_pairs: New SEED: ', seed)]):
        ds = tf.data.Dataset.from_tensor_slices((feature_flist, label_flist))
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


def load_and_preprocess_image_binary(image_path, label_path):
    # read and preprocess feature file
    fString = tf.read_file(image_path, name='featureReadFile')
    fVals = tf.reshape(tf.decode_raw(fString,
                                     dtypes.float64,
                                     name='featureDecode'
                                     ),
                       [N_BALL_SAMPS])
    fVals = tf.cast(fVals, tf.float32, name='featureToFloat')
    regex = '.*empty[^\/]*'   # Must match the full string, end-to-end!
    flagVals = tf.strings.regex_full_match(label_path, regex)
    lVals = tf.where(flagVals,
                     #tf.constant([1.0, 0.0]), tf.constant([0.0, 1.0]))
                     LABEL_FALSE, LABEL_TRUE)

    return image_path, flagVals, fVals, lVals


def input_pipeline(train_dir, batch_size, fake_data=False, num_epochs=None,
                   read_threads=1, shuffle_size=100,
                   num_expected_examples=None, seed=None):
    ds = get_data_pairs(train_dir, FLAGS.file_list,
                        num_expected_examples=num_expected_examples,
                        seed=seed
                        )
    image_label_ds = ds.map(load_and_preprocess_image,
                            num_parallel_calls=read_threads)
    image_label_ds = image_label_ds.shuffle(buffer_size=num_expected_examples,
                                            seed=seed)
    #image_label_ds = image_label_ds.repeat()
    image_label_ds = image_label_ds.batch(batch_size)
    # `prefetch` lets the dataset fetch batches, in the background while the model is training.
    #image_label_ds = image_label_ds.prefetch(buffer_size=AUTOTUNE)
    #keras_ds = image_label_ds.map(return_pair)
    iter = image_label_ds.make_initializable_iterator()
    return iter
    
def input_pipeline_binary(train_dir, batch_size, fake_data=False, num_epochs=None,
                          read_threads=1, shuffle_size=100,
                          num_expected_examples=None, seed=None):
        
    ds = get_data_pairs(train_dir, FLAGS.file_list,
                        num_expected_examples=num_expected_examples,
                        seed=seed
                        )
    image_label_ds = ds.map(load_and_preprocess_image_binary,
                            num_parallel_calls=read_threads)
    image_label_ds = image_label_ds.shuffle(buffer_size=num_expected_examples,
                                            seed=seed)
    #image_label_ds = image_label_ds.repeat()
    image_label_ds = image_label_ds.batch(batch_size)
    # `prefetch` lets the dataset fetch batches, in the background while the model is training.
    #image_label_ds = image_label_ds.prefetch(buffer_size=AUTOTUNE)
    #keras_ds = image_label_ds.map(return_pair)
    iter = image_label_ds.make_initializable_iterator()
    return iter
    
