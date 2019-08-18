
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from datetime import datetime
import math
import time

import numpy as np
import tensorflow as tf

import input_data_from_list as input_data
from input_data_from_block import get_loc_iterator, get_subblock_edge_len, get_full_block
import topology
import harmonics
from constants import *

flags = tf.app.flags
FLAGS = flags.FLAGS

flags.DEFINE_string('network_pattern', 'outer_layer_cnn',
                    'A network pattern recognized in topology.py')
flags.DEFINE_string('log_dir', '/tmp/eval',
                    """Directory where to write event logs.""")
flags.DEFINE_string('data_dir', '',
                    """Directory for evaluation data""")
flags.DEFINE_string('starting_snapshot', None,
                    'Snapshot to evaluate')
flags.DEFINE_integer('eval_interval_secs', 60 * 5,
                     """How often to run the eval.""")
flags.DEFINE_integer('read_threads', 2,
                     'Number of threads reading input files')
flags.DEFINE_integer('batch_size', 8, 'Batch size.  '
                     'Must divide evenly into the dataset sizes.')
flags.DEFINE_boolean('verbose', False, 'If true, print extra output.')
flags.DEFINE_string('layers', '%d,%d' % (MAX_L//2,MAX_L),
                    'layers to include (depends on network-pattern; MAX_L=%d)' % MAX_L)
flags.DEFINE_float('drop1', 1.0, 'Fraction to keep in first drop-out layer. Default of'
                   ' 1.0 means no drop-out layer in this position')
flags.DEFINE_float('drop2', 1.0, 'Fraction to keep in second drop-out layer. Default of'
                   ' 1.0 means no drop-out layer in this position')
flags.DEFINE_string('data_block_path', None, 'full path to the data block')
flags.DEFINE_string('file_list', None,
                   'A filename containing a list of .yaml files to use for training')


def scan(iterator, saver, label_op, loss_op, accuracy_op, predicted_op):
    """Scan through the block locations specified by the iterator

    Args:
        saver: Saver.
        loss_op: Loss op.
    """
    
    epoch = 0
    with tf.Session() as sess:
        init_op = tf.local_variables_initializer()
        sess.run(init_op)
        print('restoring from snapshot: %s'
              % [var.name for var in tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES)])
        saver.restore(sess, FLAGS.starting_snapshot)
        global_step = FLAGS.starting_snapshot.split('/')[-1].split('-')[-1]
        try:
            sess.run(iter.initializer, feed_dict={})
            step = 0
            while True:
                print(sess.run([rslt])[0].shape)
                step += 1
        except tf.errors.OutOfRangeError as e:
            print('Finished evaluating epoch %d (%d steps)' % (epoch, step))

#     total_loss = 0.0
#     n_true_pos = 0
#     n_false_pos = 0
#     n_true_neg = 0
#     n_false_neg = 0
#     step = 0
# 
#     # Number of examples evaluated
#     examples = 0
#     accuracy_samples = []
# 
#     try:
#         sess.run(iterator.initializer, feed_dict={seed: 1234})
#         while True:
#             losses, accuracy, labels, predicted = sess.run([loss_op, accuracy_op, label_op, predicted_op])
#             print('loss @ step', step, '=', np.sum(losses) / FLAGS.batch_size)
#             # Test model and check accuracy
#             print('Test Accuracy:', accuracy)
#             accuracy_samples.append(accuracy)
# 
#             total_loss += np.sum(losses)
#             step += 1
#             examples += labels.shape[0]
# 
#             #print('predicted:')
#             #print(predicted[0:5,:])
#             #print('labels:')
#             #print(labels[0:5,:])
#             for idx in range(labels.shape[0]):
#                 if labels[idx, 1]:  # label true
#                     if predicted[idx, 1]:
#                         n_true_pos += 1
#                     else:
#                         n_false_neg += 1
#                 else:               # label false
#                     if predicted[idx, 1]:
#                         n_false_pos += 1
#                     else:
#                         n_true_neg += 1
# 
# 
#     except tf.errors.OutOfRangeError as e:
#         print('Finished evaluation {}'.format(step))

#     # Compute precision @ 1.
#     loss = total_loss / examples
#     accuracyV = np.asarray(accuracy_samples)
#     print('%s: total examples @ 1 = %d' % (datetime.now(), examples))
#     print('%s: loss @ 1 = %.3f' % (datetime.now(), loss))
#     print('%s: overall accuracy %s +- %s' % (datetime.now(), np.mean(accuracyV),
#                                              np.std(accuracyV, ddof=1)))
#     print('%s: true positive @ 1 = %d' % (datetime.now(), n_true_pos))
#     print('%s: false positive @ 1 = %d' % (datetime.now(), n_false_pos))
#     print('%s: true negative @ 1 = %d' % (datetime.now(), n_true_neg))
#     print('%s: false negative @ 1 = %d' % (datetime.now(), n_false_neg))


def get_subblock_op(loc_iterator, blk_sz, full_block):
    blk_shape = tf.constant([blk_sz, blk_sz, blk_sz], dtype=tf.int64)
    x_off, y_off, z_off = loc_iterator.get_next()
    offset_mtx = tf.transpose(tf.reshape(tf.concat([x_off, y_off, z_off], 0), [3, -1]))
    rslt = tf.map_fn(lambda offset: tf.slice(tf_fish_stick, offset, blk_shape), offset_mtx)
    return rslt


def sample_single(double_cube, transformer, sig):
    return transformer.calcBallOfSamples(double_cube.reshape((edge_len, edge_len, edge_len)),
                                         sig=sig)


def sample_op(double_cube_mtx, edge_len):
    transformer = SHTransformer(edge_len, MAX_L)
    assert all([len == edge_len for len in double_cube_mtx.shape[1:]]), 'input is not appropriate cubes'
    batch_sz = double_cube_mtx.shape[0]
    double_cube_mtx = tf.reshape(double_cube_mtx, [batch_sz, edge_len*edge_len*edge_len])
    rslt = np.apply_along_axis(sample_single, 1, double_cube_mtx, transformer, sig=(1.0, -1.0, 1.0))
    return tf.convert_to_tensor(rslt.reshape((batch_sz, -1)))
    

def collect_ball_samples(double_cube_mtx, edge_len, read_threads):
    """
    Call python functions to interpolate samples from within the data subblocks.
    
    data_cub_mtx dims: [batch_sz, edge_len, edge_len, edge_len]
    """
    rslt = tf.py_function(lambda x: sample_op(x, edge_len), [double_cube_mtx],
                          tf.dtypes.float64, name='collect_ball_samples')
    return rslt


def evaluate():
    """Instantiate the network, then eval for a number of steps."""

    # seed provides the mechanism to control the shuffling which takes place reading input
    seed = tf.placeholder(tf.int64, shape=())
    
    # Generate placeholders for the images and labels.
    loc_iterator = get_loc_iter(FLAGS.data_dir, FLAGS.batch_size)

    edge_len = get_subblock_edge_len()
    subblock = get_subblock_op(loc_iterator, edge_len, get_full_block())

    images = collect_ball_samples(subblock, edge_len, read_threads=FLAGS.read_threads)

    # Build a Graph that computes predictions from the inference model.
    logits = topology.inference(images, FLAGS.network_pattern)
    
    # Set up some prediction statistics
    predicted = tf.round(tf.nn.sigmoid(logits))
    correct_pred = tf.equal(predicted, labels)
    accuracy = tf.reduce_mean(tf.cast(correct_pred, tf.float32))

    saver = tf.train.Saver()

    scan(iterator, saver, label_op, loss_op, accuracy_op, predicted_op)


def main(argv=None):  # pylint: disable=unused-argument
    # Should clean up directories here
    tf.gfile.MakeDirs(FLAGS.log_dir)
    evaluate()


if __name__ == '__main__':
    tf.app.run()