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

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from datetime import datetime
import math
import time

import numpy as np
import tensorflow as tf

import input_data_from_list as input_data
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
flags.DEFINE_integer('num_examples', 10000,
                     """Number of examples to run.""")
flags.DEFINE_boolean('run_once', True,
                     """Whether to run eval only once.""")
flags.DEFINE_boolean('fake_data', False, 'If true, uses fake data '
                     'for unit testing.')
flags.DEFINE_integer('read_threads', 2,
                     'Number of threads reading input files')
flags.DEFINE_integer('shuffle_size', 8,
                     'Number of input data pairs to shuffle (min_dequeue)')
flags.DEFINE_integer('batch_size', 8, 'Batch size.  '
                     'Must divide evenly into the dataset sizes.')
flags.DEFINE_boolean('verbose', False, 'If true, print extra output.')
flags.DEFINE_boolean('random_rotation', False, 'use un-oriented data and apply random'
                     ' rotations to each data sample')
flags.DEFINE_string('layers', '%d,%d' % (MAX_L//2,MAX_L),
                    'layers to include (depends on network-pattern; MAX_L=%d)' % MAX_L)
flags.DEFINE_float('drop1', 1.0, 'Fraction to keep in first drop-out layer. Default of'
                   ' 1.0 means no drop-out layer in this position')
flags.DEFINE_float('drop2', 1.0, 'Fraction to keep in second drop-out layer. Default of'
                   ' 1.0 means no drop-out layer in this position')


def eval_once(sess, iterator, saver, seed, label_op, loss_op, accuracy_op, predicted_op):
    """Run Eval once.

    Args:
        saver: Saver.
        loss_op: Loss op.
    """
    init_op = tf.local_variables_initializer()
    sess.run(init_op)

    print('restoring from snapshot: %s'
          % [var.name for var in tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES)])
    saver.restore(sess, FLAGS.starting_snapshot)
    global_step = FLAGS.starting_snapshot.split('/')[-1].split('-')[-1]
    # ckpt = tf.train.get_checkpoint_state(FLAGS.checkpoint_dir)
    # if ckpt and ckpt.model_checkpoint_path:
    #     # Restores from checkpoint
    #     saver.restore(sess, ckpt.model_checkpoint_path)
    #     # Assuming model_checkpoint_path looks something like:
    #     #   /my-favorite-path/cifar10_train/model.ckpt-0,
    #     # extract global_step from it.
    #     global_step = ckpt.model_checkpoint_path.split('/')[-1].split('-')[-1]
    # else:
    #     print('No checkpoint file found')
    #     return

    total_loss = 0.0
    n_true_pos = 0
    n_false_pos = 0
    n_true_neg = 0
    n_false_neg = 0
    step = 0

    # Number of examples evaluated
    examples = 0
    accuracy_samples = []

    try:
        sess.run(iterator.initializer, feed_dict={seed: 1234})
        while True:
            losses, accuracy, labels, predicted = sess.run([loss_op, accuracy_op, label_op, predicted_op])
            print('loss @ step', step, '=', np.sum(losses) / FLAGS.batch_size)
            # Test model and check accuracy
            print('Test Accuracy:', accuracy)
            accuracy_samples.append(accuracy)

            total_loss += np.sum(losses)
            step += 1
            examples += labels.shape[0]

            #print('predicted:')
            #print(predicted[0:5,:])
            #print('labels:')
            #print(labels[0:5,:])
            for idx in range(labels.shape[0]):
                if labels[idx, 1]:  # label true
                    if predicted[idx, 1]:
                        n_true_pos += 1
                    else:
                        n_false_neg += 1
                else:               # label false
                    if predicted[idx, 1]:
                        n_false_pos += 1
                    else:
                        n_true_neg += 1


    except tf.errors.OutOfRangeError as e:
        print('Finished evaluation {}'.format(step))

    # Compute precision @ 1.
    loss = total_loss / examples
    accuracyV = np.asarray(accuracy_samples)
    print('%s: total examples @ 1 = %d' % (datetime.now(), examples))
    print('%s: loss @ 1 = %.3f' % (datetime.now(), loss))
    print('%s: overall accuracy %s +- %s' % (datetime.now(), np.mean(accuracyV),
                                             np.std(accuracyV, ddof=1)))
    print('%s: true positive @ 1 = %d' % (datetime.now(), n_true_pos))
    print('%s: false positive @ 1 = %d' % (datetime.now(), n_false_pos))
    print('%s: true negative @ 1 = %d' % (datetime.now(), n_true_neg))
    print('%s: false negative @ 1 = %d' % (datetime.now(), n_false_neg))



def evaluate():
    """Instantiate the network, then eval for a number of steps."""

    # seed provides the mechanism to control the shuffling which takes place reading input
    seed = tf.placeholder(tf.int64, shape=())
    
    # Generate placeholders for the images and labels.
    iterator = input_data.input_pipeline_binary(FLAGS.data_dir,
                                                FLAGS.batch_size,
                                                fake_data=FLAGS.fake_data,
                                                num_epochs=1,
                                                read_threads=FLAGS.read_threads,
                                                shuffle_size=FLAGS.shuffle_size,
                                                num_expected_examples=FLAGS.num_examples,
                                                seed=seed)
    image_path, label_path, images, labels = iterator.get_next()

    if FLAGS.verbose:
        print_op = tf.print("images and labels this batch: ", 
                            image_path, label_path, labels)
    else:
        print_op = tf.constant('No printing')

    if FLAGS.random_rotation:
        images, labels = harmonics.apply_random_rotation(images, labels)

    # Build a Graph that computes predictions from the inference model.
    logits = topology.inference(images, FLAGS.network_pattern)
    
    # Add to the Graph the Ops for loss calculation.
    loss = topology.binary_loss(logits, labels)
    
    # Set up some prediction statistics
    predicted = tf.round(tf.nn.sigmoid(logits))
    correct_pred = tf.equal(predicted, labels)
    accuracy = tf.reduce_mean(tf.cast(correct_pred, tf.float32))

    saver = tf.train.Saver()

    with tf.Session() as sess:
    
        while True:
            eval_once(sess, iterator, saver, seed, labels, loss, accuracy, predicted)
            if FLAGS.run_once:
                break
            time.sleep(FLAGS.eval_interval_secs)


def main(argv=None):  # pylint: disable=unused-argument
    if tf.gfile.Exists(FLAGS.log_dir):
        tf.gfile.DeleteRecursively(FLAGS.log_dir)
    tf.gfile.MakeDirs(FLAGS.log_dir)
    evaluate()


if __name__ == '__main__':
    tf.app.run()
