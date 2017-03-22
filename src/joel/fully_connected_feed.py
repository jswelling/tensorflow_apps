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

"""Trains and Evaluates the MNIST network using a feed dictionary."""
# pylint: disable=missing-docstring
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import time

from six.moves import xrange  # pylint: disable=redefined-builtin
import tensorflow as tf
#import tensorflow.python.debug as tf_debug

import input_data
from input_data import N_BALL_SAMPS, OUTERMOST_SPHERE_SHAPE
import ball_net

# Basic model parameters as external flags.
flags = tf.app.flags
FLAGS = flags.FLAGS
flags.DEFINE_string('network_pattern', 'whole_ball_1_hidden',
                   'A network pattern recognized in ball_net.py')
flags.DEFINE_float('learning_rate', 0.01,
                   'Initial learning rate.')
flags.DEFINE_integer('batch_size', 12, 'Batch size.  '
                     'Must divide evenly into the dataset sizes.')
flags.DEFINE_string('train_dir', '/home/welling/data/fish_cubes',
                    'Directory to put the training data.')
flags.DEFINE_string('log_dir', '/home/welling/data/fish_logs',
                    'Directory to put logs and checkpoints.')
flags.DEFINE_boolean('fake_data', False, 'If true, uses fake data '
                     'for unit testing.')
flags.DEFINE_integer('num_epochs', 3030,  # about right for 3000 steps on small dataset
                     'Number of epochs (0 means unlimited)')
flags.DEFINE_integer('read_threads', 2,
                     'Number of threads reading input files')
flags.DEFINE_integer('shuffle_size', 12,
                     'Number of input data pairs to shuffle (min_dequeue)')
flags.DEFINE_integer('n_training_examples', 12,
                     'Number of examples used in a training epoch')
flags.DEFINE_string('starting_snapshot', '',
                    'Snapshot from the end of the previous run ("" for none)')

def placeholder_inputs(batch_size):
    """Generate placeholder variables to represent the input tensors.

    These placeholders are used as inputs by the rest of the model building
    code and will be fed from the downloaded data in the .run() loop, below.

    Args:
      batch_size: The batch size will be baked into both placeholders.

    Returns:
      images_placeholder: Images placeholder.
      labels_placeholder: Labels placeholder.
    """
    # Note that the shapes of the placeholders match the shapes of the full
    # image and label tensors, except the first dimension is now batch_size
    # rather than the full size of the train or test data sets.
    feature_ph = tf.placeholder(tf.float32,
                                shape=(batch_size, N_BALL_SAMPS))
    label_ph = tf.placeholder(tf.float32,
                              shape=(batch_size,
                                     OUTERMOST_SPHERE_SHAPE[0],
                                     OUTERMOST_SPHERE_SHAPE[1]))
    return feature_ph, label_ph


def do_eval(sess,
            eval_correct,
            feature_ph,
            label_ph,
            feature_input,
            label_input):
    """Runs one evaluation against the full epoch of data.

    Args:
      sess: The session in which the model has been trained.
      eval_correct: The Tensor that returns the number of correct predictions.
      feature_ph: The features placeholder.
      label_ph: The labels placeholder.
      feature_input, label_input: evaluation data.
    """
    print('  Some relevant eval output goes here')
#     # And run one epoch of eval.
#     true_count = 0  # Counts the number of correct predictions.
#     steps_per_epoch = FLAGS.n_training_examples // FLAGS.batch_size
#     num_examples = steps_per_epoch * FLAGS.batch_size
#     for step in xrange(steps_per_epoch):
#         feed_dict = {feature_ph: feature_input,
#                      label_ph: label_input}
#         true_count += sess.run(eval_correct, feed_dict=feed_dict)
#     precision = true_count / num_examples
#     print('  Num examples: %d  Num correct: %d  Precision @ 1: %0.04f' %
#           (num_examples, true_count, precision))


def run_training():
    """Train fish_cubes for a number of steps."""
    # Get the sets of images and labels for training, validation, and
    # test
    if FLAGS.num_epochs:
        num_epochs = FLAGS.num_epochs
    else:
        num_epochs = None

    # Tell TensorFlow that the model will be built into the default Graph.
    with tf.Graph().as_default():
        # Generate placeholders for the images and labels.
        featureBatch, labelBatch = input_data.input_pipeline(FLAGS.train_dir,
                                                             FLAGS.batch_size,
                                                             fake_data=FLAGS.fake_data,
                                                             num_epochs=num_epochs,
                                                             read_threads=FLAGS.read_threads,
                                                             shuffle_size=FLAGS.shuffle_size,
                                                             num_expected_examples=FLAGS.n_training_examples)
#         feature_ph, label_ph = placeholder_inputs(FLAGS.batch_size)
        feature_ph, label_ph = featureBatch, labelBatch

        # Build a Graph that computes predictions from the inference model.
        logits = ball_net.inference(feature_ph, FLAGS.network_pattern)

        # Add to the Graph the Ops for loss calculation.
        loss = ball_net.loss(logits, label_ph)

        # Add to the Graph the Ops that calculate and apply gradients.
        train_op = ball_net.training(loss, FLAGS.learning_rate)

        # Add the Op to compare the logits to the labels during evaluation.
        eval_correct = ball_net.evaluation(logits, label_ph)

        # Build the summary operation based on the TF collection of Summaries.
        summary_op = tf.summary.merge_all()

        # Create the graph, etc.
        init_op = tf.group(tf.global_variables_initializer(),
                           tf.local_variables_initializer())

        # Create a saver for writing training checkpoints.
        saver = tf.train.Saver()

        # Create a session for running operations in the Graph.
        sess = tf.Session(config=tf.ConfigProto(log_device_placement=True))

        # Instantiate a SummaryWriter to output summaries and the Graph.
        summary_writer = tf.summary.FileWriter(FLAGS.log_dir, sess.graph)

        # Initialize the variables (like the epoch counter).
        if len(FLAGS.starting_snapshot) == 0:
            sess.run(init_op)
        else:
            saver.restore(sess, FLAGS.starting_snapshot)

        #check_numerics_op = tf.add_check_numerics_ops()

        # Start input enqueue threads.
        coord = tf.train.Coordinator()
        threads = tf.train.start_queue_runners(sess=sess, coord=coord)

        #feed_dict = {feature_ph: featureBatch, label_ph: labelBatch}
        feed_dict = {}
        step = 0
        loss_value = -1.0  # avoid a corner case where it is unset on error
        duration = 0.0     # ditto
        num_chk = None     # ditto
        #sess = tf_debug.LocalCLIDebugWrapperSession(sess)
        try:
            while not coord.should_stop():
                # Run training steps or whatever
                start_time = time.time()
#                 num_chk, _, loss_value = sess.run([check_numerics_op, train_op, loss],
#                                                   feed_dict=feed_dict)
                _, loss_value = sess.run([train_op, loss])
                duration = time.time() - start_time
#                 with sess.as_default():
#                     print(featureBatch.eval())
#                     print(labelBatch.eval())

                # Write the summaries and print an overview fairly often.
                if ((step + 1) % 100 == 0
                    or step < 10
                    ):
                    # Print status to stdout.
                    print('Step %d: numerics = %s, loss = %.2f (%.3f sec)'
                          % (step, num_chk, loss_value, duration))
                    # Update the events file.
                    summary_str = sess.run(summary_op, feed_dict=feed_dict)
                    summary_writer.add_summary(summary_str, step)
                    summary_writer.flush()

                # Save a checkpoint and evaluate the model periodically.
                if (step + 1) % 1000 == 0:
                    saver.save(sess, FLAGS.log_dir, global_step=step)
                    # Evaluate against the training set.
                    print('Training Data Eval:')
                    do_eval(sess,
                            eval_correct,
                            feature_ph,
                            label_ph,
                            featureBatch,
                            labelBatch)
#                     # Evaluate against the validation set.
#                     print('Validation Data Eval:')
#                     do_eval(sess,
#                             eval_correct,
#                             images_placeholder,
#                             labels_placeholder,
#                             data_sets.validation)
#                     # Evaluate against the test set.
#                     print('Test Data Eval:')
#                     do_eval(sess,
#                             eval_correct,
#                             images_placeholder,
#                             labels_placeholder,
#                             data_sets.test)

                step += 1

        except tf.errors.OutOfRangeError as e:
            print('Done training -- epoch limit reached')
        finally:
            # When done, ask the threads to stop.
            coord.request_stop()
            print('Final Step %d: numerics = %s, loss = %.2f (%.3f sec)'
                  % (step, num_chk, loss_value, duration))
            summary_str = sess.run(summary_op, num_chk)
            summary_writer.add_summary(summary_str, step)
            summary_writer.flush()

        # Wait for threads to finish.
        coord.join(threads, stop_grace_period=10)
        sess.close()



def main(_):
    run_training()


if __name__ == '__main__':
    tf.app.run()
