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

"""Trains the network."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import time

from six.moves import xrange  # pylint: disable=redefined-builtin
import tensorflow as tf
#import tensorflow.python.debug as tf_debug

#import input_data
#from input_data import N_BALL_SAMPS, OUTERMOST_SPHERE_SHAPE
import input_data_from_list as input_data
from input_data_from_list import N_BALL_SAMPS, OUTERMOST_SPHERE_SHAPE
import topology

# Basic model parameters as external flags.
flags = tf.app.flags
FLAGS = flags.FLAGS
flags.DEFINE_string('network_pattern', 'outer_layer_cnn',
                   'A network pattern recognized in topology.py')
flags.DEFINE_float('learning_rate', 0.01,
                   'Initial learning rate.')
flags.DEFINE_integer('batch_size', 12, 'Batch size.  '
                     'Must divide evenly into the dataset sizes.')
flags.DEFINE_string('data_dir', '/home/welling/data/fish_cubes',
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
flags.DEFINE_integer('num_examples', 12,
                     'Number of examples used in a training epoch')
flags.DEFINE_string('starting_snapshot', '',
                    'Snapshot from the end of the previous run ("" for none)')
flags.DEFINE_boolean('check_numerics', False, 'If true, add and run check_numerics ops.')
flags.DEFINE_boolean('verbose', False, 'If true, print extra output.')
flags.DEFINE_string('snapshot_load', 'all',
                    'A comma-separated list of variable name prefixes to load from the snapshot.'
                    ' One or more of "all", "cnn", "classifier"')
flags.DEFINE_string('hold_constant', None,
                    'A comma-separated list of variable name prefixes to exclude from learning.'
                    ' One or more of "cnn", "classifier"')
flags.DEFINE_boolean('reset_global_step', False, 'If true, global_step restarts from zero')

def train():
    """Train fish_cubes for a number of steps."""
    # Get the sets of images and labels for training, validation, and
    # test
    if FLAGS.num_epochs:
        num_epochs = FLAGS.num_epochs
    else:
        num_epochs = None

    # Track global step across multiple iterations.  This is updated in
    # the optimizer.
    with tf.variable_scope('control'):
        global_step = tf.get_variable('global_step', 
                                      dtype=tf.int32, initializer=0, 
                                      trainable=False)
    
    # seed provides the mechanism to control the shuffling which takes place reading input
    seed = tf.placeholder(tf.int64, shape=())
    
    # Generate placeholders for the images and labels.
    iterator = input_data.input_pipeline_binary(FLAGS.data_dir,
                                                FLAGS.batch_size,
                                                fake_data=FLAGS.fake_data,
                                                num_epochs=num_epochs,
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

    # Build a Graph that computes predictions from the inference model.
    logits = topology.inference(images, FLAGS.network_pattern)
    
    # Add to the Graph the Ops for loss calculation.
    loss = topology.binary_loss(logits, labels)
    print('loss: ', loss)

    if FLAGS.check_numerics:
        check_numerics_op = tf.add_check_numerics_ops()
    else:
        check_numerics_op = tf.constant('not checked')

    var_pfx_map = {'cnn' : 'cnn/',
                   'classifier' : 'image_binary_classifier/'}

    vars_to_load = []  # empty list means load all
    if len(FLAGS.starting_snapshot) and FLAGS.snapshot_load:
        keys = FLAGS.snapshot_load.split(',')
        keys = [k.strip() for k in keys]
        if 'all' not in keys:
            assert all([k in var_pfx_map for k in keys]), 'unknown key to load: %s' % key
            for k in keys:
                vars_to_load.extend([v for v in tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES)
                                     if v.name.startswith(var_pfx_map[k])])
            if not FLAGS.reset_global_step:
                vars_to_load.append(global_step)

    vars_to_hold_constant = []
    if FLAGS.hold_constant is not None:
        keys = FLAGS.hold_constant.split(',')
        keys = [k.strip() for k in keys]
        assert all([k in var_pfx_map for k in keys]), 'unknown key to hold constant: %s' % key
        for k in keys:
            vars_to_hold_constant.extend([v for v in tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES)
                                          if v.name.startswith(var_pfx_map[k])])
    print('not subject to training: %s' % [v.name for v in vars_to_hold_constant])
        

    # Add to the Graph the Ops that calculate and apply gradients.
    train_op = topology.training(loss, FLAGS.learning_rate, exclude=vars_to_hold_constant)
    print('train (optimizer): ', train_op)

    # Create a saver for writing training checkpoints.
    saver = tf.train.Saver(max_to_keep=10)

    # Build the summary operation based on the TF collection of Summaries.
    summary_op = tf.summary.merge_all()

    # Create a session for running operations in the Graph.
    sess = tf.Session(config=tf.ConfigProto(log_device_placement=FLAGS.verbose))

    # Optionally restore from a checkpoint.  The right file to load seems to be
    # the one with extension '.index'
    if len(FLAGS.starting_snapshot) == 0:
        pass
    elif FLAGS.snapshot_load and vars_to_load:
        print('loading from checkpoint: %s' % [v.name for v in vars_to_load])
        partial_saver = tf.train.Saver(vars_to_load)
        partial_saver.restore(sess, FLAGS.starting_snapshot)
    else:
        print('loading all variables from checkpoint ')
        saver.restore(sess, FLAGS.starting_snapshot)

    # Create the graph, etc.
    if vars_to_load:
        vars_to_initialize = [v for v in tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES)
                              if v not in vars_to_load]
        print('vars being initialized: %s' % [v.name for v in vars_to_initialize])
        init_op = tf.variables_initializer(vars_to_initialize)
    elif len(FLAGS.starting_snapshot) == 0:
        # initialize everything
        print('initalizing all globals')
        init_op = tf.variables_initializer(tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES))
    else:
        vars_to_initialize = []
        if FLAGS.reset_global_step:
            vars_to_initialize.append(global_step)
        print('vars being initialized: %s' % [v.name for v in vars_to_initialize])
        init_op = tf.variables_initializer(vars_to_initialize)
    sess.run(init_op)

    # Instantiate a SummaryWriter to output summaries and the Graph.
    summary_writer = tf.summary.FileWriter(FLAGS.log_dir, sess.graph)

    loss_value = -1.0  # avoid a corner case where it is unset on error
    duration = 0.0     # ditto
    num_chk = None     # ditto
    #sess = tf_debug.LocalCLIDebugWrapperSession(sess)
    # Loop through training epochs
    for epoch in range(num_epochs):
        try:
            sess.run(iterator.initializer, feed_dict={seed: epoch})
            saver.save(sess, FLAGS.log_dir + 'cnn', global_step=global_step)
            last_save_epoch = 0

            while True:            
                # Run training steps or whatever
                start_time = time.time()
                _, loss_value, num_chk, _, gstp = sess.run([train_op, loss, check_numerics_op,
                                                            print_op, global_step])
                duration = time.time() - start_time

                # Write the summaries and print an overview fairly often.
                if ((gstp + 1) % 100 == 0 or gstp < 10):
                    # Print status to stdout.
                    print('Global step %d epoch %d: numerics = %s, batch mean loss = %.2f (%.3f sec)'
                          % (gstp, epoch, num_chk, loss_value.mean(), duration))
                    # Update the events file.
                    summary_str = sess.run(summary_op)
                    summary_writer.add_summary(summary_str, gstp)
                    summary_writer.flush()

                # Save a checkpoint periodically.
                if (epoch + 1) % 100 == 0 and epoch != last_save_epoch:
                    # If log_dir is /tmp/cnn/ then checkpoints are saved in that
                    # directory, prefixed with 'cnn'.
                    print('saving checkpoint at global step %d, epoch %s' % (gstp, epoch))
                    saver.save(sess, FLAGS.log_dir + 'cnn', global_step=global_step)
                    last_save_epoch = epoch

        except tf.errors.OutOfRangeError as e:
            print('Finished epoch {}'.format(epoch))
#         finally:
#             # When done, ask the threads to stop.
#             coord.request_stop()
#             print('Final Step %d: numerics = %s, loss = %.2f (%.3f sec)'
#                   % (step, num_chk, loss_value, duration))
#             summary_str = sess.run(summary_op, num_chk)
#             summary_writer.add_summary(summary_str, step)
#             summary_writer.flush()

        # Wait for threads to finish.
#        coord.join(threads, stop_grace_period=10)

    print('Final Step %d: numerics = %s, batch mean loss = %.2f (%.3f sec)'
          % (gstp, num_chk, loss_value.mean(), duration))
    try:
        summary_str = sess.run(summary_op)
        summary_writer.add_summary(summary_str, step)
        summary_writer.flush()
    except tf.errors.OutOfRangeError as e:
        print('No final summary to write')

    sess.close()

def main(_):
    if tf.gfile.Exists(FLAGS.log_dir):
        tf.gfile.DeleteRecursively(FLAGS.log_dir)
    tf.gfile.MakeDirs(FLAGS.log_dir)
    train()

if __name__ == '__main__':
    tf.app.run()
