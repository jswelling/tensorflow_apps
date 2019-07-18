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
from tensorflow.python import pywrap_tensorflow
 
import input_data_from_list as input_data
import topology
import harmonics
from constants import *

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
flags.DEFINE_boolean('random_rotation', False, 'use un-oriented data and apply random'
                     ' rotations to each data sample')

def get_cpt_name(var):
    nm = var.name
    parts = nm.split(':')
    assert len(parts) == 2, 'get_cpt_name failed for the variable %s' % nm
    return parts[0]

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

    if FLAGS.random_rotation:
        images, labels = harmonics.apply_random_rotation(images, labels)

    # Build a Graph that computes predictions from the inference model.
    logits = topology.inference(images, FLAGS.network_pattern)
    
    # Add to the Graph the Ops for loss calculation.
    loss = topology.binary_loss(logits, labels)
    print('loss: ', loss)

    if FLAGS.check_numerics:
        if FLAGS.random_rotation:
            sys.exit('check_numerics is not compatible with random_rotation')
        check_numerics_op = tf.add_check_numerics_ops()
    else:
        check_numerics_op = tf.constant('not checked')

    var_pfx_map = {'cnn' : 'cnn/',
                   'classifier' : 'image_binary_classifier/'}

    if len(FLAGS.starting_snapshot):
        keys = FLAGS.snapshot_load.split(',') if FLAGS.snapshot_load else ['all']
        keys = [k.strip() for k in keys]
        if 'all' in keys:
            vars_to_load = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES)
        else:
            assert all([k in var_pfx_map for k in keys]), 'unknown key to load: %s' % key
            vars_to_load = [global_step]
            for k in keys:
                vars_to_load.extend([v for v in tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES)
                                     if v.name.startswith(var_pfx_map[k])])
        if FLAGS.reset_global_step:
            vars_to_load.remove(global_step)
    else:
        vars_to_load = []

    vars_to_hold_constant = []  # empty list means hold nothing constant
    if FLAGS.hold_constant is not None:
        keys = [k.strip() for k in FLAGS.hold_constant.split(',')]
        assert all([k in var_pfx_map for k in keys]), 'unknown key to hold constant: %s' % key
        for k in keys:
            vars_to_hold_constant.extend([v for v in tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES)
                                          if v.name.startswith(var_pfx_map[k])])
    print('not subject to training: %s' % [v.name for v in vars_to_hold_constant])

    if FLAGS.starting_snapshot and len(FLAGS.starting_snapshot):
        vars_in_snapshot = [k for k in (pywrap_tensorflow.NewCheckpointReader(FLAGS.starting_snapshot)
                                        .get_variable_to_shape_map())]
    else:
        vars_in_snapshot = []
    vars_in_snapshot = set(vars_in_snapshot)
    print('vars in snapshot: %s' % vars_in_snapshot)

    optimizer = tf.train.AdamOptimizer(learning_rate=FLAGS.learning_rate, epsilon=0.1)
    train_op = topology.training(loss, FLAGS.learning_rate, exclude=vars_to_hold_constant,
                                 optimizer=optimizer)
    
    # Also load any variables the optimizer created for variables we want to load
    vars_to_load.extend([optimizer.get_slot(var, name) for name in optimizer.get_slot_names()
                         for var in vars_to_load])
    vars_to_load = [var for var in vars_to_load if var is not None]
    vars_to_load = list(set(vars_to_load))  # remove duplicates
    
    # Filter vars to load based on what is in the checkpoint
    in_vars = []
    out_vars = []
    for var in vars_to_load:
        if get_cpt_name(var) in vars_in_snapshot:
            in_vars.append(var)
        else:
            out_vars.append(var)
    if out_vars:
        print('WARNING: cannot load the following vars because they are not in the snapshot: %s'
              % [var.name for var in out_vars])
    if in_vars:
        print('loading from checkpoint: %s' % [var.name for var in in_vars])
        tf.train.init_from_checkpoint(FLAGS.starting_snapshot,
                                      {get_cpt_name(var): var for var in in_vars})


    # Try making histograms of *everything*
    for var in tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES):
        if var.name.startswith('cnn') or var.name.startswith('image_binary_classifier'):
            tf.summary.histogram(var.name, var)

    # Create a saver for writing training checkpoints.
    saver = tf.train.Saver(max_to_keep=10)

    # Build the summary operation based on the TF collection of Summaries.
    summary_op = tf.summary.merge_all()

    # Create a session for running operations in the Graph.
    sess = tf.Session(config=tf.ConfigProto(log_device_placement=FLAGS.verbose))

    # Create the graph, etc.
    # we either have no snapshot and must initialize everything, or we do have a snapshot
    # and have already set appropriate vars to be initialized from it
    init_op = tf.variables_initializer(tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES))
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
