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

FLAGS = tf.app.flags.FLAGS

tf.app.flags.DEFINE_string('network_pattern', 'outer_layer_cnn',
                           'A network pattern recognized in topology.py')
tf.app.flags.DEFINE_string('log_dir', '/tmp/eval',
                           """Directory where to write event logs.""")
tf.app.flags.DEFINE_string('data_dir', '',
                           """Directory for evaluation data""")
tf.app.flags.DEFINE_string('checkpoint_dir', '/tmp/train',
                           """Directory where to read model checkpoints.""")
tf.app.flags.DEFINE_integer('eval_interval_secs', 60 * 5,
                            """How often to run the eval.""")
tf.app.flags.DEFINE_integer('num_examples', 10000,
                            """Number of examples to run.""")
tf.app.flags.DEFINE_boolean('run_once', True,
                            """Whether to run eval only once.""")
tf.app.flags.DEFINE_boolean('fake_data', False, 'If true, uses fake data '
                            'for unit testing.')
tf.app.flags.DEFINE_integer('read_threads', 2,
                            'Number of threads reading input files')
tf.app.flags.DEFINE_integer('shuffle_size', 8,
                            'Number of input data pairs to shuffle (min_dequeue)')
tf.app.flags.DEFINE_integer('batch_size', 8, 'Batch size.  '
                            'Must divide evenly into the dataset sizes.')
tf.app.flags.DEFINE_boolean('verbose', False, 'If true, print extra output.')


def eval_once(saver, summary_writer, loss_op, summary_op):
    """Run Eval once.

    Args:
        saver: Saver.
        summary_writer: Summary writer.
        loss_op: Loss op.
        summary_op: Summary op.
    """
    init_op = tf.local_variables_initializer()
    sess.run(init_op)

    ckpt = tf.train.get_checkpoint_state(FLAGS.checkpoint_dir)
    if ckpt and ckpt.model_checkpoint_path:
        # Restores from checkpoint
        saver.restore(sess, ckpt.model_checkpoint_path)
        # Assuming model_checkpoint_path looks something like:
        #   /my-favorite-path/cifar10_train/model.ckpt-0,
        # extract global_step from it.
        global_step = ckpt.model_checkpoint_path.split('/')[-1].split('-')[-1]
    else:
        print('No checkpoint file found')
        return

    total_loss = 0.0
    step = 0

    # Number of examples evaluated
    examples = 0

    try:
        losses = sess.run([loss_op])
        print('loss @ step', step, '=', np.sum(losses) / FLAGS.batch_size)
        total_loss += np.sum(losses)
        step += 1
        examples += FLAGS.batch_size

        # Update the events file.
        summary_str = sess.run(summary_op)
        summary_writer.add_summary(summary_str, step)
        summary_writer.flush()

    except Exception as e:  # pylint: disable=broad-except
        coord.request_stop(e)

    # Compute precision @ 1.
    loss = total_loss / examples
    print('%s: loss @ 1 = %.3f' % (datetime.now(), loss))

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

    # Build a Graph that computes predictions from the inference model.
    logits = topology.inference(images, FLAGS.network_pattern)
    
    # Add to the Graph the Ops for loss calculation.
    loss = topology.binary_loss(logits, labels)

    saver = tf.train.Saver()

    # Build the summary operation based on the TF collection of Summaries.
    summary_op = tf.summary.merge_all()

    with tf.Session() as sess:
        summary_writer = tf.summary.FileWriter(FLAGS.log_dir, sess.graph)
    
        while True:
            eval_once(saver, summary_writer, loss, summary_op)
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
