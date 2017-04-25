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

import input_data
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
tf.app.flags.DEFINE_boolean('run_once', False,
                            """Whether to run eval only once.""")
tf.app.flags.DEFINE_boolean('fake_data', False, 'If true, uses fake data '
                            'for unit testing.')
tf.app.flags.DEFINE_integer('read_threads', 2,
                            'Number of threads reading input files')
tf.app.flags.DEFINE_integer('shuffle_size', 8,
                            'Number of input data pairs to shuffle (min_dequeue)')
tf.app.flags.DEFINE_integer('batch_size', 8, 'Batch size.  '
                            'Must divide evenly into the dataset sizes.')


def eval_once(saver, summary_writer, loss_op, summary_op):
    """Run Eval once.

    Args:
        saver: Saver.
        summary_writer: Summary writer.
        loss_op: Loss op.
        summary_op: Summary op.
    """
    with tf.Session() as sess:
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

        # Start the queue runners.
        coord = tf.train.Coordinator()
        threads = tf.train.start_queue_runners(sess=sess, coord=coord)

        total_loss = 0.0
        step = 0

        # Number of examples evaluated
        examples = 0

        try:
            while not coord.should_stop():
                losses = sess.run([loss_op])
                print('loss @ step', step, '=', np.sum(losses) / FLAGS.batch_size)
                total_loss += np.sum(losses)
                step += 1
                examples += FLAGS.batch_size

        except Exception as e:  # pylint: disable=broad-except
            coord.request_stop(e)

        # Compute precision @ 1.
        loss = total_loss / examples
        print('%s: loss @ 1 = %.3f' % (datetime.now(), loss))

        coord.request_stop()
        coord.join(threads, stop_grace_period_secs=10)


def evaluate():
    """Eval for a number of steps."""
    with tf.Graph().as_default() as g:
        # Just eval once
        num_epochs = 1

        images, labels = input_data.input_pipeline(FLAGS.data_dir,
                                                   FLAGS.batch_size,
                                                   fake_data=FLAGS.fake_data,
                                                   num_epochs=num_epochs,
                                                   read_threads=FLAGS.read_threads,
                                                   shuffle_size=FLAGS.shuffle_size,
                                                   num_expected_examples=FLAGS.num_examples)

        logits = topology.inference(images, FLAGS.network_pattern)

        # Add to the Graph the Ops for loss calculation.
        loss = topology.loss(logits, labels)

        saver = tf.train.Saver()

        # Build the summary operation based on the TF collection of Summaries.
        summary_op = tf.summary.merge_all()
        summary_writer = tf.summary.FileWriter(FLAGS.log_dir, g)

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
