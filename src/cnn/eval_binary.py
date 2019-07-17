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

    Args:
        saver: Saver.
        loss_op: Loss op.
    """
    init_op = tf.local_variables_initializer()
    sess.run(init_op)

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

