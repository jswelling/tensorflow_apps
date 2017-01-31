from __future__ import print_function

import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
rng = np.random

# Parameters
learning_rate = 0.5
training_epochs = 100
display_step = 1

# Training Data
# train_X = numpy.asarray([3.3,4.4,5.5,6.71,6.93,4.168,9.779,6.182,7.59,2.167,
#                          7.042,10.791,5.313,7.997,5.654,9.27,3.1])
# train_Y = numpy.asarray([1.7,2.76,2.09,3.19,1.694,1.573,3.366,2.596,2.53,1.221,
#                          2.827,3.465,1.65,2.904,2.42,2.94,1.3])
x_data = np.random.rand(100).astype(np.float32)
y_data = x_data * 0.1 + 0.3

# # tf Graph Input
X = tf.placeholder("float")
Y = tf.placeholder("float")

# Set model weights
W = tf.Variable(tf.random_uniform([1], -1.0, 1.0))
b = tf.Variable(tf.zeros([1]))
y = W * x_data + b

# Minimize the mean squared errors.
loss = tf.reduce_mean(tf.square(y - y_data))
tf.scalar_summary('loss', loss)

# Gradient descent
optimizer = tf.train.GradientDescentOptimizer(learning_rate)
train = optimizer.minimize(loss)

# Build the summary operation based on the TF collection of Summaries.
summary_op = tf.merge_all_summaries()

# Initializing the variables
init = tf.initialize_all_variables()

# Launch the graph
with tf.Session() as sess:
    # Instantiate a SummaryWriter to output summaries and the Graph.
    summary_writer = tf.train.SummaryWriter('/tmp/regression_logs', sess.graph)

    sess.run(init)

    # Fit all training data
    for step in range(training_epochs):
        sess.run(train)

        # Display logs per epoch step
        if (step+1) % display_step == 0:
            summary_str = sess.run(summary_op)
            summary_writer.add_summary(summary_str, step)
            summary_writer.flush()

            c = sess.run(loss, feed_dict={X: x_data, Y:y_data})
            print("Epoch:", '%04d' % (step+1), "cost=", "{:.9f}".format(c), \
                "W=", sess.run(W), "b=", sess.run(b))

#     print("Optimization Finished!")
#     training_cost = sess.run(loss, feed_dict={X: x_data, Y: y_data})
#     print("Training cost=", training_cost, "W=", sess.run(W), "b=", sess.run(b), '\n')
# 
#     # Graphic display
#     plt.plot(x_data, y_data, 'ro', label='Original data')
#     plt.plot(x_data, sess.run(W) * x_data + sess.run(b), label='Fitted line')
#     plt.legend()
#     plt.show()
# 
#     # Testing example, as requested (Issue #2)
#     test_X = np.asarray([6.83, 4.668, 8.9, 7.91, 5.7, 8.7, 3.1, 2.1])
#     test_Y = np.asarray([1.84, 2.273, 3.2, 2.831, 2.92, 3.24, 1.35, 1.03])
# 
#     print("Testing... (Mean square loss Comparison)")
#     testing_cost = sess.run(
#         tf.reduce_mean(tf.square(y - y_data)),
#         feed_dict={X: test_X, Y: test_Y})  # same function as cost above
#     print("Testing cost=", testing_cost)
#     print("Absolute mean square loss difference:", abs(
#         training_cost - testing_cost))
# 
#     plt.plot(test_X, test_Y, 'bo', label='Testing data')
#     plt.plot(x_data, sess.run(W) * x_data + sess.run(b), label='Fitted line')
#     plt.legend()
#     plt.show()