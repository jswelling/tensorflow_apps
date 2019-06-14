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

"""Builds the MNIST network.

Implements the inference/loss/training pattern for model building.

1. inference() - Builds the model as far as is required for running the network
forward to make predictions.
2. loss() - Adds to the inference model the layers required to generate loss.
3. training() - Adds to the loss model the Ops required to generate and
apply gradients.

This file is used by the various "fully_connected_*.py" files and not meant to
be run.
"""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import math

import numpy as np
import tensorflow as tf
from tensorflow.python.framework import dtypes

from input_data import N_BALL_SAMPS, OUTERMOST_SPHERE_SHAPE

def add_weights(self,shape):
    # a common method to create all sorts of weight connections
    # takes in shapes of previous and new layer as a list e.g.  
      [2,10]
    # starts with random values of that shape.
    return tf.Variable(tf.truncated_normal(shape=shape,stddev=0.05))


def add_biases(self,shape):
    # a common method to add create biases with default=0.05
    # takes in shape of the current layer e.g. x=10
    return tf.Variable(tf.constant(0.05, shape=shape))


# layer - takes in last layer.
# kernel - kernel size for convoluting on the image.
# input_shape - size of the input image.
# ouput_shape - size of the convoluted image.
# stride_size - determines kernel jump size
def conv_layer(self,layer, kernel, input_shape, output_shape, stride_size):
     weights = self.add_weights([kernel, kernel, input_shape,
     output_shape])
     biases = self.add_biases([output_shape])
     #stride=[image_jump,row_jump,column_jump,color_jump]=[1,1,1,1]
     stride = [1, stride_size, stride_size, 1]
     #does a convolution scan on the given image
     layer = tf.nn.conv2d(layer, weights, strides=stride, 
             padding='SAME') + biases               
     return layer

def pooling_layer(self,layer, kernel_size, stride_size):
    # basically it reduces the complexity involved by only    taking the important features alone
    # many types of pooling is there.. average pooling, max pooling..
    # max pooling takes the maximum of the given kernel
    #kernel=[image_jump,rows,columns,depth]
    kernel = [1, kernel_size, kernel_size, 1]
    #stride=[image_jump,row_jump,column_jump,color_jump]=[1,2,2,1] mostly
    stride = [1, stride_size, stride_size, 1]
    return tf.nn.max_pool(layer, ksize=kernel, strides=stride, padding='SAME')

def flattening_layer(self,layer):
    #make it single dimensional
    input_size = layer.get_shape().as_list()
    new_size = input_size[-1] * input_size[-2] * input_size[-3]
    return tf.reshape(layer, [-1, new_size]),new_size

def fully_connected_layer(self,layer, input_shape, output_shape):
    #create weights and biases for the given layer shape
    weights = self.add_weights([input_shape, output_shape])
    biases = self.add_biases([output_shape])
    #most important operation
    layer = tf.matmul(layer,weights) + biases  # mX+b
    return layer


def _add_logit_layer(input):
    """Build a single softmax layer.

    Args:
      input: The upstream tensor. float - [batch_size, N]

    Returns:
      softmax_linear: Output tensor with the computed logits.
    """

    print("_add_logit_layer input:", input)

    with tf.name_scope('softmax_linear'):
        nRows, nCols = OUTERMOST_SPHERE_SHAPE
        nOuterCells = nRows*nCols
        nInnerCells = int(input.get_shape()[1])
        batch_size = tf.shape(input)[0]
        wtStdv = 1.0 / math.sqrt(float(nOuterCells))
        weights = tf.Variable(tf.truncated_normal([nInnerCells, nOuterCells],
                                                  stddev= wtStdv),
                              name='logit_w')
        biases = tf.Variable(tf.zeros([nOuterCells]), name='logit_b')
        logits = tf.reshape(tf.matmul(input, weights) + biases,
                            [batch_size, nRows, nCols])
    return logits


_gbl_cross_tensor = None
_gbl_cross_mask = None

def _add_cross(inputImg):
    global _gbl_cross_tensor, _gbl_cross_mask
    if _gbl_cross_tensor is None:
        nRows, nCols = OUTERMOST_SPHERE_SHAPE
        arr = np.zeros(OUTERMOST_SPHERE_SHAPE, dtype=np.float32)
        maskArr = np.zeros_like(arr, dtype=np.bool)
#         maskArr = np.array(shape=OUTERMOST_SPHERE_SHAPE, dtype=np.bool)
        maskArr.fill(False)
        sR = float(nCols)/float(nRows)
        for i in range(nRows):
            j = int(math.floor(i * sR))
            arr[i:j] = 0.01
            arr[nRows - (i+1), j] = 0.01
            maskArr[i,j] = True
            maskArr[nRows - (i+1), j] = True
        _gbl_cross_tensor = tf.expand_dims(tf.constant(arr), 0)
        _gbl_cross_mask = tf.expand_dims(tf.constant(maskArr), 0)

    nRows, nCols = OUTERMOST_SPHERE_SHAPE
    nOuterCells = nRows*nCols
    batch_size = tf.shape(inputImg)[0]

    expandedCross = tf.tile(_gbl_cross_tensor, [batch_size, 1, 1])
    expandedMask = tf.tile(_gbl_cross_mask, [batch_size, 1, 1])

    return tf.where(expandedMask, expandedCross, inputImg)


def inference(feature, patternStr):
    """Build the model up to where it may be used for inference.

    Args:
      feature : [batch_size, N_BALL_SAMPS] - feature placeholder, from inputs().

      patternStr : str - string specifying network pattern. One of:
                  'outer_layer_cnn'

    Returns:
      softmax_linear: Output tensor with the computed logits.

    """

    if patternStr == 'outer_layer_cnn':

        """ Apply a CNN to the outer layer of the ball.

        """

        with tf.name_scope('cnn'):
            nRows, nCols = OUTERMOST_SPHERE_SHAPE
            nOuterCells = nRows*nCols

            # The layers of the ball are stored in a 1D array as
            # [ --layer0-- , --layer1--, ... , --outermost_layer-- ]
            skinStart = N_BALL_SAMPS - nOuterCells

            # Slice the outer layer of pixels from the feature
            # outer_skin : [batch_size, nOuterCells]
            outer_skin = tf.slice(feature, [0, skinStart], [-1, nOuterCells])

            # Reshape the outer_skin into 2 dimensions and 1 channel : [nRows, nCols, 1]
            input_skin = tf.reshape(outer_skin, [-1, nRows, nCols, 1], name="input")

          #  tf.summary.image('input_outer_skin',
           #                  tf.reshape(input_skin, [-1, nRows, nCols, 1]),
            #                 max_outputs=100)

            # Convolutional layer #1
            # conv1 : [batch_size, nRows, nCols, 8]
            conv1 = tf.contrib.layers.conv2d(
                inputs=input_skin,
                num_outputs=8,
                kernel_size=[5, 5],
                activation_fn=tf.nn.relu,
                padding="SAME",
                scope="conv1")

            # Pooling layer #1
            # pool1 : [batch_size, ceil(nRows / 2), ceil(nCols / 2), 8]
            pool1 = tf.contrib.layers.max_pool2d(
                inputs=conv1,
                kernel_size=[2, 2],
                stride=2,
                padding="SAME",
                scope="pool1")

            print('pool1:', pool1)

            # Convolutional layer #2
            # conv2 : [batch_size, ceil(nRows / 2), ceil(nCols / 2), 8]
            conv2 = tf.contrib.layers.conv2d(
                inputs=pool1,
                num_outputs=16,
                kernel_size=[5, 5],
                activation_fn=tf.nn.relu,
                padding="SAME",
                scope="conv2")

            # Pooling layer #2
            # pool2 : [batch_size, ceil(nRows / 4), ceil(nCols / 4), 16 ]
            pool2 = tf.contrib.layers.max_pool2d(
                inputs=conv2,
                kernel_size=[2, 2],
                stride=2,
                padding="SAME",
                scope="pool2")

            pool2_dim = pool2.get_shape().as_list()
            batch_size, pool2_height, pool2_width, pool2_filters = pool2_dim

            num_units = pool2_height * pool2_width * pool2_filters

            # Flatten pool2 into [batch_size, h * w * filters]
            pool2_flat = tf.reshape(pool2, [-1, num_units],
                                    name="pool2_flat")

            print('pool2_flat:', pool2_flat)

            # Fully connected relu layer
            with tf.variable_scope("dense") as scope:
                num_neurons = nOuterCells

                # Flatten pool2 into [batch_size, h * w * filters]
                pool2_flat = tf.reshape(pool2, [-1, num_units],
                                        name="pool2_flat")

                weights = weight_variable([num_units, num_neurons])
                biases  = bias_variable([num_neurons])

                # dense : [batch_size, nOuterCells]
                dense = tf.nn.relu(tf.matmul(pool2_flat, weights) + biases, name=scope.name)

            # Apply dropout to prevent overfitting
            # dropout = tf.contrib.layers.dropout(
            #     inputs=dense, rate=0.4, training=mode == learn.ModeKeys.TRAIN)

        logits = _add_logit_layer(dense)

    else:
        raise RuntimeError('Unknown inference pattern "%s"' % patternStr)

    return logits


def loss(logits, labels):
    """Calculates the loss from the logits and the labels.

    Args:
      logits: Logits tensor, float - [batch_size, NUM_CLASSES].
      labels: Labels tensor, float - [batch_size].

    Returns:
      loss: Loss tensor of type float. - []
    """
    with tf.name_scope('loss') as scope:
        batch_size = tf.shape(labels)[0]
        logits = tf.reshape(logits, [batch_size, -1])
        tf.summary.histogram(scope + 'logits', logits)
        labels = tf.reshape(labels, [batch_size, -1])
        softLogits = tf.nn.l2_normalize(logits, 1)
        tf.summary.histogram(scope + 'soft_logits', softLogits)

        diffSqr = tf.squared_difference(softLogits, labels)
        tf.summary.histogram(scope + 'squared_difference', diffSqr)
        loss = tf.reduce_sum(diffSqr, 1)
        tf.summary.histogram(scope + 'loss', loss)
        nRows, nCols = OUTERMOST_SPHERE_SHAPE
        logitImg = _add_cross(tf.reshape(softLogits,[batch_size, nRows, nCols]))
        tf.summary.image(scope + 'logits',
                         tf.reshape(logitImg, [batch_size, nRows, nCols, 1]),
                         max_outputs=100)
        tf.summary.image(scope + 'labels',
                         tf.reshape(labels, [batch_size, nRows, nCols, 1]),
                         max_outputs=100)
    return loss


def training(loss, learning_rate):
    """Sets up the training Ops.

    Creates a summarizer to track the loss over time in TensorBoard.

    Creates an optimizer and applies the gradients to all trainable variables.

    The Op returned by this function is what must be passed to the
    `sess.run()` call to cause the model to train.

    Args:
      loss: Loss tensor, from loss().
      learning_rate: The learning rate to use for gradient descent.

    Returns:
      train_op: The Op for training.
    """
    # Add a scalar summary for the snapshot loss.
    tf.summary.scalar('mean_training_loss_this_batch', tf.reduce_mean(loss))
    # Create the gradient descent optimizer with the given learning rate.
    optimizer = tf.train.GradientDescentOptimizer(learning_rate)
    # Create a variable to track the global step.
    global_step = tf.Variable(0, name='global_step', trainable=False)
    # Use the optimizer to apply the gradients that minimize the loss
    # (and also increment the global step counter) as a single training step.
    train_op = optimizer.minimize(loss, global_step=global_step)
    return train_op
