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

"""Builds the CIFAR-10 network.
Summary of available functions:
 # Compute input images and labels for training. If you would like to run
 # evaluations, use inputs() instead.
 inputs, labels = distorted_inputs()
 # Compute inference on the model inputs to make a prediction.
 predictions = inference(inputs)
 # Compute the total loss of the prediction with respect to the labels.
 loss = loss(predictions, labels)
 # Create a graph to run one step of training with respect to the loss.
 train_op = train(loss, global_step)
"""
# pylint: disable=missing-docstring
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import gzip
import os
import re
import sys
import tarfile

from six.moves import urllib
import tensorflow as tf
import numpy as np

import segnet_input as fish_input

FLAGS = tf.app.flags.FLAGS

# Basic model parameters.
tf.app.flags.DEFINE_integer('batch_size', 4,
                            """Number of images to process in a batch.""")
tf.app.flags.DEFINE_integer('classes', 11,
                            """Number of classes.""")

tf.app.flags.DEFINE_string('file_dir', "/pylon1/sy4s8lp/anniez/project/caffe-segnet/tutorial/CamVid/old-train.txt",
                           """Path to the file directory.""")

# Global constants describing the fish data set.
IMAGE_WIDTH,IMAGE_HEIGHT = fish_input.IMAGE_WIDTH,fish_input.IMAGE_HEIGHT
LABEL_SIZE=fish_input.LABEL_SIZE
NUM_EXAMPLES_PER_EPOCH_FOR_TRAIN = fish_input.NUM_EXAMPLES_PER_EPOCH_FOR_TRAIN
NUM_EXAMPLES_PER_EPOCH_FOR_EVAL = fish_input.NUM_EXAMPLES_PER_EPOCH_FOR_EVAL


# Constants describing the training process.
MOVING_AVERAGE_DECAY = 0.9999     # The decay to use for the moving average.
NUM_EPOCHS_PER_DECAY = 350.0      # Epochs after which learning rate decays.
LEARNING_RATE_DECAY_FACTOR = 1.0  # Learning rate decay factor.
INITIAL_LEARNING_RATE = 0.1       # Initial learning rate.

# If a model is trained with multiple GPUs, prefix all Op names with tower_name
# to differentiate the operations. Note that this prefix is removed from the
# names of the summaries when visualizing a model.
TOWER_NAME = 'tower'




def _activation_summary(x):
  """Helper to create summaries for activations.
  Creates a summary that provides a histogram of activations.
  Creates a summary that measure the sparsity of activations.
  Args:
    x: Tensor
  Returns:
    nothing
  """
  # Remove 'tower_[0-9]/' from the name in case this is a multi-GPU training
  # session. This helps the clarity of presentation on tensorboard.
  tensor_name = re.sub('%s_[0-9]*/' % TOWER_NAME, '', x.op.name)
  tf.histogram_summary(tensor_name + '/activations', x)
  tf.scalar_summary(tensor_name + '/sparsity', tf.nn.zero_fraction(x))


def _variable_on_cpu(name, shape, initializer):
  """Helper to create a Variable stored on CPU memory.
  Args:
    name: name of the variable
    shape: list of ints
    initializer: initializer for Variable
  Returns:
    Variable Tensor
  """
  with tf.device('/cpu:0'):
    var = tf.get_variable(name, shape, initializer=initializer)
  return var


def _variable_with_weight_decay(name, shape, stddev, wd):
  """Helper to create an initialized Variable with weight decay.
  Note that the Variable is initialized with a truncated normal distribution.
  A weight decay is added only if one is specified.
  Args:
    name: name of the variable
    shape: list of ints
    stddev: standard deviation of a truncated Gaussian
    wd: add L2Loss weight decay multiplied by this float. If None, weight
        decay is not added for this Variable.
  Returns:
    Variable Tensor
  """
  var = _variable_on_cpu(name, shape,
                         tf.truncated_normal_initializer(stddev=stddev))
  if wd is not None:
    weight_decay = tf.mul(tf.nn.l2_loss(var), wd, name='weight_loss')
    tf.add_to_collection('losses', weight_decay)
  return var



def inputsCNN(shuffle,evall):
  """Construct input for CIFAR evaluation using the Reader ops.
  Args:
    eval_data: bool, indicating if one should use the train or eval data set.
  Returns:
    images: Images. 4D tensor of [batch_size, IMAGE_SIZE, IMAGE_SIZE, 3] size.
    labels: Labels. 1D tensor of [batch_size] size.
  Raises:
    ValueError: If no data_dir
  """
  if not FLAGS.file_dir :
    raise ValueError('Please supply a data_dir')

  print ("cnn inputs")
  return fish_input.inputsROOT(file_dir=FLAGS.file_dir,
                              batch_size=FLAGS.batch_size,shuffle=shuffle,evall=evall)


def inference(images):
  """Build the CIFAR-10 model.
  Args:
    images: Images returned from distorted_inputs() or inputs().
  Returns:
    Logits.
  """
  # We instantiate all variables using tf.get_variable() instead of
  # tf.Variable() in order to share variables across multiple GPU training runs.
  # If we only ran this model on a single GPU, we could simplify this function
  # by replacing all instances of tf.get_variable() with tf.Variable().
  

  #local response normalization
  norm = tf.nn.lrn(images, 5, bias=1.0, alpha=0.0001, beta=0.75,
                    name='norm')
  
  # conv1
  with tf.variable_scope('conv1') as scope:
    kernel = _variable_with_weight_decay('weights', shape=[7, 7, 3, 64],
                                         stddev=1/float(7*7*64), wd=0.0)
    conv = tf.nn.conv2d(images, kernel, strides=[1, 1, 1, 1], padding='SAME')
    biases = _variable_on_cpu('biases',[64], tf.constant_initializer(0.0))
    bias = tf.nn.bias_add(conv, biases)

    mean,variance=tf.nn.moments(bias,axes=[0,1,2])
    conv_bn=tf.nn.batch_normalization(bias,mean,variance,offset=0.001,scale=1,variance_epsilon=0.1e-4)

    conv1 = tf.nn.relu(conv_bn, name=scope.name)
    
    _activation_summary(conv1)
    print ("covn1",conv1.get_shape())

  # pool1
  pool1 = tf.nn.max_pool(conv1, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1],
                         padding='SAME', name='pool1')
  print ("pool1",pool1.get_shape())
  # conv2
  with tf.variable_scope('conv2') as scope:
    kernel = _variable_with_weight_decay('weights', shape=[7, 7, 64, 64],
                                         stddev=1/float(7*7*64), wd=0.0)
    conv = tf.nn.conv2d(pool1, kernel, strides=[1, 1, 1, 1], padding='SAME')
    biases = _variable_on_cpu('biases', [64], tf.constant_initializer(0.0))
    bias = tf.nn.bias_add(conv, biases)

    mean,variance=tf.nn.moments(bias,axes=[0,1,2])
    conv_bn=tf.nn.batch_normalization(bias,mean,variance,offset=0.001,scale=1,variance_epsilon=0.1e-4)

    conv2 = tf.nn.relu(conv_bn, name=scope.name)
    
    _activation_summary(conv2)
    print ("covn2",conv2.get_shape())

  # pool2
  pool2= tf.nn.max_pool(conv2, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1],
                         padding='SAME', name='pool2')
  print ("pool2",pool2.get_shape())
  # conv3
  with tf.variable_scope('conv3') as scope:
    kernel = _variable_with_weight_decay('weights', shape=[7, 7, 64, 64],
                                         stddev=1/float(7*7*64), wd=0.0)
    conv = tf.nn.conv2d(pool2, kernel, strides=[1, 1, 1, 1], padding='SAME')
    biases = _variable_on_cpu('biases', [64], tf.constant_initializer(0.0))
    bias = tf.nn.bias_add(conv, biases)

    mean,variance=tf.nn.moments(bias,axes=[0,1,2])
    conv_bn=tf.nn.batch_normalization(bias,mean,variance,offset=0.001,scale=1,variance_epsilon=0.1e-4)

    conv3 = tf.nn.relu(conv_bn, name=scope.name)
    
    _activation_summary(conv3)
    print ("covn3",conv3.get_shape())

  # pool3
  pool3= tf.nn.max_pool(conv3, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1],
                         padding='SAME', name='pool3')

  print ("pool3",pool3.get_shape())
  # conv4
  with tf.variable_scope('conv4') as scope:
    kernel = _variable_with_weight_decay('weights', shape=[7, 7, 64, 64],
                                         stddev=1/float(7*7*64), wd=0.0)
    conv = tf.nn.conv2d(pool3, kernel, strides=[1, 1, 1, 1], padding='SAME')
    biases = _variable_on_cpu('biases', [64], tf.constant_initializer(0.0))
    bias = tf.nn.bias_add(conv, biases)

    mean,variance=tf.nn.moments(bias,axes=[0,1,2])
    conv_bn=tf.nn.batch_normalization(bias,mean,variance,offset=0.001,scale=1,variance_epsilon=0.1e-4)

    conv4 = tf.nn.relu(conv_bn, name=scope.name)
    
    _activation_summary(conv4)
    print ("covn4",conv4.get_shape())

  # pool4
  pool4= tf.nn.max_pool(conv4, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1],
                         padding='SAME', name='pool4')


#############################END OF ENCODING PART##################################
  print ("pool4",pool4.get_shape())
  
  #up4=upsample(pool4,mask=pool4_mask,scale=2,pad_out_h=True)

  #upsampling 4
  with tf.variable_scope('up4') as scope:
    kernel = _variable_on_cpu('up4', [2,2,64,64], tf.truncated_normal_initializer())
    up4 = tf.nn.conv2d_transpose(pool4, filter=kernel, 
          output_shape=[FLAGS.batch_size,int(IMAGE_HEIGHT/8+1), int(IMAGE_WIDTH/8), 64], 
          strides=[1, 2, 2, 1],padding="VALID")
    _activation_summary(up4)
    
    print ("up4",up4.get_shape())

  up4_reshape= tf.slice(up4,[0,0,0,0],[FLAGS.batch_size,int(IMAGE_HEIGHT/8), int(IMAGE_WIDTH/8),64])

  # de-conv4
  with tf.variable_scope('de_conv4') as scope:
    kernel = _variable_with_weight_decay('weights', shape=[7, 7, 64, 64],
                                         stddev=1/float(7*7*64), wd=0.0)
    de_conv = tf.nn.conv2d(up4_reshape, kernel, strides=[1, 1, 1, 1], padding='SAME')
    biases = _variable_on_cpu('biases', [64], tf.constant_initializer(0.0))
    bias = tf.nn.bias_add(de_conv, biases)

    mean,variance=tf.nn.moments(bias,axes=[0,1,2])
    de_conv4=tf.nn.batch_normalization(bias,mean,variance,
                  offset=0.001,scale=1,variance_epsilon=0.1e-4,name=scope.name)
    
    _activation_summary(de_conv4)
    
    print ("de_covn4",de_conv4.get_shape())
  
  #upsampling 3
  with tf.variable_scope('up3') as scope:
    kernel = _variable_on_cpu('up3', [2,2,64,64], tf.truncated_normal_initializer())
    up3 = tf.nn.conv2d_transpose(de_conv4,filter=kernel, 
                output_shape=[FLAGS.batch_size,int(IMAGE_HEIGHT/4), int(IMAGE_WIDTH/4), 64], 
                strides=[1, 2, 2, 1],padding="VALID")
    _activation_summary(up3)
    
    print ("up3",up3.get_shape())


  
  #up3=upsample(de_conv4,mask=pool3_mask,scale=2)

  
  # de-conv3
  with tf.variable_scope('de_conv3') as scope:
    kernel = _variable_with_weight_decay('weights', shape=[7, 7, 64, 64],
                                         stddev=1/float(7*7*64), wd=0.0)
    de_conv = tf.nn.conv2d(up3, kernel, strides=[1, 1, 1, 1], padding='SAME')
    biases = _variable_on_cpu('biases',[64], tf.constant_initializer(0.0))
    bias = tf.nn.bias_add(de_conv, biases)

    mean,variance=tf.nn.moments(bias,axes=[0,1,2])
    de_conv3=tf.nn.batch_normalization(bias,mean,variance,
                  offset=0.001,scale=1,variance_epsilon=0.1e-4,name=scope.name)
    
    _activation_summary(de_conv3)
    print ("de_covn3",de_conv3.get_shape())


  #upsampling 2
  with tf.variable_scope('up2') as scope:
    kernel = _variable_on_cpu('up2', [2,2,64,64], tf.truncated_normal_initializer())
    up2 = tf.nn.conv2d_transpose(de_conv3, filter=kernel, 
                output_shape=[FLAGS.batch_size,int(IMAGE_HEIGHT/2), int(IMAGE_WIDTH/2), 64], 
                strides=[1, 2, 2, 1],padding="VALID")
    _activation_summary(up2)
    
    print ("up2",up2.get_shape())
  #up2=upsample(de_conv3,mask=pool2_mask,scale=2)
  

  # de-conv2
  with tf.variable_scope('de_conv2') as scope:
    kernel = _variable_with_weight_decay('weights', shape=[7, 7, 64, 64],
                                         stddev=1/float(7*7*64), wd=0.0)
    de_conv = tf.nn.conv2d(up2, kernel, strides=[1, 1, 1, 1], padding='SAME')
    biases = _variable_on_cpu('biases', [64], tf.constant_initializer(0.0))
    bias = tf.nn.bias_add(de_conv, biases)

    mean,variance=tf.nn.moments(bias,axes=[0,1,2])
    de_conv2=tf.nn.batch_normalization(bias,mean,variance,
                  offset=0.001,scale=1,variance_epsilon=0.1e-4,name=scope.name)
    
    _activation_summary(de_conv2)
    print ("de_covn2",de_conv2.get_shape())

  #upsampling 1
  with tf.variable_scope('up1') as scope:
    kernel = _variable_on_cpu('up1', [2,2,64,64], tf.truncated_normal_initializer())
    up1 = tf.nn.conv2d_transpose(de_conv2, filter=kernel, 
                  output_shape=[FLAGS.batch_size, int(IMAGE_HEIGHT), int(IMAGE_WIDTH), 64], 
                  strides=[1, 2, 2, 1],padding="VALID")
    _activation_summary(up1)
    
    print ("up1",up1.get_shape())
  
  #up1=upsample(de_conv2,mask=pool1_mask,scale=2)
 

  # de-conv1
  with tf.variable_scope('de_conv1') as scope:
    kernel = _variable_with_weight_decay('weights', shape=[7, 7, 64, 64],
                                         stddev=1/float(7*7*64), wd=0.0)
    de_conv = tf.nn.conv2d(up1, kernel, strides=[1, 1, 1, 1], padding='SAME')
    biases = _variable_on_cpu('biases', [64], tf.constant_initializer(0.0))
    bias = tf.nn.bias_add(de_conv, biases)

    mean,variance=tf.nn.moments(bias,axes=[0,1,2])
    de_conv1=tf.nn.batch_normalization(bias,mean,variance,
                  offset=0.001,scale=1,variance_epsilon=0.1e-4,name=scope.name)
    
    _activation_summary(de_conv1)
    print ("de_covn1",de_conv1.get_shape())
#########################END OF DECODING PART#####################################
  # conv_classifier
  with tf.variable_scope('conv_classifier') as scope:
    kernel = _variable_with_weight_decay('weights', shape=[7, 7, 64, 11],
                                         stddev=1/float(7*7*64), wd=0.0)
    conv = tf.nn.conv2d(up1, kernel, strides=[1, 1, 1, 1], padding='SAME')
    biases = _variable_on_cpu('biases', [11], tf.constant_initializer(0.0))
    classifier = tf.nn.bias_add(conv, biases)
    
    _activation_summary(classifier)

  tf.image_summary('outputs', tf.cast(tf.reshape(tf.argmax(classifier,3),shape=[FLAGS.batch_size,IMAGE_HEIGHT,IMAGE_WIDTH,1]),tf.float32))
  return classifier

#cannot be used,causing problems for auto gradient calculation in backpropagation.
def upsample(pool,mask,scale,pad_out_h=False):
  shape=pool.get_shape().as_list()
  b=shape[0]
  h=shape[1]
  w=shape[2]
  c=shape[3]
  realmask=tf.reshape(mask,[-1])
  realpool=tf.reshape(pool,[-1])
  result_np=np.zeros(b*h*w*c*scale*scale)
  result_var=tf.Variable(result_np,dtype=tf.float32)

  result_raw=tf.scatter_update(ref=result_var, indices=realmask, updates=realpool)

  result = tf.reshape(result_raw, [b,h*scale,w*scale,c])
  if pad_out_h:
    result= tf.slice(result,[0,0,0,0],[b,h*scale-1,w*scale,c])

  return result

def accuracy(logits,labels):
  pass

def loss(logits, labels):
  """Add L2Loss to all the trainable variables.
  Add summary for "Loss" and "Loss/avg".
  Args:
    logits: Logits from inference().
    labels: Labels from distorted_inputs or inputs(). 1-D tensor
            of shape [batch_size]
  Returns:
    Loss tensor of type float.
  """
  # Calculate the average cross entropy loss across the batch.
  cast_labels = tf.cast(labels, tf.int64)


  logit_mod=tf.reshape(logits,shape=[FLAGS.batch_size*IMAGE_HEIGHT*IMAGE_WIDTH,FLAGS.classes])
  label_mod=tf.reshape(cast_labels,shape=[-1])
  cross_entropy_mod= tf.nn.sparse_softmax_cross_entropy_with_logits(logit_mod,label_mod, 
          name='cross_entropy_per_example')
  cross_entropy_mean = tf.reduce_mean(cross_entropy_mod, name='cross_entropy')
  tf.add_to_collection('losses', cross_entropy_mean)
  return tf.add_n(tf.get_collection('losses'), name='total_loss')



def _add_loss_summaries(total_loss):
  """Add summaries for losses in CIFAR-10 model.
  Generates moving average for all losses and associated summaries for
  visualizing the performance of the network.
  Args:
    total_loss: Total loss from loss().
  Returns:
    loss_averages_op: op for generating moving averages of losses.
  """
  # Compute the moving average of all individual losses and the total loss.
  loss_averages = tf.train.ExponentialMovingAverage(0.9, name='avg')
  losses = tf.get_collection('losses')
  loss_averages_op = loss_averages.apply(losses + [total_loss])

  # Attach a scalar summary to all individual losses and the total loss; do the
  # same for the averaged version of the losses.
  for l in losses + [total_loss]:
    # Name each loss as '(raw)' and name the moving average version of the loss
    # as the original loss name.
    tf.scalar_summary(l.op.name +' (raw)', l)
    tf.scalar_summary(l.op.name, loss_averages.average(l))

  return loss_averages_op


def train(total_loss, global_step):
  """Train CIFAR-10 model.
  Create an optimizer and apply to all trainable variables. Add moving
  average for all trainable variables.
  Args:
    total_loss: Total loss from loss().
    global_step: Integer Variable counting the number of training steps
      processed.
  Returns:
    train_op: op for training.
  """
  # Variables that affect learning rate.
  num_batches_per_epoch = NUM_EXAMPLES_PER_EPOCH_FOR_TRAIN / FLAGS.batch_size
  decay_steps = int(num_batches_per_epoch * NUM_EPOCHS_PER_DECAY)

  # Decay the learning rate exponentially based on the number of steps.
  lr = tf.train.exponential_decay(INITIAL_LEARNING_RATE,
                                  global_step,
                                  decay_steps,
                                  LEARNING_RATE_DECAY_FACTOR,
                                  staircase=True)
  tf.scalar_summary('learning_rate', lr)

  # Generate moving averages of all losses and associated summaries.
  loss_averages_op = _add_loss_summaries(total_loss)

  # Compute gradients.
  with tf.control_dependencies([loss_averages_op]):
    opt = tf.train.GradientDescentOptimizer(lr)
    grads = opt.compute_gradients(total_loss)

  # Apply gradients.
  apply_gradient_op = opt.apply_gradients(grads, global_step=global_step)

  # Add histograms for trainable variables.
  for var in tf.trainable_variables():
    tf.histogram_summary(var.op.name, var)

  # Add histograms for gradients.
  for grad, var in grads:
    if grad is not None:
      tf.histogram_summary(var.op.name + '/gradients', grad)

  # Track the moving averages of all trainable variables.
  variable_averages = tf.train.ExponentialMovingAverage(
      MOVING_AVERAGE_DECAY, global_step)
  variables_averages_op = variable_averages.apply(tf.trainable_variables())

  with tf.control_dependencies([apply_gradient_op, variables_averages_op]):
    train_op = tf.no_op(name='train')

  return train_op



