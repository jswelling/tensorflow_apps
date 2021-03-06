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

import harmonics
from constants import *

flags = tf.app.flags
FLAGS = flags.FLAGS

OUTERMOST_SPHERE_N = OUTERMOST_SPHERE_SHAPE[0] * OUTERMOST_SPHERE_SHAPE[1]


def layer_list_from_flags():
    """Extract a list of layers from the command line flag"""
    layer_list = FLAGS.layers.split(',')
    layer_list = [int(elt.strip()) for elt in layer_list]
    assert all([elt >= 0 and elt <= MAX_L for elt in layer_list]), 'invalid layer requested'
    print('layer_list: %s'%layer_list)
    return layer_list
    


def weight_variable(shape):
    """Generate a tensor of weight variables of dimensions `shape`.
    Initialize them with a small amount of noise for symmetry breaking

    Args:
      shape : [...] - desired shape of the weights

    """
    # return tf.get_variable('weight', shape=shape,
    #                        initializer=tf.truncated_normal_initializer(mean=0.0,
    #                                                                    stddev=0.1))
    # return tf.get_variable('weight', shape=shape,
    #                        initializer=tf.contrib.layers.xavier_initializer(uniform=False))

    return tf.get_variable('weight', shape=shape,
                           initializer=tf.initializers.he_normal())


def bias_variable(shape):
    """Generate a tensor of bias variables of dimensions `shape`.
    Initialize them with a small positive bias to avoid dead neurons
    (if using relu).

    Args:
      shape : [...] - desired shape of the biases

    """
    return tf.get_variable('bias', shape=shape,
                           initializer=tf.initializers.he_uniform())

def _add_dense_linear_layer(input, n_outer_cells):
    """Build a single densely connected layer with no activation component

    Args:
      input: The upstream tensor. float - [batch_size, dim1, dim2, 1] (last 2 dims optional)

    Returns:
      logits: Output tensor with the computed logits. - [batch_size, n_outer_cells]
    """

    with tf.name_scope('dense_linear'):
        input_shape = input.get_shape().as_list()
        assert (len(input_shape) < 4
                or len(input_shape) == 4 and input_shape[-1] == 1), 'bad input shape'
        n_inner_cells = 1
        for dim in input_shape[1:]:
            n_inner_cells *= dim
        print('n_inner_cells: ', n_inner_cells, type(n_inner_cells))
        batch_size = tf.shape(input)[0]
        wtStdv = 1.0 / math.sqrt(float(n_outer_cells))
        # weights = tf.get_variable('logit_w', shape=[n_inner_cells, n_outer_cells],
        #                           initializer=tf.truncated_normal_initializer(mean=0.0,
        #                                                                       stddev=wtStdv))
        # biases = tf.get_variable('logit_b', shape=[n_outer_cells],
        #                          initializer=tf.contrib.layers.xavier_initializer())
        weights = tf.get_variable('logit_w', shape=[n_inner_cells, n_outer_cells],
                                  initializer=tf.initializers.he_normal())
        tf.summary.histogram('logit_w', weights)
        biases = tf.get_variable('logit_b', shape=[n_outer_cells],
                                 initializer=tf.initializers.he_uniform())
        tf.summary.histogram('logit_b', biases)
        # print('input', input)
        shaped_input = tf.reshape(input, [batch_size, n_inner_cells])
        # print('shaped input: ', shaped_input)
        # print('weights: ', weights)
        # print('biases: ', biases)
        logits = tf.matmul(shaped_input, weights) + biases
        # print('logits: ', logits)
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
    n_outer_cells = nRows*nCols
    batch_size = tf.shape(inputImg)[0]

    expandedCross = tf.tile(_gbl_cross_tensor, [batch_size, 1, 1])
    expandedMask = tf.tile(_gbl_cross_mask, [batch_size, 1, 1])

    return tf.where(expandedMask, expandedCross, inputImg)


# def build_convolution( input, kernel_size, stride_size, output_dims):
#     """
#     Given input with shape [batch_size, xdim, ydim], build a layer with
#     the steps:
#     * convolve with a moving filter of size filter_dims, producing
#     """
#     
#     layer - takes in last layer.
# kernel - kernel size for convoluting on the image.
# input_shape - size of the input image.
# ouput_shape - size of the convoluted image.
# stride_size - determines kernel jump size
# def conv_layer(self,layer, kernel, input_shape, output_shape, stride_size):
#      weights = self.add_weights([kernel, kernel, input_shape,
#      output_shape])
#      biases = self.add_biases([output_shape])
#      #stride=[image_jump,row_jump,column_jump,color_jump]=[1,1,1,1]
#      stride = [1, stride_size, stride_size, 1]
#      #does a convolution scan on the given image
#      layer = tf.nn.conv2d(layer, weights, strides=stride, 
#              padding='SAME') + biases               
#      return layer

def build_filter(input, pattern_str, **kwargs):
    """Build the model up to where it may be converted to a logit.

    Args:
      input : input tensor; required shape varies by pattern_str

      pattern_str : str - string specifying network pattern. One of:
        'strip_outer_layer' : [batch_size, N_BALL_SAMPS] -> [batch_size, nRows, nCols, 1]
        'outer_layer_cnn': [batch_size, nRows, nCols, 1]
    
      (kwargs are pattern-specific)
                  

    Returns:
      output: output tensor; expected shape varies by pattern_str

    """

    # Some convenient constants
    nRows, nCols = OUTERMOST_SPHERE_SHAPE
    n_outer_cells = nRows*nCols
    print('### Beginning filter pattern ', pattern_str)

    if pattern_str == 'strip_outer_layer':

        """ 
        Strip the outer layer of the ball as an array.
        The expected input shape is [batch_size, N_BALL_SAMPS].
        Output shape is [batch_size, nRows, nCols, 1]
        """

        # The layers of the ball are stored in a 1D array as
        # [ --layer0-- , --layer1--, ... , --outermost_layer-- ]
        skinStart = N_BALL_SAMPS - n_outer_cells

        # Slice the outer layer of pixels from the feature
        # outer_skin : [batch_size, n_outer_cells]
        outer_skin = tf.slice(input, [0, skinStart], [-1, n_outer_cells])

        # Reshape the outer_skin into 2 dimensions and 1 channel : [nRows, nCols, 1]
        input_skin = tf.reshape(outer_skin, [-1, nRows, nCols, 1], name="input")
        
        #tf.summary.image('input_outer_skin', input_skin, max_outputs=100)

        return input_skin
    
    elif pattern_str == 'strip_layers':
        """
        Select one or more layers, resampling all up to the L value of the 
        highest resolution layer.
        Input shape: [batch_size, N_BALL_SAMPS]  (1 channel)
        Output shape: [batch_size, n_layers, nRows, nCols, 1]
        """
        return harmonics.extract_and_pair(input, layer_list=kwargs['layers'])

    elif pattern_str == 'outer_layer_cnn':
        """
        Apply a CNN to the outer layer of the ball.
        input shape: [batch_size, nRows, nCols, 1] or [batch_size, nLayers, nRows, nCols, 1]
        output shape: [batch_size, nRows, nCols, 1]
        """

        # Convolutional layer #1
        # conv1 : [batch_size, nRows, nCols, 8]
        dim_list = input.get_shape().as_list()
        if len(dim_list) == 5:
            batch_size, n_layers, n_rows, n_cols, input_chan = dim_list
        else:
            batch_size, n_rows, n_cols, input_chan = dim_list
            n_layers = 1

        if n_layers == 1:
            conv1 = tf.contrib.layers.conv2d(
                inputs=input,
                num_outputs=8,
                kernel_size=[5, 5],
                activation_fn=tf.nn.relu,
                padding="SAME",
                scope="conv1")
        else:
            conv1 = tf.contrib.layers.conv3d(
                inputs=input,
                num_outputs=8,
                kernel_size=[n_layers, 5, 5],
                stride = [n_layers, 1, 1],
                activation_fn=tf.nn.relu,
                padding="SAME",
                scope="conv1")
        print('conv1: ', conv1)
        conv1 = tf.reshape(conv1, [-1, n_rows, n_cols, 8])

        # Pooling layer #1
        # pool1 : [batch_size, ceil(nRows / 2), ceil(nCols / 2), 8]
        pool1 = tf.contrib.layers.max_pool2d(
            inputs=conv1,
            kernel_size=[2, 2],
            stride=2,
            padding="SAME",
            scope="pool1")
        print('pool1:', pool1)

        if FLAGS.drop1 < 1.0:
            pool1 = tf.nn.dropout(pool1, keep_prob=FLAGS.drop1)

        # Convolutional layer #2
        # conv2 : [batch_size, ceil(nRows / 2), ceil(nCols / 2), 8]
        conv2 = tf.contrib.layers.conv2d(
            inputs=pool1,
            num_outputs=16,
            kernel_size=[5, 5],
            activation_fn=tf.nn.relu,
            padding="SAME",
            scope="conv2")
        print('conv2: ', conv2)

        # Pooling layer #2
        # pool2 : [batch_size, ceil(nRows / 4), ceil(nCols / 4), 16 ]
        pool2 = tf.contrib.layers.max_pool2d(
            inputs=conv2,
            kernel_size=[2, 2],
            stride=2,
            padding="SAME",
            scope="pool2")
        print('pool2: ', pool2)

        if FLAGS.drop2 < 1.0:
            pool2 = tf.nn.dropout(pool2, keep_prob=FLAGS.drop2)

        pool2_dim = pool2.get_shape().as_list()
        batch_size, pool2_height, pool2_width, pool2_filters = pool2_dim

        num_units = pool2_height * pool2_width * pool2_filters

        # Fully connected relu layer
        with tf.variable_scope("dense") as scope:
            num_neurons = n_outer_cells

            # Flatten pool2 into [batch_size, h * w * filters]
            pool2_flat = tf.reshape(pool2, [-1, num_units],
                                    name="pool2_flat")
            print('pool2_flat:', pool2_flat)

            weights = weight_variable([num_units, num_neurons])
            biases  = bias_variable([num_neurons])

            # dense : [batch_size, n_outer_cells]
            dense = tf.nn.relu(tf.matmul(pool2_flat, weights) + biases, name=scope.name)

        return tf.reshape(dense, [-1, nRows, nCols, 1])
    elif pattern_str == 'dense_linear':
        """
        Add a dense linear layer (no relu component)
        input shape: [batch_size, dim1, dim2, 1]  with last two columns
        output shape: [batch_size, OUTERMOST_SPHERE_N]
        """
        return _add_dense_linear_layer(input, OUTERMOST_SPHERE_N)

    elif pattern_str == 'image_binary_classifier':
        """
        Classify an image into 1 of 2 categories.
        input shape: [batch_size, nRows, nCols, 1]
        output shape: [batch_size, 2]
        """
        
        conv1 = tf.contrib.layers.conv2d(
            inputs=input,
            num_outputs=8,
            kernel_size=[5, 5],
            activation_fn=tf.nn.relu,
            padding="SAME",
            scope="conv1_binary")
        print('conv1: ', conv1)

        pool1 = tf.contrib.layers.max_pool2d(
            inputs=conv1,
            kernel_size=[2, 2],
            stride=2,
            padding="SAME",
            scope="pool1_binary")
        print('pool1:', pool1)

        conv2 = tf.contrib.layers.conv2d(
            inputs=pool1,
            num_outputs=16,
            kernel_size=[5, 5],
            activation_fn=tf.nn.relu,
            padding="SAME",
            scope="conv2_binary")
        print('conv2: ', conv2)

        pool2 = tf.contrib.layers.max_pool2d(
            inputs=conv2,
            kernel_size=[2, 2],
            stride=2,
            padding="SAME",
            scope="pool2_binary")
        print('pool2: ', pool2)

        batch_size, pool2_height, pool2_width, pool2_channels = pool2.get_shape().as_list()

        num_units = pool2_height * pool2_width * pool2_channels

        # Fully connected relu layer
        with tf.variable_scope("dense_binary") as scope:
            num_neurons = 1024

            # Flatten pool2 into [batch_size, h * w * channels]
            pool2_flat = tf.reshape(pool2, [-1, num_units],
                                    name="pool2_flat_binary")
            print('pool2_flat:', pool2_flat)

            weights = weight_variable([num_units, num_neurons])
            biases  = bias_variable([num_neurons])

            # dense : [batch_size, num_neurons]
            dense = tf.nn.relu(tf.matmul(pool2_flat, weights) + biases, name='dense_binary_relu')
            print('dense: ', dense)

        with tf.variable_scope("dense_linear") as scope:
            logits = _add_dense_linear_layer(dense, 2)
        return logits
    
    elif pattern_str == 'l2_norm':
        """
        input shape: [batch_size, nRows*nCols, nChan]  # last dim optional
        output shape: [batch_size, nRows, nCols, nChan]
        
        """
        nRows, nCols = OUTERMOST_SPHERE_SHAPE
        nChan = 1
        
        feature = tf.nn.l2_normalize(input, 1)
        feature = tf.reshape(feature, [-1, nRows, nCols, nChan])
        return feature

    else:
        raise RuntimeError('Unknown inference pattern "%s"' % pattern_str)


# def lnet_single(images, full_chain, l_dict, l_weights, lmix_weights, parms):
#     l_max, l_min, n_chan = parms
#     l_max = int(l_max)
#     l_min = int(l_min)
#     n_chan = int(n_chan)
#     sampOffset = 0
#     rslt = np.zeros((2, l_min+1, l_min+1, n_chan))
#     wtOffset = 0
#     for _, _, l in full_chain:
#         sampDim1, sampDim2 = dims_from_l(l)
#         sampBlkSz = sampDim1 * sampDim2
#         if l >= l_min and l <= l_max:
#             sampBlk = images[sampOffset: sampOffset+sampBlkSz]
#             samps = sampBlk.reshape((sampDim1, sampDim2))
#             nodes, weights, rotMtx = l_dict[l]
#             hrm = psh.SHExpandGLQ(samps, weights, nodes)
#             hrm = hrm[:, :l_min+1, :l_min+1]  # drop higher harmonics
#             weighted_hrm = np.einsum('alm,ln->almn', hrm, l_weights)
#             rslt += lmix_weights[wtOffset] * weighted_hrm
#             wtOffset += 1
#         sampOffset += sampBlkSz
#     return rslt
# 
# 
# def lnet_op(images, lmix_weights, l_weights, parms):
#     l_max, l_min, n_chan = parms
#     edge_len = 2 * RAD_PIXELS + 1
#     r_max = 0.5*float(edge_len + 1)
#     full_chain, l_dict = prep_rotations(edge_len, MAX_L, r_max)
#     print('images.shape is ', images.shape)
#     print('lmix_weights is type ', type(lmix_weights), ' shape ', lmix_weights.shape)
#     rslt = np.apply_along_axis(lnet_single, 1, images,
#                                full_chain=full_chain, l_dict=l_dict,
#                                l_weights=l_weights, lmix_weights=lmix_weights,
#                                parms=parms)
#     print('rslt.shape is ', rslt.shape)
#     blk_sz, a_sz, b_sz, c_sz, channels  = rslt.shape
#     assert a_sz == 2 and b_sz == c_sz, 'returned array has unexpected shape %s' % str(rslt.shape)
#     return tf.convert_to_tensor(rslt.reshape((blk_sz, a_sz*b_sz*c_sz, channels)).astype(np.float32))


def calc_offset(dim1, dim2, dim3, x1, x2, x3):
    return x3 + ((x1 + (dim1 * x2)) * dim2)


def build_select_mtx(l_min, l_max, layer_list, full_chain):
    """
    Return a matrix that will pick the sub-matrices with fixed radial l out of a ball
    """
    tot_sz = sum([2 * (l+1) * (l+1) for l in layer_list])
    out_sz = len(layer_list) * 2 * (l_min + 1) * (l_min + 1)
    indices = []
    values = []
    dense_shape = [tot_sz, out_sz]
    in_base = 0
    out_base = 0
    for rad_l in layer_list:
        for a in range(2):
            for b in range(l_min + 1):
                for c in range(l_min + 1):
                    in_offset = calc_offset(2, rad_l + 1, rad_l + 1, a, b, c)
                    out_offset = calc_offset(2, l_min + 1, l_max + 1, a, b, c)
                    indices.append([in_offset + in_base, out_offset + out_base])
                    values.append(1.0)
        in_base += 2 * (rad_l + 1) * (rad_l + 1)
        out_base += 2 * (l_min + 1) * (l_min + 1)
    return indices, values, dense_shape


def lnet(images, l_min, l_max, n_chan):
    """
    Implement the lnet network
    """
    edge_len = 2 * RAD_PIXELS + 1
    r_max = 0.5*float(edge_len + 1)
    full_chain = harmonics.prep_chain(edge_len, MAX_L, r_max)
    layer_list = [l for _, _, l in full_chain if l >= l_min and l <= l_max]
    n_mix_weights = len(layer_list)
    elts_per_layer = 2 * (l_min + 1) * (l_min + 1)
    elts_before_select = sum([2*(l+1)*(l+1) for l in layer_list])

    l_weights = tf.get_variable('lweight',
                                shape=[(l_min + 1), n_chan],
                                initializer=tf.initializers.he_uniform())
    lmix_weights = tf.get_variable('lmixweight', shape=[n_mix_weights, n_chan],
                                   initializer=tf.initializers.he_uniform())

    indices, values, dense_shape = build_select_mtx(l_min, l_max, layer_list, full_chain)
    select_mtx = tf.SparseTensor(indices=indices, values=values, dense_shape=dense_shape)
    select_mtx = tf.sparse.reorder(select_mtx)

    images = tf.reshape(images, [-1, N_BALL_SAMPS, 1])  # since next op expects explicit channels
    ylms = harmonics.samples_to_ylms(images, layer_list, 1)
    # dims of ylms should be [batch_sz, tot_hrm_sz, 1] where
    #  tot_hrm_sz = [2*(l+1)*(l+1)] summed over l in layer_list
    tot_hrm_sz = tf.shape(ylms)[1]
    mtx = tf.reshape(ylms, [-1, tot_hrm_sz])
    mtx = tf.sparse_tensor_dense_matmul(select_mtx, mtx, adjoint_a=True, adjoint_b=True)
    mtx = tf.transpose(mtx)
    # dims now [batch_sz, n_mix_weights, elts_per_layer)]
    mtx = tf.reshape(mtx, [-1, n_mix_weights, 2, l_min + 1, l_min + 1])
    mtx = tf.einsum('bxalm,lc->bxalmc', mtx, l_weights)
    tf.summary.histogram('l_weights', l_weights)
    mtx = tf.einsum('bxalmc,xc->balmc', mtx, lmix_weights)
    tf.summary.histogram('lmix_weights', lmix_weights)
    # dims now [batch_sz, 2, l_min+1, l_min+1, n_chan]

    return mtx


def inference(feature, pattern_str, **kwargs):
    """Build the model up to where it may be used for inference.

    Args:
      feature : [batch_size, N_BALL_SAMPS] - feature placeholder, from inputs().

      pattern_str : str - string specifying network pattern. One of:
                  'outer_layer_cnn'

    Returns:
      something_not_softmax_linear: Output tensor with the computed logits.

    """

    if pattern_str == 'outer_layer_cnn':

        """ Apply a CNN to the outer layer of the ball.

        """

        with tf.variable_scope('cnn'):
            
            outer_layer = build_filter(feature, 'strip_outer_layer')

            dense = build_filter(outer_layer, 'outer_layer_cnn')
            print('dense: ', dense)
            
            # Apply dropout to prevent overfitting
            # dropout = tf.contrib.layers.dropout(
            #     inputs=dense, rate=0.4, training=mode == learn.ModeKeys.TRAIN)

            logits = build_filter(dense, 'dense_linear')
            print('logits: ', logits)
            return logits
    
    elif pattern_str == 'outer_layer_cnn_to_binary':
        nRows, nCols = OUTERMOST_SPHERE_SHAPE
        nChan = 1
        with tf.variable_scope('cnn') as scope:
            
            outer_layer = build_filter(feature, 'strip_outer_layer')

            dense = build_filter(outer_layer, 'outer_layer_cnn')
            print('dense: ', dense)
            tf.summary.image(scope.name + 'dense',
                             tf.reshape(dense, [-1, nRows, nCols, nChan]))
            
        with tf.variable_scope('image_binary_classifier') as scope:
            
            output = build_filter(dense, 'image_binary_classifier')
            tf.summary.histogram(scope.name+'output', output)
            print('output: ', output)
            return output

    elif pattern_str == 'outer_layer_logits_to_binary':
        nRows, nCols = OUTERMOST_SPHERE_SHAPE
        nChan = 1
        with tf.variable_scope('cnn') as scope:
            outer_layer = build_filter(feature, 'strip_outer_layer')
            print('outer_layer: %s' % outer_layer.get_shape().as_list())

            dense = build_filter(outer_layer, 'outer_layer_cnn')
            print('dense: %s' % dense.get_shape().as_list())
            tf.summary.image(scope.name + 'dense', 
                             tf.reshape(dense, [-1, nRows, nCols, nChan]))
            
            logits = build_filter(dense, 'dense_linear')
            print('logits: %s' % logits.get_shape().as_list())
            logits = build_filter(logits, 'l2_norm')
            print('logits: %s' % logits.get_shape().as_list())
            tf.summary.image(scope.name + 'logits', 
                             tf.reshape(logits, [-1, nRows, nCols, nChan]))
            
        with tf.variable_scope('image_binary_classifier') as scope:
            
            output = build_filter(logits, 'image_binary_classifier')
            tf.summary.histogram(scope.name+'output', output)
            print('output: ', output)
            return output

    elif pattern_str == 'two_layers_logits_to_binary':
        nRows, nCols = OUTERMOST_SPHERE_SHAPE
        nChan = 1
        with tf.variable_scope('cnn') as scope:
            layer_list = layer_list_from_flags()
            layers = build_filter(feature, 'strip_layers', layers=layer_list)

            dense = build_filter(layers, 'outer_layer_cnn', layers=layer_list)
            print('dense: %s' % dense.get_shape().as_list())
            tf.summary.image(scope.name + 'dense', 
                             tf.reshape(dense, [-1, nRows, nCols, nChan]))
            with tf.variable_scope('dense_linear') as scope:
                logits = build_filter(dense, 'dense_linear')
            print('logits: %s' % logits.get_shape().as_list())
            logits = build_filter(logits, 'l2_norm')
            print('logits: %s' % logits.get_shape().as_list())
            tf.summary.image(scope.name + 'logits', 
                             tf.reshape(logits, [-1, nRows, nCols, nChan]))
            
        with tf.variable_scope('image_binary_classifier') as scope:
            
            output = build_filter(logits, 'image_binary_classifier')
            tf.summary.histogram(scope.name+'output', output)
            print('output: ', output)
            return output

    elif pattern_str == 'two_layers_short_binary':
        n_rows, n_cols = OUTERMOST_SPHERE_SHAPE
        layer_list = layer_list_from_flags()
        n_layers = len(layer_list)
        n_chan_0 = 1
        n_chan_1 = 8
        n_chan_2 = 16
        num_neurons = 1024  # In the dense nonlinear layer
        with tf.variable_scope('short_binary') as scope:
            layers = build_filter(feature, 'strip_layers', layers=layer_list)

            layers = tf.nn.l2_normalize(tf.reshape(layers,
                                                   [-1, n_layers*n_rows*n_cols]),
                                        1)
            layers = tf.reshape(layers, [-1, n_layers, n_rows, n_cols, 1])

            conv1 = tf.contrib.layers.conv3d(
                inputs=layers, num_outputs=n_chan_1,
                kernel_size=[n_layers, 5, 5],
                stride = [n_layers, 1, 1],
                activation_fn=tf.nn.relu,
                padding="SAME",
                scope="conv1")
            print('conv1: ', conv1)
            conv1 = tf.reshape(conv1, [-1, n_rows, n_cols, n_chan_1])
            tf.summary.histogram('conv1', conv1)

            pool1 = tf.contrib.layers.max_pool2d(
                inputs=conv1,
                kernel_size=[2, 2],
                stride=2,
                padding="SAME",
                scope="pool1")
            print('pool1:', pool1)

            conv2 = tf.contrib.layers.conv2d(
                inputs=pool1,
                num_outputs=n_chan_2,
                kernel_size=[5, 5],
                activation_fn=tf.nn.relu,
                padding="SAME",
                scope="conv2")
            print('conv2: ', conv2)
            tf.summary.histogram('conv2', conv2)

            pool2 = tf.contrib.layers.max_pool2d(
                inputs=conv2,
                kernel_size=[2, 2],
                stride=2,
                padding="SAME",
                scope="pool2")
            print('pool2: ', pool2)

        pool2_dim = pool2.get_shape().as_list()
        batch_size, pool2_height, pool2_width, pool2_filters = pool2_dim
        num_units = pool2_height * pool2_width * pool2_filters

        with tf.variable_scope("dense_binary") as scope:

            # Flatten pool2 into [batch_size, h * w * channels]
            pool2_flat = tf.reshape(pool2, [-1, num_units],
                                    name="pool2_flat")
            print('pool2_flat:', pool2_flat)

            weights = weight_variable([num_units, num_neurons])
            biases  = bias_variable([num_neurons])

            # dense : [batch_size, num_neurons]
            dense = tf.nn.relu(tf.matmul(pool2_flat, weights) + biases, name='dense_binary_relu')
            print('dense: ', dense)
            tf.summary.histogram('dense', dense)

            logits = _add_dense_linear_layer(dense, 2)
            tf.summary.histogram('binary_logits', logits)
            return logits

    elif pattern_str == 'lnet':
        n_chan = 4
        l_min = 7
        num_neurons = 1024  # in dense non-linear layer
        
        with tf.variable_scope('lnet') as scope:
            feature_harmonics = lnet(feature, l_min, MAX_L, n_chan)
            tf.summary.histogram('lnet', feature_harmonics)

        num_units = 2 * (l_min + 1) * (l_min + 1) * n_chan
        
        with tf.variable_scope("dense_binary") as scope:
            hrm_flat = tf.reshape(feature_harmonics, [-1, num_units],
                                  name="feature_harmonics_flat")

            # dense : [batch_size, num_neurons]
            weights = weight_variable([num_units, num_neurons])
            biases = bias_variable([num_neurons])
            dense = tf.nn.relu(tf.matmul(hrm_flat, weights) + biases, name='dense_binary_relu')
            print('dense: ', dense)
            tf.summary.histogram('dense', dense)

            logits = _add_dense_linear_layer(dense, 2)
            tf.summary.histogram('binary_logits', logits)
            return logits

    else:
        raise RuntimeError('Unknown inference pattern "{}"'.format(pattern_str))


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
        tf.summary.image(scope + 'image_pairs',
                         tf.concat([tf.reshape(labels, [batch_size, nRows, nCols, 1]),
                                    tf.reshape(logitImg, [batch_size, nRows, nCols, 1])
                                    ], 2),
                         max_outputs=100)
#         tf.summary.image(scope + 'logits',
#                          tf.reshape(logitImg, [batch_size, nRows, nCols, 1]),
#                          max_outputs=100)
#         tf.summary.image(scope + 'labels',
#                          tf.reshape(labels, [batch_size, nRows, nCols, 1]),
#                          max_outputs=100)
    return loss


def binary_loss(logits, labels):
    """Calculates the loss from the logits and the labels.

    Args:
      logits: Logits tensor, float - [batch_size, NUM_CLASSES].
      labels: Labels tensor, float - [batch_size].

    Returns:
      loss: Loss tensor of type float. - []
    """
    with tf.name_scope('binary_loss') as scope:
        labels = tf.stop_gradient(labels)
        cost = tf.losses.softmax_cross_entropy(onehot_labels=labels,
                                               logits=logits)

        # #calculate the total mean of all the errors from all the nodes
        # with tf.control_dependencies([tf.print('values: ', tf.concat([logits, labels, tf.round(tf.nn.softmax(logits)), 
        #                                                               tf.reshape(cross_entropy,[-1, 1])],1))]):
        #     cost=tf.reduce_mean(cross_entropy)

    return cost


def training(loss, learning_rate, exclude=None, optimizer=None):
    """Sets up the training Ops.

    Creates a summarizer to track the loss over time in TensorBoard.

    Creates an optimizer and applies the gradients to all trainable variables.

    The Op returned by this function is what must be passed to the
    `sess.run()` call to cause the model to train.

    Args:
      loss: Loss tensor, from loss().
      learning_rate: The learning rate to use for gradient descent.
      optimizer: None, or one of the options for tf.contrib.layers.optimize_loss

    Returns:
      train_op: The Op for training.
    """
    if exclude is None:
        exclude = []
    # Add a scalar summary for the snapshot loss.
    tf.summary.scalar('mean_training_loss_this_batch', tf.reduce_mean(loss))
    # Create the gradient descent optimizer with the given learning rate.
    #optimizer = tf.train.GradientDescentOptimizer(learning_rate)
#    optimizer = tf.train.AdamOptimizer(learning_rate)
    # Create a variable to track the global step.
    # Use the optimizer to apply the gradients that minimize the loss
    # (and also increment the global step counter) as a single training step.
#    train_op = optimizer.minimize(loss, global_step=global_step)
    with tf.variable_scope('control', reuse=True):
        global_step = tf.get_variable('global_step', dtype=tf.int32)
    if optimizer is None:
        optimizer = 'Adam'
    train_these_vars = [v for v in tf.trainable_variables()
                        if v not in exclude]
    print('vars to be trained: %s' % [var.name for var in train_these_vars])
    train_op = tf.contrib.layers.optimize_loss(loss, global_step, learning_rate,
                                               optimizer,
                                               summaries=['loss',
                                                          'learning_rate',
                                                          'gradients',
                                                          'gradient_norm'],
                                               variables=train_these_vars)
    return train_op
