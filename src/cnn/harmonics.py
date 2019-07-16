from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import sys

from six.moves import urllib
from six.moves import xrange  # pylint: disable=redefined-builtin
import tensorflow as tf
from tensorflow.python.framework import dtypes
import numpy as np
from math import ceil

from brainroller.transforms import transToEulerRzRyRz, Quaternion
import pyshtools.shtools as psh
from constants import *

#tf.enable_eager_execution()


def latToTheta(lat):
    """
    Takes latitude values as provided by GLQGridCoord and converts to correctly oriented radians.
    """
    return (90.0 - lat) * (np.pi/180.0)


def lonToPhi(lon):
    """
    Takes longitude values as provided by GLQGridCoord and converts to correctly oriented radians.
    """
    return (np.pi/180.0) * lon


def prep_rotations(edgeLen, maxL, rMax):
    """
    Build harmonics tables.  This is expected to be called once per block.
    """
    fullChain = [(0, 0, 0)]
    for edge in range(1, edgeLen+1):
        r = 0.5 * edge
        l = int(ceil((edge * maxL)/ float(edgeLen + 1)))
        fullChain.append((edge, r, l))
    fullChain.append(((edgeLen+1), rMax, maxL))
    
    lDict = {}
    for _, _, l in fullChain:
        nodes, weights = psh.SHGLQ(l)
        lDict[l] = (nodes, weights, psh.djpi2(l))
    print('##### Completed prep_rotations')
    return fullChain, lDict

def dims_from_l(l):
    """
    convenience function for calculating sample array dims from the associated l
    """
    return (l+1, (2 * l) + 1)

def random_rotate_ball_data(vals_to_rotate, full_chain, l_dict):
    r0, r1, r2 = np.random.random(size=3)
    theta = np.arccos((2.0 * r0) - 1.0)
    phi = 2.0 * np.pi * r1
    alpha = 2.0 * np.pi * r2
    z = np.cos(theta)
    x = np.sin(theta) * np.cos(phi)
    y = np.sin(theta) * np.sin(phi)
    vecM = np.asarray([x, y, z]).reshape((3,1))
    rot = Quaternion.fromAxisAngle(vecM, alpha).toTransform()
    theta0, theta1, theta2 = transToEulerRzRyRz(rot)
    thetaArr = np.array([-theta2, -theta1, -theta0])
    
    sampOffset = 0
    rslt = np.zeros_like(vals_to_rotate)
    for _, _, l in full_chain:
        sampDim1, sampDim2 = dims_from_l(l)
        sampBlkSz = sampDim1 * sampDim2
        #if l in [MAX_L]:  # to only output one layer
        if True:
            sampBlk = vals_to_rotate[sampOffset: sampOffset+sampBlkSz]
            nodes, weights, rotMtx = l_dict[l]
            hrmBlk = psh.SHExpandGLQ(sampBlk.reshape((sampDim1, sampDim2)), weights, nodes)
            hrmRotBlk = psh.SHRotateRealCoef(hrmBlk, thetaArr, rotMtx)
            sampRotBlk = psh.MakeGridGLQ(hrmRotBlk, nodes)
            rslt[sampOffset: sampOffset + sampBlkSz] = sampRotBlk.flat
        sampOffset += sampBlkSz
    return rslt


def random_rotate_op(images):

    edge_len = 2 * RAD_PIXELS + 1
    r_max = 0.5*float(edge_len + 1)
    full_chain, l_dict = prep_rotations(edge_len, MAX_L, r_max)

    rslt = np.apply_along_axis(random_rotate_ball_data, 1, images,
                               full_chain=full_chain, l_dict=l_dict)
    return tf.convert_to_tensor(rslt)


def apply_random_rotation(images, labels):
    """
    Apply a separate random rotation to each ball of input data.  This is a
    very expensive operation- it adds 3 hours to the run time for 100 epochs
    of 7990 radius=20 maxL=48 samples.  Even rotating only the outermost layer
    adds 35 minutes.
    
    images:
        input dimensions: [batch_size, N_BALL_SAMPS]  (one channel)
        output dimensions: same as input
    labels:
        input dimensions: anything different from input shape
        output dimensions: same as input
    
    """
    # Make sure we only need to rotate the images
    with tf.control_dependencies([tf.Assert(labels.get_shape() != images.get_shape(),
                                            [labels])]):
        images = tf.py_function(random_rotate_op,
                                [images],
                                dtypes.float32,
                                name='random_rot')

        return images, labels


def extract_and_pair_single(images, full_chain, l_dict, top_l, layers):
    outDim1 = top_l + 1
    outDim2 = (2 * top_l) + 1
    top_nodes, top_weights, _ = l_dict[top_l]
    sampOffset = 0
    layerOffset = 0
    rslt = np.zeros((len(layers), outDim1, outDim2))
    for _, _, l in full_chain:
        sampDim1 = l + 1
        sampDim2 = (2 * l) + 1
        sampBlkSz = sampDim1 * sampDim2
        if l in layers:
            sampBlk = images[sampOffset: sampOffset+sampBlkSz]
            if l == top_l:
                rslt[layerOffset, :, :] = sampBlk.reshape((sampDim1, sampDim2))
            else:
                nodes, weights, rotMtx = l_dict[l]
                hrmBlk = psh.SHExpandGLQ(sampBlk.reshape((sampDim1, sampDim2)), weights, nodes)
                padHrmBlk = np.zeros_like(top_weights)
                hrmD1, hrmD2 = hrmBlk.shape
                padHrmBlk[:hrmD1, :hrmD2] = hrmBlk
                padSampBlk = psh.MakeGridGLQ(hrmRotBlk, top_nodes)
                rslt[layerOffset, :, :] = padSampBlk
                layerOffset += 1
        sampOffset += sampBlkSz
    return rslt


def extract_and_pair_op(images, layer_list):
    edge_len = 2 * RAD_PIXELS + 1
    r_max = 0.5*float(edge_len + 1)
    full_chain, l_dict = prep_rotations(edge_len, MAX_L, r_max)

    rslt = np.apply_along_axis(extract_and_pair_single, 1, images,
                               full_chain=full_chain, l_dict=l_dict,
                               top_l=max(layer_list).numpy(), layers=layer_list)
    blk_sz, layers, rows, cols  = rslt.shape
    return tf.convert_to_tensor(rslt.reshape((blk_sz, layers, rows, cols, 1)).astype(np.float32))
    

def extract_and_pair(images, layer_list):
    """
    Extract, up-sample, and merge multiple layers from each ball.
    
    images:
        input dimensions: [batch_size, N_BALL_SAMPS]  (one channel)
        output dimensions: [batch_size, len(layer_list), nRows, nCols, 1]
    where nRows and nCols are appropriate to the selected layer with the highest l
    """
    n_layers = len(layer_list)
    dim1, dim2 = dims_from_l(max(layer_list))
    rslt = tf.py_function(extract_and_pair_op, [images, tf.constant(layer_list)],
                          dtypes.float32, name='extract_and_pair')
    nRows, nCols = OUTERMOST_SPHERE_SHAPE
    rslt = tf.reshape(rslt, [-1, n_layers, dim1, dim2, 1])
    tf.summary.image('scaledlayers', tf.reshape(rslt, [-1, n_layers*dim1, dim2, 1]), max_outputs=100)
    return rslt

                                                        
    

