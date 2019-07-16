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

#tf.enable_eager_execution()

# initiate constant 
N_BALL_SAMPS = 71709  # Total number of GLQ samples in a full ball of samples
OUTERMOST_SPHERE_SHAPE = [49, 97]  # Number of GLQ samples in outermost shells
RAD_PIXELS = 20   # radius of outermost shell in pixels
MAX_L = 48  # maximum L value for outermost shell harmonic expansion

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
        sampDim1 = l + 1
        sampDim2 = (2 * l) + 1
        sampBlkSz = sampDim1 * sampDim2
        if l in [MAX_L]:
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
    # Make sure we only need to rotate the images
    with tf.control_dependencies([tf.Assert(labels.get_shape() != images.get_shape(),
                                            [labels])]):
        images = tf.py_function(random_rotate_op,
                                [images],
                                dtypes.float32,
                                name='random_rot')

        return images, labels
                                                        
    

