
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from datetime import datetime
import math
import time

import numpy as np
import tensorflow as tf

from input_data_from_block import get_loc_iterator, get_subblock_edge_len, get_full_block
import topology
import harmonics
from constants import *
from brainroller.shtransform import SHTransformer
from util import parse_int_list

flags = tf.app.flags
FLAGS = flags.FLAGS

flags.DEFINE_string('network_pattern', 'outer_layer_cnn',
                    'A network pattern recognized in topology.py')
flags.DEFINE_string('log_dir', '/tmp/eval',
                    """Directory where to write event logs.""")
flags.DEFINE_string('data_dir', '',
                    """Directory for evaluation data""")
flags.DEFINE_string('starting_snapshot', None,
                    'Snapshot to evaluate')
flags.DEFINE_integer('eval_interval_secs', 60 * 5,
                     """How often to run the eval.""")
flags.DEFINE_integer('read_threads', 2,
                     'Number of threads reading input files')
flags.DEFINE_integer('batch_size', 8, 'Batch size.  '
                     'Must divide evenly into the dataset sizes.')
flags.DEFINE_boolean('verbose', False, 'If true, print extra output.')
flags.DEFINE_string('layers', '%d,%d' % (MAX_L//2,MAX_L),
                    'layers to include (depends on network-pattern; MAX_L=%d)' % MAX_L)
flags.DEFINE_float('drop1', 1.0, 'Fraction to keep in first drop-out layer. Default of'
                   ' 1.0 means no drop-out layer in this position')
flags.DEFINE_float('drop2', 1.0, 'Fraction to keep in second drop-out layer. Default of'
                   ' 1.0 means no drop-out layer in this position')
flags.DEFINE_string('data_block_path', None, 'full path to the data block')
flags.DEFINE_string('data_block_dims', None, 
                    '3 comma-separated ints for '
                    'x, y, and z dimensions of the data block')
flags.DEFINE_integer('data_block_offset', 0, 'offset in bytes to start reading the data block file')
flags.DEFINE_string('scan_start', None,
                    '3 comma-separated ints for'
                    ' x, y, and z location of the start of the scan')
flags.DEFINE_string('scan_size', None,
                    '3 comma-separated ints for the '
                    'x, y, and z region size to scan')
flags.DEFINE_string('file_list', None,
                   'A filename containing a list of .yaml files to use for training')
flags.DEFINE_string('outname', 'scanner',
                    'A prefix for the scanned and predicted output file names')
flags.DEFINE_boolean('random_rotation', False, 'use un-oriented data and apply random'
                     ' rotations to each data sample')


def scan(coord_base_V, scan_sz_V,
         loc_iterator, x_off_op, y_off_op, z_off_op, saver, ctrpt_op, predicted_op):
    """Scan through the block locations specified by the iterator

    Args:
      saver: Saver.
      x_off_op, y_off_op, z_off_op: ops giving lower left front corner of sub-cube
      predicted_op: op giving prediction for sub-cube

    """
    
    epoch = 0
    x_base, y_base, z_base = coord_base_V
    out_blk = np.zeros(scan_sz_V, dtype=np.uint8)
    pred_blk = np.zeros(scan_sz_V, dtype=np.uint8)
    with tf.Session() as sess:
        init_op = tf.local_variables_initializer()
        sess.run(init_op)
        print('restoring from snapshot: %s'
              % [var.name for var in tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES)])
        saver.restore(sess, FLAGS.starting_snapshot)
        global_step = FLAGS.starting_snapshot.split('/')[-1].split('-')[-1]
        
        try:
            sess.run(loc_iterator.initializer, feed_dict={})
            step = 0
            while True:
                xV, yV, zV, ctrptV, predV = sess.run([x_off_op, y_off_op, z_off_op,
                                                      ctrpt_op, predicted_op])
                pred_byte = np.where((predV[:, 1] == 1), 255, 0)
                if any(pred_byte):
                    print('HIT!!!')
                print('testing: ', xV, yV, zV, ctrptV, predV, x_base, y_base, z_base)
                out_blk[xV - x_base, yV - y_base, zV - z_base] = ctrptV
                pred_blk[xV - x_base, yV - y_base, zV - z_base] = pred_byte
                step += 1
        except tf.errors.OutOfRangeError as e:
            print('Finished evaluating epoch %d (%d steps)' % (epoch, step))
    return out_blk, pred_blk


def get_subblock_op(x_off_op, y_off_op, z_off_op, blk_sz, full_block):
    blk_shape = tf.constant([blk_sz, blk_sz, blk_sz], dtype=tf.int32)
    offset_mtx = tf.transpose(tf.reshape(tf.concat([z_off_op, y_off_op, x_off_op], 0), [3, -1]))
    print('full_block: %s' % full_block)
    rslt = tf.map_fn(lambda offset: tf.slice(full_block, offset, blk_shape), offset_mtx,
                     dtype=tf.dtypes.uint8)
    print('rslt: %s' % rslt)
    return rslt


# def sample_single(double_cube, transformer, edge_len, sig):
#     return transformer.calcBallOfSamples(double_cube.reshape((edge_len, edge_len, edge_len)),
#                                          sig=sig)


# def sample_op(double_cube_mtx, edge_len):
#     transformer = SHTransformer(edge_len, MAX_L)
#     assert all([len == edge_len for len in double_cube_mtx.shape[1:]]), 'input is not appropriate cubes'
#     batch_sz = double_cube_mtx.shape[0]
#     double_cube_mtx = tf.reshape(double_cube_mtx, [batch_sz, edge_len*edge_len*edge_len])
#     rslt = np.apply_along_axis(sample_single, 1, double_cube_mtx, transformer, 
#                                edge_len=edge_len, sig=(1.0, -1.0, 1.0))
#     return tf.convert_to_tensor(rslt.reshape((batch_sz, -1)))
    

# def collect_ball_samples(double_cube_mtx, edge_len, read_threads):
#     """
#     Call python functions to interpolate samples from within the data subblocks.
    
#     data_cub_mtx dims: [batch_sz, edge_len, edge_len, edge_len]
#     """
#     rslt = tf.py_function(lambda x: sample_op(x, edge_len), [double_cube_mtx],
#                           tf.dtypes.float64, name='collect_ball_samples')
#     return rslt

def collect_ball_samples(double_cube_mtx, edge_len, read_threads):
    dense_shape = np.asarray([edge_len * edge_len * edge_len, N_BALL_SAMPS],
                             dtype=np.int64)
#    npzfile = np.load('zslow_precalc_sampler_full.npz')
    npzfile = np.load('precalc_sampler_full.npz')
    print('loaded!')
    index_full = npzfile['arr_0']
    vals_full = npzfile['arr_1']
    sampler_mtx = tf.SparseTensor(indices=index_full, values=vals_full,
                                  dense_shape=dense_shape)
    sampler_mtx = tf.sparse.reorder(sampler_mtx)
    sampler_mtx_T = tf.sparse.transpose(sampler_mtx)

    double_cube_mtx = tf.reshape(double_cube_mtx, 
                                 [-1, edge_len*edge_len*edge_len])
    rslt = tf.sparse_tensor_dense_matmul(sampler_mtx_T, 
                                         tf.transpose(double_cube_mtx))
    rslt = tf.transpose(rslt)
    return rslt

def writeBOV(fname_base, byte_blk, var_name):
    byte_blk.tofile(fname_base+'.bytes')
    scan_sz = byte_blk.shape
    with open(fname_base + '.bov', 'w') as f:
        f.write("TIME: 0\n")
        f.write("DATA_FILE: %s\n" % (fname_base + '.bytes'))
        f.write("DATA_SIZE: %d %d %d\n" % (scan_sz[0], scan_sz[1], scan_sz[2]))
        f.write("DATA_FORMAT: BYTE\n")
        f.write("VARIABLE: %s\n" % var_name)
        f.write("DATA_ENDIAN: LITTLE\n")
        f.write("CENTERING: ZONAL\n")
        f.write("BRICK_ORIGIN: 0.0 0.0 0.0\n")
#        f.write("BRICK_ORIGIN: %f %f %f\n" % (float(x_base + 4750), float(y_base + 2150),
#                                              float(z_start_offset + z_base + 4000)))

        f.write("BRICK_SIZE: %f %f %f\n" % (float(scan_sz[0]), float(scan_sz[1]), float(scan_sz[2])))


def reorder_array(blk):
    """
    Operations to flip the array orders to match the original data
    samples
    """
    return np.flip(blk.transpose(), 1)


def evaluate():
    """Instantiate the network, then eval for a number of steps."""

    # What volume are we scanning?
    edge_len = get_subblock_edge_len()
    scan_sz_V = np.asarray(parse_int_list(FLAGS.scan_size, 3, low_bound=[1, 1, 1]),
                           dtype=np.int)

    scan_start_V = np.asarray(parse_int_list(FLAGS.scan_start, 3, low_bound=[0, 0, 0]),
                              dtype=np.int)
    coord_base_V = scan_start_V - ((edge_len - 1)//2)  # correct for shift to center of ball


    # Generate placeholders for the images and labels.
    loc_iterator = get_loc_iterator(FLAGS.data_dir, FLAGS.batch_size, coord_base_V,
                                    scan_sz_V)

    x_off_op, y_off_op, z_off_op = loc_iterator.get_next()
    subblock = get_subblock_op(x_off_op, y_off_op, z_off_op, edge_len, get_full_block())
    subblock = tf.dtypes.cast(subblock, tf.dtypes.float64)

    images = collect_ball_samples(subblock, edge_len, read_threads=FLAGS.read_threads)
    ctrpt_op = tf.dtypes.cast(images[:,0], tf.dtypes.uint8)

    if FLAGS.random_rotation:
        # ctrpt_op is not modified; it's just being used to fill a needed parameter
        images = tf.cast(images, tf.float32)
        images, ctrpt_op = harmonics.apply_random_rotation(images, ctrpt_op)

    # Build a Graph that computes predictions from the inference model.
    logits = topology.inference(images, FLAGS.network_pattern)
    
    # Set up some prediction statistics
    predicted_op = tf.round(tf.nn.sigmoid(logits))

    saver = tf.train.Saver()

    scanned_blk, pred_blk = scan(coord_base_V, scan_sz_V, loc_iterator,
                                 x_off_op, y_off_op, z_off_op,
                                 saver, ctrpt_op, predicted_op)
    x_base, y_base, z_base = scan_start_V
    scan_sz = scanned_blk.shape
    fname_base = '%s_scanned_%d_%d_%d_%d_%d_%d' % (FLAGS.outname, x_base, y_base, z_base,
                                                   scan_sz[0], scan_sz[1], scan_sz[2])
    writeBOV(fname_base, reorder_array(scanned_blk), 'density')
    fname_base = '%s_pred_%d_%d_%d_%d_%d_%d' % (FLAGS.outname, x_base, y_base, z_base,
                                                scan_sz[0], scan_sz[1], scan_sz[2])
    writeBOV(fname_base, reorder_array(pred_blk), 'prediction')


def main(argv=None):  # pylint: disable=unused-argument
    # Should clean up directories here
    tf.gfile.MakeDirs(FLAGS.log_dir)
    evaluate()


if __name__ == '__main__':
    tf.app.run()
