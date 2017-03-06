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

"""Routine for decoding the CIFAR-10 binary file format."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os

from six.moves import xrange  # pylint: disable=redefined-builtin
import tensorflow as tf
import numpy as np

# Process images of this size. Note that this differs from the original CIFAR
# image size of 32 x 32. If one alters this number, then the entire model
# architecture will change and any model would need to be retrained.
IMAGE_HEIGHT=360
IMAGE_WIDTH=480
NUM_EXAMPLES_PER_FOLDER=367
#IMAGE_DIR="/pylon2/sy4s8lp/isn4/project/z-slices/flip0"
#LABEL_DIR="/pylon2/sy4s8lp/isn4/project/fish_output/z-output/"


# Global constants for fish slices.
LABEL_SIZE = 360*480
NUM_EXAMPLES_PER_EPOCH_FOR_TRAIN = NUM_EXAMPLES_PER_EPOCH_FOR_EVAL = 100



def readFish(img_queue,lbl_queue):
  #read files from directory
  
  #use a class to save info
  class records(object): pass
  our_input=records()

  our_input.height=360
  our_input.width=480
  our_input.depth=3

  #wholefilereader can read one entire file each time we call .read()
  img_reader= tf.WholeFileReader()
  lbl_reader= tf.WholeFileReader()

  _,image=img_reader.read(img_queue)
  decode_image = tf.image.decode_png(image)
  our_input.image=tf.reshape(decode_image,shape=[our_input.height,our_input.width,our_input.depth])
#read labels
  our_input.key,label=lbl_reader.read(lbl_queue)
  decode_label = tf.image.decode_png(label)
  reshaped_label=tf.reshape(decode_label,shape=[our_input.height,our_input.width,1])
  our_input.label=modLabel(tf.cast(reshaped_label,tf.int16))
  return our_input

# changing ignorable class "11" into a "-1" as req. by tensorflow
def modLabel(label):
  eleven= tf.constant(
    np.array([11]*IMAGE_HEIGHT*IMAGE_WIDTH).reshape(IMAGE_HEIGHT,IMAGE_WIDTH,1),
    dtype=tf.int16)

  mask11= tf.equal(label,eleven)

  negone= tf.constant(
    np.array([-1]*IMAGE_HEIGHT*IMAGE_WIDTH).reshape(IMAGE_HEIGHT,IMAGE_WIDTH,1),
    dtype=tf.int16)

  return tf.select(mask11,negone,label)


def batchGenerator(image,label,batch_size,min_num,shuffle):
  #get a batch

  
  num_threads=20

  if shuffle:
    images,labels=tf.train.shuffle_batch(
          [image, label],
          batch_size=batch_size,
          num_threads=num_threads,
          capacity=min_num + 3 * batch_size+20,
          min_after_dequeue=min_num)
  else:
    images,labels=tf.train.batch(
          [image, label],
          batch_size=batch_size,
          num_threads=num_threads,
          capacity=min_num + 3 * batch_size+20)



  # Display the training images in the visualizer.
  tf.image_summary('images', images)
  tf.image_summary('labels', labels)

  print ("batch of images",images.get_shape())
  print ("batch of labels",labels.get_shape())


  return images,tf.reshape(labels,[batch_size,IMAGE_HEIGHT,IMAGE_WIDTH,1])


def inputsROOT(file_dir,batch_size,shuffle,evall):
  
  num_examples_per_epoch = NUM_EXAMPLES_PER_EPOCH_FOR_TRAIN

  #a list of image file names, nested loop since we need each 5 names off by one

  prefix="/pylon1/sy4s8lp/anniez"

  filenames=open(file_dir,"r")
  imgL=[]
  lblL=[]
  for line in filenames:
    img,label=line.split(" ")
    imgL.append(prefix+img[19:])
    lblL.append(prefix+label[19:-1])


  #create queues for imagenames & labelnames. 
  img_queue=tf.train.string_input_producer(imgL)
  lbl_queue=tf.train.string_input_producer(lblL)


  #get one example 
  our_input=readFish(img_queue,lbl_queue)

  float_image = tf.cast(our_input.image, tf.float32)


  # Ensure that the random shuffling has good mixing properties.
  min_fraction_of_examples_in_queue = 0.4
  min_num = int(num_examples_per_epoch *
                           min_fraction_of_examples_in_queue)

  # Generate a batch of images and labels by building up a queue of examples.
  return batchGenerator(float_image, our_input.label,
                                          batch_size,min_num,
                                         shuffle=shuffle)