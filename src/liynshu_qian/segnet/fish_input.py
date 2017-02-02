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

# Process images of this size. Note that this differs from the original CIFAR
# image size of 32 x 32. If one alters this number, then the entire model
# architecture will change and any model would need to be retrained.
IMAGE_SIZE=1024
NUM_EXAMPLES_PER_FOLDER=10#int(4900/2 -2)
#IMAGE_DIR="/pylon2/sy4s8lp/isn4/project/z-slices/flip0"
#LABEL_DIR="/pylon2/sy4s8lp/isn4/project/fish_output/z-output/"


# Global constants for fish slices.
LABEL_SIZE = 1024*1024
NUM_EXAMPLES_PER_EPOCH_FOR_TRAIN = NUM_EXAMPLES_PER_EPOCH_FOR_EVAL = 16000



def readFish(image_queue,label_queue):
  #read files from directory

  
  #use a class to save info
  class records(object): pass
  our_input=records()

  our_input.height=1024
  our_input.width=1024
  our_input.depth=5

  #wholefilereader can read one entire file each time we call .read()
  image_reader= tf.WholeFileReader()
  label_reader= tf.WholeFileReader()

  #hard coded a bit. read 5 images by calling it 5 times
  image5=[]
  for i in xrange(our_input.depth):
    _,imagei=image_reader.read(image_queue)
    image=tf.decode_raw(imagei,tf.uint8)
    image5.append(image)

  #put 5 tensors into one
  result=tf.pack(image5)

  #read labels
  our_input.key,label=label_reader.read(label_queue)
  record_label=tf.decode_raw(label,tf.uint8)

  #reshape all
  our_input.label=tf.reshape(record_label,
                             [our_input.height,our_input.width])

  
  almost_image=tf.reshape(result,
                          [our_input.depth,our_input.height,our_input.width])

  # Convert from [depth, height, width] to [height, width, depth].
  our_input.image=tf.transpose(almost_image, [1, 2, 0])

  print ("one 5-images sample",our_input.image)
  return our_input


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
  tf.image_summary('images', images[:,:,:,1:4])

  print ("batch of images")
  print ("batch of labels")


  return images,tf.reshape(labels,[batch_size,24,24])#IMAGE_SIZE,IMAGE_SIZE])


def inputsROOT(image_dir,label_dir,batch_size,shuffle,evall):
  
  num_examples_per_epoch = NUM_EXAMPLES_PER_EPOCH_FOR_TRAIN

  #a list of image file names, nested loop since we need each 5 names off by one

  imagenames=[]
  labelnames=[]
  if evall=="eval":
    addon=1
  else:
    addon=0
  folderList=["flip0/","flip1/","flip2/","flip3/","rot0/","rot1/","rot2/","rot3/"]
  for foldername in folderList:
    im_dir=image_dir+foldername
    lbl_dir=label_dir+foldername

    for i in xrange(NUM_EXAMPLES_PER_FOLDER):
      labelNum=2+i*2+addon
      #print ("generated %d many exmaple names, this might take forever."%i)
      imagenames+=[os.path.join(im_dir,"z%s"%str(j).zfill(4)) 
                     for j in xrange(labelNum-2,labelNum+3)]
      labelnames+=[os.path.join(lbl_dir,"z%s"%str(labelNum).zfill(4))]

  #create queues for imagenames & labelnames. 
  image_queue=tf.train.string_input_producer(imagenames)
  label_queue=tf.train.string_input_producer(labelnames)

  #get one example 
  our_input=readFish(image_queue,label_queue)

  float_image = tf.cast(our_input.image, tf.float32)

  height = 24
  width = 24
  distorted_image = tf.random_crop(float_image, [height, width, 5])

  distorted_label =tf.random_crop(our_input.label, [height, width])

  # Ensure that the random shuffling has good mixing properties.
  min_fraction_of_examples_in_queue = 0.4
  min_num = int(num_examples_per_epoch *
                           min_fraction_of_examples_in_queue)

  # Generate a batch of images and labels by building up a queue of examples.
  return batchGenerator(distorted_image, distorted_label,
                                          batch_size,min_num,
                                         shuffle=shuffle)




  