from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os

from six.moves import xrange
import tensorflow as tf
import numpy as np

BATCH_SIZE=4
MIN_NUM_INQUEUE=1
RUN_STEP=2
NUM_EXAMPLES=20
IMAGE_HEIGHT=360
IMAGE_WIDTH=480
LOAD_DIR="/pylon1/sy4s8lp/anniez/project/caffe-segnet/tutorial/CamVid/old-train.txt"


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

def modLabel(label):
  eleven= tf.constant(
    np.array([11]*IMAGE_HEIGHT*IMAGE_WIDTH).reshape(IMAGE_HEIGHT,IMAGE_WIDTH,1),
    dtype=tf.int16)

  mask11= tf.equal(label,eleven)

  negone= tf.constant(
    np.array([-1]*IMAGE_HEIGHT*IMAGE_WIDTH).reshape(IMAGE_HEIGHT,IMAGE_WIDTH,1),
    dtype=tf.int16)

  return tf.select(mask11,negone,label)

def batchGenerator(image,label,batch_size,min_num):
  #get a batch
  
  num_threads=20

  images,labels=tf.train.shuffle_batch(
        [image, label],
        batch_size=batch_size,
        num_threads=num_threads,
        capacity=min_num + 3 * batch_size+20,
        min_after_dequeue=min_num)

  # Display the training images in the visualizer.
  tf.image_summary('images', images)

  return images,tf.reshape(labels,[batch_size,IMAGE_HEIGHT,IMAGE_WIDTH])



def inputs(file_dir,batch_size):
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

  #get a batch
  img,lbl=batchGenerator(our_input.image,our_input.label,BATCH_SIZE,MIN_NUM_INQUEUE)

  return tf.argmax(img,3),lbl


def reader():

  #create a graph
  with tf.Graph().as_default():
    getData=inputs(LOAD_DIR,BATCH_SIZE)

    #create a session to run the graph, and some initialization here
    init=tf.initialize_all_variables()
    sess = tf.Session(config=tf.ConfigProto(
        log_device_placement=False))
    sess.run(init)
    tf.train.start_queue_runners(sess=sess)

    for step in xrange(RUN_STEP):
      print("step",step)
      #get batches here
      img,lbl=sess.run(getData)
      print (img,"imgshape",img.shape)

      print ("YAYAYAYAYAY")
      print(lbl,"lblshape",lbl.shape)
      

reader()