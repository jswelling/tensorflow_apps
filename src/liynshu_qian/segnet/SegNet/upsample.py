from __future__ import absolute_import
from __future__ import division
from __future__ import print_function


import numpy as np
from six.moves import xrange  # pylint: disable=redefined-builtin
import tensorflow as tf

def upsample(inputs,pool,original,scale,pad_out_h=False):
  shape=inputs.get_shape().as_list()
  b=shape[0]
  h=shape[1]
  w=shape[2]
  c=shape[3]

  zero_np=np.zeros(b*(h*scale)*(w*scale)*c)
  zero_flat=tf.Variable(zero_np,dtype=tf.float32)
  zero_cnst=tf.reshape(zero_flat,shape=[b,h*scale,w*scale,c])
  upsampool=tf.image.resize_images(pool,h*scale, w*scale, method=tf.image.ResizeMethod.NEAREST_NEIGHBOR)
  if pad_out_h:
    original=tf.pad(original,paddings=[[0,0],[0,1],[0,0],[0,0]])
  mask= tf.equal(upsampool,original)
  upsampl=tf.image.resize_images(inputs,h*scale, w*scale, method=tf.image.ResizeMethod.NEAREST_NEIGHBOR)  
  result=tf.select(mask,upsampl,zero_cnst)
  if pad_out_h:
  	result=tf.slice(result,[0,0,0,0],[b,h*scale-1,w*scale,c]) 

  return result









def pooling():

	#create a graph
	with tf.Graph().as_default():

		thing=tf.random_uniform(shape=[1,3,4,1],seed=1)
		inputs=tf.random_uniform(shape=[1,2,2,1],seed=1)

		pool= tf.nn.max_pool(thing, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1],
                         padding='SAME', name='pool')
		#print (pool,pool_mask)
		upsamp=upsample(inputs,pool,thing,2,True)

		#create a session to run the graph, and some initialization here
		init=tf.initialize_all_variables()
		sess = tf.Session()
		sess.run(init)
		tf.train.start_queue_runners(sess=sess)

		for step in xrange(1):
			x,i,y,o=sess.run([thing,inputs,pool,upsamp])

			print ("thing",x,"shape",x.shape)

			print ("pool",y,"shape",y.shape)		
			print ("input",i,"shape",i.shape)


			print ("upsamp",o,"shape",o.shape)



pooling()




