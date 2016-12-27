import numpy as np 
import tensorflow as tf 
from vgg16 import vgg16
from scipy.misc import imread, imresize,imsave
import matplotlib.pyplot as plt

mean = np.array([123.68, 116.779, 103.939])
def read_images(path):
	img = imread(path)
	img = imresize(img,(224,224))
	img = img[None,]
	img = img.astype(float)
#	print "read-images",img.dtype
	return img


def conv_layers(img,flag = False):
	"""
	given img, return the corresponding content_layer(tensor),style_layers(list of tensors)
	"""
#	print img.get_shape()
	
	if flag == True:
		img = tf.Variable(img, dtype = tf.float32,trainable = True)
	# else:
	# if flag == False:
		# img = tf.constant(img,dtype = tf.float32)

	vgg_img = vgg16(img, weight_file = 'vgg16_weights.npz')
	content_layer = vgg_img.conv4_2
	style_layers = [vgg_img.conv1_1,vgg_img.conv2_1,vgg_img.conv3_1,vgg_img.conv4_1,vgg_img.conv5_1]
	return content_layer,style_layers


def content_loss(img_content_layer,content_img_layer):
	content_loss_op = tf.nn.l2_loss(img_content_layer-content_img_layer)
	return content_loss_op


def gram_one_layer(one_layer):
	one_layer_dim = np.array(one_layer.get_shape().as_list())
	num_filter = one_layer_dim[-1]
	one_layer = tf.transpose(one_layer,[3,0,1,2])
	one_layer = tf.reshape(one_layer,[num_filter,-1])
	num_ele_matrix = np.prod(one_layer_dim[:3])
	gram  = tf.matmul(one_layer,tf.transpose(one_layer))
	return gram,num_filter,num_ele_matrix


def style_loss(img_style_layers,style_img_layers):
	num_layers = len(style_img_layers)
	loss = 0
	for i in range(num_layers):
		img_one_layer = img_style_layers[i]
		style_img_one_layer = style_img_layers[i]
		gram_img,num_filter,size_matrix = gram_one_layer(img_one_layer)
		gram_style,_,_ = gram_one_layer(style_img_one_layer)
		loss += tf.nn.l2_loss(gram_img-gram_style)/(4*num_filter**2*size_matrix**2)
	return loss

def train(loss_,learning_rate):
	tf.summary.scalar('loss',loss_)
	# sess.run(tf.initialize_all_variables())
	global_step = tf.Variable(0,name = 'global_step', trainable=False)
	learning_rate = tf.train.exponential_decay(learning_rate =0.01,global_step = global_step,decay_steps = 10,decay_rate = 0.94,staircase = True)
	optimizer = tf.train.AdamOptimizer(learning_rate, beta1 = 0.9,beta2 = 0.999,epsilon = 1e-08)

	#learning_rate = tf.train.exponential_decay(learning_rate=2.0, global_step=global_step, decay_steps=100, decay_rate=0.94, staircase=True)
	#train_step =optimizer.minimize(loss_, global_step=global_step)
	train_step = optimizer.minimize(loss_)
	return train_step


def loss(img_content_layer,img_style_layers,content_img_layer,style_img_layers):
	l1 = content_loss(img_content_layer,content_img_layer)
	l2 = style_loss(img_style_layers,style_img_layers)
	alpha = 1
	beta = 1
	loss_ = alpha*l1+beta*l2
	return loss_

def modify(img,sess):
	matrix = sess.run(img)
	shape = matrix.shape
	flat  =np.array(matrix.flat)
	flat[flat<0] = 0
	flat[flat>255] = 255
	matrix_ = np.reshape(flat,shape)
	return matrix_
	





def run(content_path = 'cat.jpg',style_path = 'stary_night.jpg'):
	content_img = read_images(content_path)
	style_img = read_images(style_path)
	img =np.random.normal(0,10**(-3),content_img.shape)
	num_iter =10 
	g = tf.Graph()
	with g.device("/gpu:0"),g.as_default(),tf.Session(graph = g,config = tf.ConfigProto(allow_soft_placement=True)) as sess:
		img = tf.Variable(img, dtype = tf.float32,trainable = True)
#		img =tf.Variable(content_img,dtype = tf.float32,trainable = True)
		content_img_layer,_ = conv_layers(content_img,flag = True)
		_,style_img_layers = conv_layers(style_img,flag = True)
		img_content_layer,img_style_layers = conv_layers(img,flag = False)
		loss_ = loss(img_content_layer,img_style_layers,content_img_layer,style_img_layers)
		train_step = train(loss_,10**(-5))
		init = tf.global_variables_initializer()
		sess.run(init)	
#		plt.imshow(sess.run(img)[0])
#		plt.show()

		# print sess.run(img)
		for i in xrange(num_iter):
			print "i",i
			print "loss",sess.run(loss_)
			sess.run([train_step,loss_])
		image = sess.run(img)[0]+mean
		image = image.astype('uint8')
	imsave("trained_imagei.jpg",image)	
	#plt.imshow(B)
	#plt.show()
if __name__=="__main__":
	run()




