import numpy as np 
import tensorflow as tf 
from vgg16 import vgg16
from scipy.misc import imread, imresize

def read_images(path):
	img = imread(path)
	img = imresize(img,(224,224))
#	print "read-images",img.dtype
	return img


def conv_layers(img,flag = False):
	"""
	given img, return the corresponding content_layer(tensor),style_layers(list of tensors)
	"""
#	print img.get_shape()
	img = img[None,]
	if flag == True:
		img = tf.Variable(img,dtype = tf.float32,trainable = True)
	else:
		img = tf.constant(img,dtype = tf.float32)

	vgg_img = vgg16(img,'vgg16_weights.npz',tf.Session())
	content_layer = vgg_img.conv4_2
	style_layers = [vgg_img.conv1_1,vgg_img.conv2_1,vgg_img.conv3_1,vgg_img.conv4_1,vgg_img.conv5_1]
	return content_layer,style_layers












def content_loss(img,content_img):
	img_content_layer,_ = conv_layers(img,True)
	content_img_layer,_ = conv_layers(content_img)
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







def style_loss(img,style_img):
	_,img_style_layers = conv_layers(img,True)
	_,style_img_layers = conv_layers(style_img)
	num_layers = len(style_img_layers)
	loss = 0
	for i in range(num_layers):
		img_one_layer = img_style_layers[i]
		style_img_one_layer = style_img_layers[i]
		gram_img,num_filter,size_matrix = gram_one_layer(img_one_layer)
		gram_style,_,_ = gram_one_layer(style_img_one_layer)
		loss += tf.nn.l2_loss(gram_img-gram_style)/(4*num_filter**2*size_matrix**2)
	return loss

def train(img,content_img,style_img):
	l1 = content_loss(img,content_img)
	l2 = style_loss(img,style_img)
	alpha = 1
	beta = 0.01
	loss = alpha*l1+beta*l2
	global_step = tf.Variable(0, trainable=False)
	learning_rate = tf.train.exponential_decay(learning_rate=2.0, global_step=global_step, decay_steps=100, decay_rate=0.94, staircase=True)
	train_step = tf.train.AdamOptimizer(learning_rate).minimize(loss, global_step=global_step)
	return train_step


def run(sess,content_path = 'stary_night.jpg',style_path = 'tofu.jpg'):
	content_img = read_images(content_path)
	print "run",content_img.shape
	style_img = read_images(style_path)
	img =np.random.normal(0,1,content_img.shape)
	num_iter = 100
	sess.run(tf.initialize_all_variables())
	for i in range(num_iter):
		train_step = train(img,content_img,style_img)
		sess.run(train_step)


sess = tf.Session()
run(sess)






