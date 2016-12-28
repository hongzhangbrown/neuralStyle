import numpy as np 
import tensorflow as tf 
from vgg16 import vgg16
from scipy.misc import imread, imresize,imsave
import matplotlib.pyplot as plt
import cProfile


mean = np.array([123.68, 116.779, 103.939])
def read_images(path):
	img = imread(path)
	img = imresize(img,(512,512))
	img = img[None,]
	img = img.astype(float)
#	print "read-images",img.dtype
	return img


def conv_layers(img,flag = False):
	"""
	given img, return the corresponding content_layer(tensor),style_layers(list of tensors)
	set the tensor to be untrainable if it is the reference image
	"""
#	print img.get_shape()
	
	if flag == True:
		img = tf.constant(img, dtype = tf.float32)
	# else:
	# if flag == False:
		# img = tf.constant(img,dtype = tf.float32)

	vgg_img = vgg16(img, weight_file = 'vgg16_weights.npz')
	content_layer = vgg_img.conv4_2
	style_layers = [vgg_img.conv1_1,vgg_img.conv2_1,vgg_img.conv3_1,vgg_img.conv4_1,vgg_img.conv5_1]
	return content_layer,style_layers

def preprocess_layer(layer):
	with tf.Session() as sess:
		temp = sess.run(layer)
	return temp

def preprocess_layers(layers):
	temp = []
	with tf.Session() as sess:
		for i in range(len(layers)):
			temp.append(sess.run(layers[i]))
	return temp		


def content_loss(img_content_layer,content_img_layer):
	"""
	compute content loss from conv4_4
	"""
	content_loss_op = tf.nn.l2_loss(img_content_layer-content_img_layer)
	return content_loss_op


def gram_one_layer(one_layer):
	"""
	compute matrix.T dot matrix
	"""
	if type(one_layer).__module__ == np.__name__:
		one_layer_dim = one_layer.shape
	else:
		one_layer_dim = np.array(one_layer.get_shape().as_list())
	num_filter = one_layer_dim[-1]
	one_layer = tf.transpose(one_layer,[3,0,1,2])
	one_layer = tf.reshape(one_layer,[num_filter,-1])
	num_ele_matrix = np.prod(one_layer_dim[:3])
	gram  = tf.matmul(one_layer,tf.transpose(one_layer))
	return gram,num_filter,num_ele_matrix


def style_loss(img_style_layers,style_img_layers):
	"""
	compute style loss
	"""
	num_layers = len(style_img_layers)
	loss = 0
	for i in xrange(num_layers):
		img_one_layer = img_style_layers[i]
		style_img_one_layer = style_img_layers[i]
		gram_img,num_filter,size_matrix = gram_one_layer(img_one_layer)
		gram_style,_,_ = gram_one_layer(style_img_one_layer)
		loss += tf.nn.l2_loss(gram_img-gram_style)/(4*num_filter**2*size_matrix**2)
	return loss

def train(loss_):
	"""
	train the model using adamoptimizer
	"""
	tf.summary.scalar('loss',loss_)
	# sess.run(tf.initialize_all_variables())
	global_step = tf.Variable(0,name = 'global_step', trainable=False)
	learning_rate = tf.train.exponential_decay(learning_rate =2.0,global_step = global_step,decay_steps = 10,decay_rate = 0.94,staircase = True)
	optimizer = tf.train.AdamOptimizer(learning_rate, beta1 = 0.9,beta2 = 0.999,epsilon = 1e-08)

	#learning_rate = tf.train.exponential_decay(learning_rate=2.0, global_step=global_step, decay_steps=100, decay_rate=0.94, staircase=True)
	#train_step =optimizer.minimize(loss_, global_step=global_step)
	train_step = optimizer.minimize(loss_)
	return train_step


# def loss(img_content_layer,img_style_layers,content_img_layer,style_img_layers):
	"""
	combine content loss and style loss
	"""
	# l1 = content_loss(img_content_layer,content_img_layer)
	# l2 = style_loss(img_style_layers,style_img_layers)
	# alpha = 1 
	# beta = 0.01
	# loss_ = alpha*l1+beta*l2
	# return loss_


	





def run(content_path = 'emma_watson.jpg',style_path = 'william_turner.jpeg'):
	"""
	compute the target image with random initialization
	"""
	
	content_img = read_images(content_path)
	style_img = read_images(style_path)
#	img =np.random.normal(0,10**(-3),content_img.shape)
	img = read_images('trained_emma.jpg')
	num_iter =1
	g = tf.Graph()
	with g.device("/cpu:0"),g.as_default(),tf.Session(graph = g,config = tf.ConfigProto(allow_soft_placement=True)) as sess:
		img = tf.Variable(img, dtype = tf.float32,trainable = True)
		# img =tf.Variable(style_img,dtype = tf.float32,trainable = True)
		content_img_layer,_ = conv_layers(content_img,flag = True)
		_,style_img_layers = conv_layers(style_img,flag = True)

		content_img_layer = preprocess_layer(content_img_layer)
		style_img_layers = preprocess_layers(style_img_layers)


		img_content_layer,img_style_layers = conv_layers(img,flag = False)

		alpha = 1
		beta = 0.01

		# print type(content_img_layer),type(style_img_layers[0])

		loss_ = alpha*content_loss(img_content_layer,content_img_layer)+beta*style_loss(img_style_layers,style_img_layers)
		train_step = train(loss_)
		init = tf.global_variables_initializer()
		sess.run(init)	
		for i in xrange(num_iter):
			print "i",i
			print "loss",sess.run(loss_)
			sess.run(train_step)

	
		image = sess.run(img)[0]+mean
		image = image.astype('uint8')
		imsave("trained_test.jpg",image)	
	#plt.imshow(B)
	#plt.show()
if __name__=="__main__":
	run()




