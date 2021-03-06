import uuid
import argparse

# ==================== CONSTANTS ====================

OUT_IMAGE = str(uuid.uuid4()) + '.jpg'

VGG_MODEL = 'vgg19_normalized.pkl'

IMAGE_SIZE = 512	# Size of the internal handling of the image.
MAX_EPOCH = 8		# Number of epochs to run the script

NOISE_RATIO = 0.5	# Percentage of weight of the noise for intermixing with the content image.
ALPHA = 1			# Constant to put more emphasis on CONTENT LOSS.
BETA = 0.001		# Constant to put more emphasis on STYLE LOSS.
GAMMA = 0.1e-7		# Constant to put more emphasis on TOTAL VARIATION LOSS.

# ===================================================

parser = argparse.ArgumentParser(description="""
This is a class project for CC5204 Busqueda por Contenido de Imagenes y Videos (Content-Based Image and Video Retrieval).
Implementation of "A Neural Algorithm of Artistic Style" by Leon A. Gatys, Alexander S. Ecker, Matthias Bethge.
""")

parser.add_argument('content_image', type=str, help="The image to transfer the style into.", metavar='content')
parser.add_argument('style_image',	type=str, help="The image with the style to transfer.", metavar='style')

parser.add_argument('--out',	type=str, help="Path of the generated image. Defaults to a JPG saved on the script folder.", default=OUT_IMAGE, dest='out')
parser.add_argument('--vgg19',	type=str, help="Path to the VGG model. Default is %s on the script folder." % (VGG_MODEL,), default=VGG_MODEL, dest='vgg19')
parser.add_argument('--size',	type=int, help="The size of the generated image. Default is %d." % (IMAGE_SIZE,), default=IMAGE_SIZE, dest='size')
parser.add_argument('--epochs',	type=int, help="Number of iterations. Default is %d." % (MAX_EPOCH,), default=MAX_EPOCH, dest='epochs')
parser.add_argument('--noise',	type=float, help="Noise percentage of the initial guess. Default is %f." % (NOISE_RATIO,), default=NOISE_RATIO, dest='noise')
parser.add_argument('--alpha',	type=float, help="Emphasis on content loss. Default is %f." % (ALPHA,), default=ALPHA, dest='alpha')
parser.add_argument('--beta',	type=float, help="Emphasis on style loss. Default is %f." % (BETA,), default=BETA, dest='beta')
parser.add_argument('--gamma',	type=float, help="Emphasis on total variation loss. Default is %f." % (GAMMA,), default=GAMMA, dest='gamma')

args = parser.parse_args()

# Parsing the arguments before loading the libraries
# In case of any error we don't have to wait the slow loading of the heavy GPU-based libraries like Theano or Lasagne

import numpy as np
import pickle

import theano
import theano.tensor as T

import lasagne
from lasagne.utils import floatX
from lasagne.layers import InputLayer, Pool2DLayer
from lasagne.layers.dnn import Conv2DDNNLayer

from skimage.transform import resize
from skimage.io import imread, imsave
from scipy.optimize import fmin_l_bfgs_b

from time import clock
from datetime import timedelta

class StyleNet:
	MEAN_VALUES = np.array([103.939, 116.779, 123.68]).reshape((3, 1, 1))

	STYLE_WEIGHTS = {
		'conv1_1': 0.5e6,
		'conv2_1': 1.0e6,
		'conv3_1': 1.5e6,
		'conv4_1': 3.0e6,
		'conv5_1': 4.0e6 
	}

	def __init__(self,image_size,max_epoch,noise_ratio,alpha,beta,gamma,vgg):
		self._image_size = image_size
		self._max_epoch = max_epoch
		self._noise_ratio = noise_ratio

		self._alpha = alpha
		self._beta = beta
		self._gamma = gamma

		self._vgg = vgg

		self.build()

	def build(self):
		net = {}
		net['input']   = InputLayer((1, 3, self._image_size, self._image_size))
		net['conv1_1'] = Conv2DDNNLayer(net['input'],	64, 3, pad=1, flip_filters=False)
		net['conv1_2'] = Conv2DDNNLayer(net['conv1_1'],	64, 3, pad=1, flip_filters=False)
		net['pool1']   = Pool2DLayer   (net['conv1_2'],	2, mode='average_exc_pad')

		net['conv2_1'] = Conv2DDNNLayer(net['pool1'],	128, 3, pad=1, flip_filters=False)
		net['conv2_2'] = Conv2DDNNLayer(net['conv2_1'],	128, 3, pad=1, flip_filters=False)
		net['pool2']  = Pool2DLayer   (net['conv2_2'],	2, mode='average_exc_pad')

		net['conv3_1'] = Conv2DDNNLayer(net['pool2'],	256, 3, pad=1, flip_filters=False)
		net['conv3_2'] = Conv2DDNNLayer(net['conv3_1'],	256, 3, pad=1, flip_filters=False)
		net['conv3_3'] = Conv2DDNNLayer(net['conv3_2'],	256, 3, pad=1, flip_filters=False)
		net['conv3_4'] = Conv2DDNNLayer(net['conv3_3'],	256, 3, pad=1, flip_filters=False)
		net['pool3']   = Pool2DLayer   (net['conv3_4'],	2, mode='average_exc_pad')

		net['conv4_1'] = Conv2DDNNLayer(net['pool3'],	512, 3, pad=1, flip_filters=False)
		net['conv4_2'] = Conv2DDNNLayer(net['conv4_1'],	512, 3, pad=1, flip_filters=False)
		net['conv4_3'] = Conv2DDNNLayer(net['conv4_2'],	512, 3, pad=1, flip_filters=False)
		net['conv4_4'] = Conv2DDNNLayer(net['conv4_3'],	512, 3, pad=1, flip_filters=False)
		net['pool4']   = Pool2DLayer   (net['conv4_4'],	2, mode='average_exc_pad')

		net['conv5_1'] = Conv2DDNNLayer(net['pool4'],	512, 3, pad=1, flip_filters=False)
		net['conv5_2'] = Conv2DDNNLayer(net['conv5_1'],	512, 3, pad=1, flip_filters=False)
		net['conv5_3'] = Conv2DDNNLayer(net['conv5_2'],	512, 3, pad=1, flip_filters=False)
		net['conv5_4'] = Conv2DDNNLayer(net['conv5_3'],	512, 3, pad=1, flip_filters=False)
		net['pool5']   = Pool2DLayer   (net['conv5_4'],	2, mode='average_exc_pad')

		values = pickle.load(open(self._vgg, 'rb'))['param values']
		lasagne.layers.set_all_param_values(net['pool5'], values)

		self._net = net
		return net

	def process_image(self,image):
		if len(image.shape) == 2:
			image = im[:, :, np.newaxis]
			image = np.repeat(image, 3, axis=2)
		
		h, w = image.shape[:2]
		if h < w:
			image = resize(image, (self._image_size, w*self._image_size/h), preserve_range=True)
		else:
			image = resize(image, (h*self._image_size/w, self._image_size), preserve_range=True)

		h, w = image.shape[:2]
		image = image[h//2-self._image_size//2:h//2+self._image_size//2, w//2-self._image_size//2:w//2+self._image_size//2]

		# Turn image to the shape (3, self._image_size, self._image_size)
		image = np.swapaxes(np.swapaxes(image, 1, 2), 0, 1)

		image = image[::-1,:,:]
		image -= self.__class__.MEAN_VALUES

		# Output of the shape (1, 3, self._image_size, self._image_size)
		return floatX(image[np.newaxis])

	def deprocess_image(self,image):
		image = np.copy(image[0])
		image += self.__class__.MEAN_VALUES

		image = image[::-1]
		image = np.swapaxes(np.swapaxes(image, 0, 1), 1, 2)

		return np.clip(image,0,255).astype('uint8')

	def generate_noise(self,content_image,noise_ratio=None):
		if noise_ratio is None:
			noise_ratio = self._noise_ratio

		noise_image = floatX(np.random.uniform(-128, 128, (1, 3, self._image_size, self._image_size)))
		return noise_image * noise_ratio + content_image * (1 - noise_ratio)

	def content_loss(self, p, x):
		return 0.5 * ((x - p)**2).sum()

	def style_loss(self, a, x):

		def gram_matrix(x):
			x = x.flatten(ndim=3)
			return T.tensordot(x, x, axes=([2], [2]))

		A = gram_matrix(a)
		G = gram_matrix(x)
		N = a.shape[1]
		M = a.shape[2] * a.shape[3]

		return (1.0/(4 * N**2 * M**2)) * ((G - A)**2).sum()

	def total_variation_loss(self,x):
		return (((x[:, :, :-1, :-1] - x[:, :, 1:, :-1])**2 + (x[:, :, :-1, :-1] - x[:, :, :-1, 1:])**2)**1.25).sum()

	def __call__(self,content_image,style_image):
		content_image = self.process_image(content_image)
		style_image = self.process_image(style_image)

		generated_image = theano.shared( self.generate_noise(content_image) )

		layers = ['conv4_2', 'conv1_1', 'conv2_1', 'conv3_1', 'conv4_1', 'conv5_1']
		layer_vals = [ self._net[k] for k in layers ]

		input_image = T.tensor4()
		outputs = lasagne.layers.get_output(layer_vals, input_image)
		outputs = {k: v for k, v in zip(layers, outputs)}

		features = {
			'conv4_2' : theano.shared(outputs['conv4_2'].eval({input_image: content_image})),
			'conv1_1' : theano.shared(outputs['conv1_1'].eval({input_image: style_image})),
			'conv2_1' : theano.shared(outputs['conv2_1'].eval({input_image: style_image})),
			'conv3_1' : theano.shared(outputs['conv3_1'].eval({input_image: style_image})),
			'conv4_1' : theano.shared(outputs['conv4_1'].eval({input_image: style_image})),
			'conv5_1' : theano.shared(outputs['conv5_1'].eval({input_image: style_image}))
		}

		generated_features = lasagne.layers.get_output(layer_vals, generated_image)
		generated_features = {k: v for k, v in zip(layers, generated_features)}

		L_c = self.content_loss( features['conv4_2'], generated_features['conv4_2'] )

		L_s  = self.__class__.STYLE_WEIGHTS['conv1_1'] * self.style_loss( features['conv1_1'], generated_features['conv1_1'])
		L_s += self.__class__.STYLE_WEIGHTS['conv2_1'] * self.style_loss( features['conv2_1'], generated_features['conv2_1'])
		L_s += self.__class__.STYLE_WEIGHTS['conv3_1'] * self.style_loss( features['conv3_1'], generated_features['conv3_1'])
		L_s += self.__class__.STYLE_WEIGHTS['conv4_1'] * self.style_loss( features['conv4_1'], generated_features['conv4_1'])
		L_s += self.__class__.STYLE_WEIGHTS['conv5_1'] * self.style_loss( features['conv5_1'], generated_features['conv5_1'])

		L_tv = self.total_variation_loss(generated_image)

		L_total = self._alpha * L_s + self._beta * L_c + self._gamma * L_tv

		grad = T.grad(L_total, generated_image)
		
		f_loss = theano.function([], L_total)
		f_grad = theano.function([], grad)

		def eval_loss(x0):
			x0 = floatX(x0.reshape((1, 3, self._image_size, self._image_size)))
			generated_image.set_value(x0)
			return f_loss().astype('float64')

		def eval_grad(x0):
			x0 = floatX(x0.reshape((1, 3, self._image_size, self._image_size)))
			generated_image.set_value(x0)
			return np.array(f_grad()).flatten().astype('float64')

		# Reset the noise on the generated variable
		generated_image.set_value( self.generate_noise(content_image) )
		x0 = generated_image.get_value().astype('float64')

		print "Will run for %d epochs." % (self._max_epoch,)
		for i in range(self._max_epoch):
			print "Start Epoch %d\t" % (i+1,),
			old_t = clock()
			
			x0,_,_ = fmin_l_bfgs_b(eval_loss, x0.flatten(), fprime=eval_grad, maxfun=40)
			x0 = floatX(x0.reshape((1, 3, self._image_size, self._image_size)))
			#x0 = generated_image.get_value().astype('float64')
			
			new_t = clock()
			print "...\tEpoch %d finished in %s"  % (i+1,str(timedelta(seconds=int(new_t-old_t))))
		return self.deprocess_image(x0)

if __name__ == '__main__':
	content_image = imread(args.content_image, False)
	style_image = imread(args.style_image, False)

	style_transfer = StyleNet(
		image_size = args.size,
		max_epoch = args.epochs,
		noise_ratio = args.noise,
		alpha = args.alpha,
		beta = args.beta,
		gamma = args.gamma,
		vgg = args.vgg19)
	img = style_transfer(content_image, style_image)

	imsave(args.out, img)