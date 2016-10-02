# Neural Style Transfer

Python 2.7 (Theano, Lasagne) implementation of “A Neural Algorithm of Artistic Style” by Leon A. Gatys, Alexander S. Ecker, Matthias Bethge.

This is a class project for CC5204 Búsqueda por Contenido de Imágenes y Videos (Content-Based Image and Video Retrieval).

This project aims to serve as an introduction to Neural Networks.

## How to use

You need to download the normalized pretrained weights of the VGG19 network:
* [Lasagne](https://s3.amazonaws.com/lasagne/recipes/pretrained/imagenet/vgg19_normalized.pkl)
* [Original source](https://bethgelab.org/deepneuralart/)

## Gallery

These images were processed by the `style_transfer.py` script. It took around 20 minutes for each image on a NVIDIA GeForce 940M running on Windows.

<img src="docs/95fd3ae2-c5fe-4b1a-bd10-28cd902911ce.jpg" height="256" alt="Tuebingen/VanGogh">
<img src="docs/c4447ae5-debc-4922-bf9a-29934cf9bfb9.jpg" height="256" alt="Tuebingen/Seurat">
<img src="docs/c6385fff-bcc7-4a06-a0b7-c343e9025941.jpg" height="256" alt="Tuebingen/Mondrian">

<img src="docs/7beac151-a96d-4c00-890e-4976e4217901.jpg" height="256" alt="Cat/VanGogh">
<img src="docs/98e8009f-1443-4298-9dde-829cc9444321.jpg" height="256" alt="Cat/Seurat">
<img src="docs/31b9a95a-87a5-452d-9900-98360e9dfba1.jpg" height="256" alt="Cat/Mondrian">

## Requirements

Some of the following dependencies are requirements of other libraries and pip will automatically install them.

* Theano
* Lasagne
* CUDA and PyCUDA
* Numpy
* Scikit and Scikit-image

## Step by step installation of Theano on Python 2.7 on Windows

These are the steps to install Theano on Windows with Python 2.7. Currently you can't install CUDA on Python 3.5 but it is possible over Python 3.4.

* Install Visual Studio Community 2013 (2015 won't work as of now)
* Install CUDA
* Install Anaconda 2.7 (Or miniconda)
* `conda update conda`
* `conda update --all`
* `conda install mingw libpython`
* `pip install theano`
* Create a .theanorc in home (Your C:\Users\username) with:
	```
	[global]
	floatX = float32
	device = gpu

	[nvcc]
	flags=-LC:\Anaconda\libs
	compiler_bindir=C:\Program Files (x86)\Microsoft Visual Studio 12.0\VC\bin
	```
	Make sure to set ***flags*** `C:\Anaconda\libs` as your Anaconda install directory and ***compiler_bindir*** to the path of Visual Studio with `cl.exe` on it.
* `pip install pipwin`
* `pipwin install pycuda`
* Install [cuDNN](https://developer.nvidia.com/cudnn) from NVIDIA.
* `pip install lasagne`

## Other

Based on:

* [TensorFlow Implementation of "A Neural Algorithm of Artistic Style"](http://www.chioka.in/tensorflow-implementation-neural-algorithm-of-artistic-style#)
* [Art Style Transfer.ipynb](https://github.com/Lasagne/Recipes/blob/master/examples/styletransfer/Art%20Style%20Transfer.ipynb)
