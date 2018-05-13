from __future__ import print_function
import numpy as np
from collections import defaultdict
from pprint import pprint
from PIL import Image
import glob
import sys
import os
import random
import math


class Network:
	LEARN_RATE = 1
	input_size = 0
	hidden_size = 0
	output_size = 0

	input_layer = []
	hidden_layer = []
	output_layer = []

	input_hidden_links = []
	hidden_output_links = []
	hidden_biases = []
	output_biases = []
	costs = []

	def __init__(self, input_size, hidden_size, output_size):
		self.input_size = input_size
		self.hidden_size = hidden_size
		self.output_size = output_size

		self.input_layer = [0*input_size]
		self.hidden_layer = [0*hidden_size]
		self.output_layer = [0*output_size]

		self.input_hidden_links = [[(np.random.randint(-100, 100)/100.0) for j in xrange(
			0, self.hidden_size)] for i in xrange(0, self.input_size)]
		self.hidden_output_links = [[(np.random.randint(-100, 100)/100.0) for k in xrange(
			0, self.output_size)] for j in xrange(0, self.hidden_size)]
		self.hidden_biases = [
			(np.random.randint(-100, 100)/100.0) for j in xrange(0, self.hidden_size)]
		self.output_biases = [
			(np.random.randint(-100, 100)/100.0) for k in xrange(0, self.output_size)]

	def fill_input(self, pixels):
		width, height, img = pixels
		res = []
		for i in xrange(width):
			for j in xrange(height):
				res.append(img[(i, j)])
		return res

	def activation_func(self, val, mode="sigmoid"):
		# return 0.08 * val
		if val > -700:
			return 1.0/(1.0 + math.exp(-1.0*val))
		else:
			return 1.0/(1.0 + math.exp(700))

	def derivative(self, val, mode="sigmoid"):
		# return 0.08
		return val*(1.0-val)

	def calc_new_node(self, layer, links, biases, j):
		res = biases[j]
		for i, node in enumerate(layer):
			res += node*links[i][j]
		return res

	def nonlin(self, layer, links, biases, new_layer_size):
		new_layer = []
		for i in xrange(new_layer_size):
			z = self.calc_new_node(layer, links, biases, i)
			new_layer.append(z)
		return new_layer

	def activate(self, layer):
		return [self.activation_func(i) for i in layer]

	def calc_cost(self, output, expected):
		res = []
		for i, j in zip(output, expected):
			res.append(0.5*((j-i)**2))
		return res

	# s*(1-s)

	def calc_delta(self, cost, layer):
		res = []
		for i, j in zip(cost, layer):
			# pprint(str(i) + "*" + str(self.derivative(j)))
			res.append(i * self.derivative(j))
		return res

	def matrix_add(self, mat1, mat2):
		res = [[mat1[i][j] + mat2[i][j] for j in xrange(len(mat1[0]))] for i in xrange(len(mat1))]
		return res

	def array_matrix_dot(self, mat1, mat2):
		res = [[0 for j in xrange(len(mat2))] for i in xrange(len(mat1))]
		for i in xrange(len(mat1)):
			for j in xrange(len(mat2)):
				res[i][j] = mat1[i]*mat2[j]
		return res

	def calc_l1_cost(self, l2_delta, hidden_output):
		# 1*12 . 12*10 => 1*10:
		res = [0] * len(hidden_output)
		for i in range(len(hidden_output)):
			for j in range(len(hidden_output[0])):
				res[i] += l2_delta[j] * hidden_output[i][j]
		return res

	def backprop(self, pixels, expected_output):
		"""
			* pixels: 2 dim array of pixels of an image, each field in xrange(0, 255)
			* expected_output:  one dim array with length of output layer, values: (0, 1)

			* updates weight of links based on cost of output
			* returns cost
		"""
		# forward propagate
		self.input_layer = self.fill_input(pixels)
		self.hidden_layer = self.activate(self.nonlin(self.input_layer, self.input_hidden_links, self.hidden_biases, self.hidden_size))
		self.output_layer = self.activate(self.nonlin(self.hidden_layer, self.hidden_output_links, self.output_biases, self.output_size))
		if (self.output_layer[0] == 0.5):
			# pprint("input layer")
			# pprint(self.input_layer)
			# pprint("hidden layer")
			# pprint(self.hidden_layer)
			# pprint("hidden_output_linkss")
			# print(self.hidden_output_links)
			exit()
		# pprint("output layer")
		# pprint(self.output_layer)
		cost_array = self.calc_cost(self.output_layer, expected_output)  # 1*10
		# pprint("output_layer")
		# pprint(self.output_layer)
		# end of forward propagate

		# error backpropagation
		l2_delta = self.calc_delta(cost_array, self.output_layer)  # 1*10
		hidden_output_change = self.array_matrix_dot(self.hidden_layer, l2_delta)  # 12*10
		# pprint(h	idden_output_change)
		self.hidden_output_links = self.matrix_add(self.hidden_output_links, hidden_output_change)  # 12*10

		l1_cost_array = self.calc_l1_cost(l2_delta, self.hidden_output_links)  # 1*12
		l1_delta = self.calc_delta(l1_cost_array, self.hidden_layer)  # 1*12
		# pprint(l1_delta)
		input_hidden_change = self.array_matrix_dot(self.input_layer, l1_delta)  # 700*12
		self.input_hidden_links = self.matrix_add(self.input_hidden_links, input_hidden_change)  # 700*12
		# end of error backpropagation

		return sum(cost_array)

	def read_image(self, image_path):
		img = Image.open(image_path)
		width, height = img.size
		return (width, height, img.load())

	def map_letter_to_array(self, letter):
		return [1 if ord(letter) == ord('A')+i else 0 for i in xrange(0, self.output_size)]

	def train(self, dataset, mode="GD"):
		"""
			* mode: SGD | GD
				gd: gradiant descent
				sgd: stochastic gradiant descent
			* dataset: an object of arrays of names of png files:
				{
					'a': [1.png, 2.png]
				}

			* returns: nothings
				runs backprop on data based on mode.
		"""
		for epoch in range(10):
			# for i in range(len(dataset['A'])):
			for letter, images in dataset.iteritems():
				for img in images:
					# print (letter)
					# print(letter)
					new_cost = self.backprop(self.read_image(
						img), self.map_letter_to_array(letter))
			pprint(str(epoch) + ": " + str(new_cost))
			self.costs.append(new_cost)
		return

	def predict(self, input_path):
		"""
			* input: a png file (or pixels, not decided yet)
			* output: returns a character between (A, J) with a probability
		"""
		pixels = self.read_image(input_path)
		self.input_layer = self.fill_input(pixels)
		self.hidden_layer = self.activate(self.nonlin(self.input_layer, self.input_hidden_links, self.hidden_biases, self.hidden_size))
		self.output_layer = self.activate(self.nonlin(self.hidden_layer, self.hidden_output_links, self.output_biases, self.output_size))
		pprint(self.output_layer)
		prob = max(self.output_layer)
		letter = chr(97 + self.output_layer.index(prob))
		return letter, prob


def get_dataset(path):
	filelists = defaultdict(list)
	for label in xrange(0, 10):
		dirname = os.path.join(path, chr(ord('A') + label))
		for file in os.listdir(dirname):
			if (file.endswith('.png')):
				fullname = os.path.join(dirname, file)
				if len(filelists[chr(ord('A') + label)]) < 10:
					if os.path.getsize(fullname) > 0:
						filelists[chr(ord('A') + label)].append(fullname)
					else:
						print('file ' + fullname + ' is empty')
	return filelists

#   labelsAndFiles = []
#   for label in xrange(0,10):
#     filelist = random.sample(filelists[label], number)
#     for filename in filelist:
#       labelsAndFiles.append((label, filename))


def main():
	net = Network(784, 16, 10)
	dataset = get_dataset('./notMNIST_small')
	net.train(dataset)
	letter, prob = net.predict(
		'./notMNIST_small/E/SWNvbmUgTFQgUmVndWxhciBJdGFsaWMgT3NGLnR0Zg==.png')
	print (letter, prob)

if __name__ == '__main__':
	main()
