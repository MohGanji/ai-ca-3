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
# import matplotlib.pyplot as plt

EPOCH = 10000
INPUT = 10

INPUT_SIZE = 784
HIDDEN_SIZE = 16
OUTPUT_SIZE = 10
links0 = []
links1 = []
hidden_layer = [0 for i in range(HIDDEN_SIZE)]
output_layer = [0 for i in range(OUTPUT_SIZE)]
inputs = []
expected = []
cost = []

def init_network():
	global links0, links1, INPUT_SIZE, HIDDEN_SIZE, OUTPUT_SIZE
	# generate weights between -1,1
	links0 = 2*np.random.random((INPUT_SIZE, HIDDEN_SIZE)) - 1
	links1 = 2*np.random.random((HIDDEN_SIZE, OUTPUT_SIZE)) - 1

def read_image(image_path):
		img = Image.open(image_path)
		width, height = img.size
		pixels = img.load()
		flatten_pixels = [pixels[(i, j)] for i in range(0, width) for j in range(0, height)]
		return np.array(flatten_pixels)

def map_letter_to_array(letter):
	global OUTPUT_SIZE
	return [1 if ord(letter) == ord('A')+i else 0 for i in range(0, OUTPUT_SIZE)]

def get_dataset(path):
	global inputs, expected, INPUT
	limit = INPUT
	count = 0
	for label in range(0, 10):
		dirname = os.path.join(path, chr(ord('A') + label))
		for file in os.listdir(dirname):
			if(count >= limit):
				count = 0
				break
			if (file.endswith('.png')):
				fullname = os.path.join(dirname, file)
				if os.path.getsize(fullname) > 0:
					inputs.append(read_image(fullname))
					expected.append(map_letter_to_array(chr(ord('A') + label)))
				else:
					print('file ' + fullname + ' is empty')
			count += 1
	inputs = np.array(inputs)
	print(len(inputs))
	expected = np.array(expected)

def sigmoid (x):
	return 1/(1 + np.exp(-x))

def derivatives_sigmoid(x):
	return x * (1 - x)

def calc_cost(exp, out):
	return ((exp - out)**2)/2

def train():
	global inputs, expected, links0, links1, INPUT_SIZE, HIDDEN_SIZE, OUTPUT_SIZE, cost, EPOCH, INPUT
	for epoch in range(EPOCH):
		#forward
		hidden_layer = sigmoid(inputs.dot(links0))
		output_layer = sigmoid(hidden_layer.dot(links1))
		cost.append(np.sum(calc_cost(expected, output_layer))/(INPUT))
		pprint(cost[-1])

		#backprop
		output_delta = (expected - output_layer)*derivatives_sigmoid(output_layer)
		hidden_delta = output_delta.dot(links1.T) * derivatives_sigmoid(hidden_layer)
		links1 += hidden_layer.T.dot(output_delta)
		links0 += inputs.T.dot(hidden_delta)
	# plt.plot(cost)
		

def predict(input_path):
	global links0, links1
	test_input = np.array([read_image(input_path)])
	hidden_layer = sigmoid(test_input.dot(links0))
	output_layer = sigmoid(hidden_layer.dot(links1))
	pprint(output_layer[0].tolist())
	prob = max(output_layer[0].tolist())
	letter = chr(ord('A') + output_layer[0].tolist().index(prob))
	return letter, prob
	

def main():
	get_dataset('./notMNIST_small')
	init_network()
	train()
	letter, prob = predict(
		'./notMNIST_small/J/MDEtMDEtMDAudHRm.png')
	print (letter, prob)

if __name__ == '__main__':
	main()
