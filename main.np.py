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
import matplotlib.pyplot as plt

EPOCH = 10000
INPUT = 85
LEARNING_RATE = 0.00001
LINEAR_FACTOR = 0.008
TEST_INPUT = 100
INPUT_SIZE = 784
HIDDEN_SIZE = 16
OUTPUT_SIZE = 10
DROPOUT = False
DROPOUT_PERCENT = 0.01
SEGMENT_FACTOR = 10
links0 = []
links1 = []
hidden_layer = [0 for i in range(HIDDEN_SIZE)]
hidden_biases = []
output_layer = [0 for i in range(OUTPUT_SIZE)]
output_biases = []
inputs = []
expected = []
test_inputs = []
test_expected = []
cost = []

def init_network():
	global links0, links1, INPUT_SIZE, HIDDEN_SIZE, OUTPUT_SIZE, hidden_biases, output_biases
	# generate weights between -1,1
	links0 = (2*np.random.random((INPUT_SIZE, HIDDEN_SIZE)) - 1)/100
	links1 = (2*np.random.random((HIDDEN_SIZE, OUTPUT_SIZE)) - 1)/100
	hidden_biases = np.random.uniform(size=(1, HIDDEN_SIZE))
	output_biases = np.random.uniform(size=(1, OUTPUT_SIZE))

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
	global inputs, expected, INPUT, test_inputs, test_expected
	limit = INPUT
	test_limit = TEST_INPUT
	count = 0
	mode = 'gheyme'
	for label in range(0, 10):
		dirname = os.path.join(path, chr(ord('A') + label))
		for file in os.listdir(dirname):
			if(count >= limit):
				mode = 'mast'
			if count >= limit + test_limit:
				mode = 'gheyme'
				count = 0
				break
			if (file.endswith('.png')):
				fullname = os.path.join(dirname, file)
				if os.path.getsize(fullname) > 0:
					if mode == "gheyme":
						inputs.append(read_image(fullname))
						expected.append(map_letter_to_array(chr(ord('A') + label)))
					elif mode == "mast":
						test_inputs.append(read_image(fullname))
						test_expected.append(chr(ord('A') + label))
				else:
					print('file ' + fullname + ' is empty')
			count += 1
	inputs = np.array(inputs)
	expected = np.array(expected)
	test_inputs = np.array(test_inputs)
	test_expected = np.array(test_expected)

def sigmoid (x):
	return 1/(1 + np.exp(-x))

def derivatives_sigmoid(x):
	return x * (1 - x)

def linear (x):
	return LINEAR_FACTOR*x

def derivatives_linear(x):
	return LINEAR_FACTOR


def calc_cost(exp, out):
	return ((exp - out)**2)/2

def train(mode="GD"):
	global inputs, expected, links0, links1, INPUT_SIZE, HIDDEN_SIZE, OUTPUT_SIZE, cost, EPOCH, INPUT, output_biases, hidden_biases
	for epoch in range(EPOCH):
		if mode == "GD":
			#forward
			hidden_layer = sigmoid(inputs.dot(links0) + hidden_biases)
			if(DROPOUT):
				hidden_layer *= np.random.binomial([np.ones((len(inputs), HIDDEN_SIZE))],1-DROPOUT_PERCENT)[0] * (1.0/(1-DROPOUT_PERCENT))
			output_layer = sigmoid(hidden_layer.dot(links1) + output_biases)
			# hidden_layer = linear(inputs.dot(links0) + hidden_biases)
			# output_layer = linear(hidden_layer.dot(links1) + output_biases)
			cost.append(np.sum(calc_cost(expected, output_layer))/(INPUT))
			# pprint(cost[-1])

			#backprop
			output_delta = (expected - output_layer)
			hidden_delta = output_delta.dot(links1.T) * derivatives_sigmoid(hidden_layer)
			# hidden_delta = output_delta.dot(links1.T) * derivatives_linear(hidden_layer)
			links1 += hidden_layer.T.dot(output_delta) * LEARNING_RATE
			links0 += inputs.T.dot(hidden_delta) * LEARNING_RATE
			output_biases += np.sum(output_delta, axis=0, keepdims=True) * LEARNING_RATE
			hidden_biases += np.sum(hidden_delta, axis=0, keepdims=True) * LEARNING_RATE
			# for ind, out in enumerate(output_layer):
				# print(str(ind) + ": " + str(out[ind]))
			# pprint(output_layer)
			# print("links0: " + str(links0.shape))
			# print("links1: " + str(links1.shape))
			# print("hidden_biases: " + str(hidden_biases.shape))
			# print("output_biases: " + str(output_biases.shape))
			# print("inputs: " + str(inputs.shape))
			# print("hidden_layer: " + str(hidden_layer.shape))
			# print("output_layer: " + str(output_layer.shape))
			# exit()
		elif mode=="SGD":
			for ind in range(0, SEGMENT_FACTOR):
				#forward
				inp = np.array(inputs[ind::SEGMENT_FACTOR])
				expec = np.array(expected[ind::SEGMENT_FACTOR])
				hidden_layer = sigmoid(inp.dot(links0) + hidden_biases)
				if(DROPOUT):
					hidden_layer *= np.random.binomial([np.ones((len(inp), HIDDEN_SIZE))],1-DROPOUT_PERCENT)[0] * (1.0/(1-DROPOUT_PERCENT))
				output_layer = sigmoid(hidden_layer.dot(links1) + output_biases)
				# hidden_layer = linear(inp.dot(links0) + hidden_biases)
				# output_layer = linear(hidden_layer.dot(links1) + output_biases)
				cost.append(np.sum(calc_cost(expec, output_layer))/(INPUT))
				# pprint(cost[-1])

				#backprop
				output_delta = expec - output_layer
				hidden_delta = output_delta.dot(links1.T) * derivatives_sigmoid(hidden_layer)
				# hidden_delta = output_delta.dot(links1.T) * derivatives_linear(hidden_layer)
				links1 += hidden_layer.T.dot(output_delta) * LEARNING_RATE
				links0 += inp.T.dot(hidden_delta) * LEARNING_RATE
				output_biases += np.sum(output_delta, axis=0, keepdims=True) * LEARNING_RATE
				hidden_biases += np.sum(hidden_delta, axis=0, keepdims=True) * LEARNING_RATE


		

def predict():
	global links0, links1, hidden_biases, output_biases, test_inputs, test_expected
	# pprint("links0")
	# pprint(links0)
	# pprint("links1")
	# pprint(links1)
	# pprint("hidden_biases")
	# pprint(hidden_biases)
	# pprint("output_biases")
	# pprint(output_biases)
	corrects = 0
	for test_input, expec in zip(test_inputs, test_expected):
		hidden_layer = sigmoid(test_input.dot(links0) + hidden_biases)
		output_layer = sigmoid(hidden_layer.dot(links1) + output_biases)
		prob = max(output_layer[0].tolist())
		letter = chr(ord('A') + output_layer[0].tolist().index(prob))
		pprint(output_layer[0].tolist())
		print(letter, expec)
		if letter == expec:
			corrects += 1
	pprint("correct rate: " + str(corrects) + "/" + str(TEST_INPUT*10))


def main():
	get_dataset('./notMNIST_small')
	init_network()
	train(mode="SGD")
	predict()
	plt.plot(cost)
	plt.ylabel("error")
	plt.xlabel("iteration")
	plt.show()

if __name__ == '__main__':
	main()
