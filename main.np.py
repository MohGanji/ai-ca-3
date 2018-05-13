import numpy as np
from pprint import pprint
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

for j in range(60000):
    l1 = 1/(1+np.exp(-(np.dot(X, syn0))))
    l2 = 1/(1+np.exp(-(np.dot(l1, syn1))))
    l2_delta = (y - l2)*(l2*(1-l2))
    l1_delta = l2_delta.dot(syn1.T) * (l1 * (1-l1))
    syn1 += l1.T.dot(l2_delta)
    syn0 += X.T.dot(l1_delta)

pprint(l2)
pprint(y)

INPUT_SIZE = 784
HIDDEN_SIZE = 16
OUTPUT_SIZE = 10
links0 = []
links1 = []
hidden_layer = [0 for i in xrange(HIDDEN_SIZE)]
output_layer = [0 for i in xrange(OUTPUT_SIZE)]
inputs = []
expected = []

def init_network():
	global links0, links1, INPUT_SIZE, HIDDEN_SIZE, OUTPUT_SIZE
	# generate weights between -1,1
	links0 = 2*np.random.random((INPUT_SIZE, HIDDEN_SIZE)) - 1
	links1 = 2*np.random.random((HIDDEN_SIZE, OUTPUT_SIZE)) - 1


def read_image(image_path):
		img = Image.open(image_path)
		width, height = img.size
		img = img.load()
		return np.array([img[i][j] for i in xrange(width) for j in xrange(height)])

def map_letter_to_array(letter):
		return [1 if ord(letter) == ord('A')+i else 0 for i in xrange(0, self.output_size)]

def get_dataset(path):
	global inputs, expected
	limit = 10	
	count = 0
	for label in xrange(0, 10):
		dirname = os.path.join(path, chr(ord('A') + label))
		for file in os.listdir(dirname):
			if(count > limit):
				count = 0
				break
			if (file.endswith('.png')):
				fullname = os.path.join(dirname, file)
				if len(inputs) < 10:
					if os.path.getsize(fullname) > 0:
						inputs.append(read_image(fullname))
						expected.append(map_letter_to_array(chr(ord('A') + label)))
					else:
						print('file ' + fullname + ' is empty')
			count += 1

def train():
	global inputs, expected, links0, links1, INPUT_SIZE, HIDDEN_SIZE, OUTPUT_SIZE
	for epoch in xrange(1000):
		#forward
		hidden_layer = 1/(1+np.exp(-(np.dot(X, syn0))))
    	output_layer = 1/(1+np.exp(-(np.dot(l1, syn1))))

def main():
	
	get_dataset('./notMNIST_small')
	train()
	letter, prob = predict(
		'./notMNIST_small/E/SWNvbmUgTFQgUmVndWxhciBJdGFsaWMgT3NGLnR0Zg==.png')
	print (letter, prob)


if __name__ == '__main__':
	main()
