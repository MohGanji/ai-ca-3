from __future__ import print_function
import numpy as np
from collections import defaultdict
from pprint import pprint
from PIL import Image
import  glob, sys, os, random

class Network:
    input_size= 0
    hidden_size= 0
    output_size= 0

    input_layer= []
    hidden_layer= []
    output_layer= []

    input_hidden_links = []
    hidden_output_links = []
    hidden_bias = []
    output_bias = []
    costs = []

    def __init__(self, input_size, hidden_size, output_size):
        self.input_size= input_size
        self.hidden_size= hidden_size
        self.output_size= output_size 

        self.input_layer= [0*input_size]
        self.hidden_layer= [0*hidden_size]
        self.output_layer= [0*output_size]

        self.input_hidden_links = [ [np.random.uniform(-1, 1) for j in xrange(0, self.hidden_size)] for i in xrange(0, self.input_size)]
        self.hidden_output_links = [ [np.random.uniform(-1, 1) for k in xrange(0, self.output_size)] for j in xrange(0, self.hidden_size)]
        self.hidden_bias = [np.random.uniform(-1, 1) for j in xrange(0, self.hidden_size)]
        self.output_bias = [np.random.uniform(-1, 1) for k in xrange(0, self.output_size)]

        # pprint(self.input_hidden_links)
        # pprint(self.hidden_output_links)
        
    def fill_input(self, pixels):
        res = []
        for cols in pixels:
            for pixel in cols:
                res.append(pixel/255.0)
        return res

    def backprop(self, pixels, expected_output):
        """
            * pixels: 2 dim array of pixels of an image, each field in range(0, 255)
            * expected_output:  one dim array with length of output layer, values: (0, 1)

            * updates weight of links based on cost of output
            * returns cost
        """
        self.input_layer = self.fill_input(pixels)
        # next_layer = activate(layer, links, biases)
        pass

    def read_image(self, image_path):
        return Image.open(image_path).load()  

    def map_letter_to_array(self, letter):
        return [ 1 if ord(letter) == ord('A')+i else 0 for i in xrange(0, self.output_size)]

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
        for letter, images in dataset.iteritems():
            for img in images:
                new_cost = self.backprop(self.read_image(img), self.map_letter_to_array(letter))
                self.costs.append(new_cost)
        return

    def test(self, input):
        """
            * input: a png file (or pixels, not decided yet)
            * output: returns a character between (A, J) with a probability
        """
        pass

def get_dataset(path):
    filelists = defaultdict(list)
    for label in range(0,10):
        dirname = os.path.join(path, chr(ord('A') + label))
        for file in os.listdir(dirname):
            if (file.endswith('.png')):
                fullname = os.path.join(dirname, file)
                # if len(filelists[chr(ord('A') + label)]) < 2:
                if (os.path.getsize(fullname) > 0):
                    filelists[chr(ord('A') + label)].append(fullname)
                else:
                    print('file ' + fullname + ' is empty')
    return filelists

#   labelsAndFiles = []
#   for label in range(0,10):
#     filelist = random.sample(filelists[label], number)
#     for filename in filelist:
#       labelsAndFiles.append((label, filename))

def main():
    net = Network(784, 16, 10)
    dataset = get_dataset('./notMNIST_small')
    net.train(dataset)
    # net.test('')

if __name__ == '__main__':
    main()
