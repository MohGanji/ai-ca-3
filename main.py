from __future__ import print_function
import numpy as np
from collections import defaultdict
from pprint import pprint
from PIL import Image
import glob, sys, os, random, math

class Network:
    LEARN_RATE = 1
    input_size= 0
    hidden_size= 0
    output_size= 0

    input_layer= []
    hidden_layer= []
    output_layer= []

    input_hidden_links = []
    hidden_output_links = []
    hidden_biases = []
    output_biases = []
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
        self.hidden_biases = [np.random.uniform(-1, 1) for j in xrange(0, self.hidden_size)]
        self.output_biases = [np.random.uniform(-1, 1) for k in xrange(0, self.output_size)]
        
    def fill_input(self, pixels):
        width, height, img = pixels
        res = []
        for i in range(width):
            for j in range(height):
                res.append(img[(i, j)]/255.0)
        return res

    def activation_func(self, val, mode="sigmoid"):
        return 1.0/(1.0 + math.exp(-1*val))

    def calc_new_node(self, layer, links, biases, j):
        res = biases[j]
        for i, node in enumerate(layer):
            res += node*links[i][j]
        return res

    def activate(self, layer, links, biases, new_layer_size):
        new_layer = []
        for i in range(new_layer_size):
            z = self.calc_new_node(layer, links, biases, i)
            new_layer.append(self.activation_func(z))
        return new_layer    

    def calc_cost(self, output, expected):
        res = 0
        for i, j in zip(output, expected):
            res += 0.5*((j-i)**2)
        return res

    # s*(1-s)

    def update_links(self, old_links, old_biases, first_layer, second_layer, expected):
        new_links = [[0*len(old_links[i])] for i in range(old_links)]
        for k, second_val in enumerate(second_layer):
            for j, first_val in enumerate(first_layer):
                z = self.calc_new_node(second_layer, old_links, old_biases, j)
                new_links[k][j] = old_links[k][j] - self.LEARN_RATE * z * (1.0 - z) * 2.0 * (first_val - second_val)


    def backprop(self, pixels, expected_output):
        """
            * pixels: 2 dim array of pixels of an image, each field in range(0, 255)
            * expected_output:  one dim array with length of output layer, values: (0, 1)

            * updates weight of links based on cost of output
            * returns cost
        """
        # forward propagate
        self.input_layer = self.fill_input(pixels)
        self.hidden_layer = self.activate(self.input_layer, self.input_hidden_links, self.hidden_biases, self.hidden_size)
        self.output_layer = self.activate(self.hidden_layer, self.hidden_output_links, self.output_biases, self.output_size)
        the_cost = self.calc_cost(self.output_layer, expected_output)
        # end of forward propagate

        # error backpropagation
        #self.hidden_output_links, self.output_biases = self.update_links(self.hidden_output_links, self.output_biases, self.output_layer, self.hidden_layer, expected_output)
        #pprint(hidden_output_links)
        #self.input_hidden_links, self.hidden_biases = self.update_links(self.input_hidden_links, self.hidden_biases, self.output_layer, self.input_layer, expected_output)
        # does not work

        return the_cost

    def read_image(self, image_path):
        img = Image.open(image_path)
        width, height = img.size
        return (width, height, img.load())

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

    def predict(self, input):
        """
            * input: a png file (or pixels, not decided yet)
            * output: returns a character between (A, J) with a probability
        """
        pass

def get_dataset(path):
    filelists = defaultdict(list)
    for label in range(0,1):
        dirname = os.path.join(path, chr(ord('A') + label))
        for file in os.listdir(dirname):
            if (file.endswith('.png')):
                fullname = os.path.join(dirname, file)
                if len(filelists[chr(ord('A') + label)]) < 2:
                    if os.path.getsize(fullname) > 0:
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

if __name__ == '__main__':
    main()
