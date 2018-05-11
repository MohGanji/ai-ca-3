from __future__ import print_function
import numpy as np
from pprint import pprint

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

        pprint(self.input_hidden_links)
        pprint(self.hidden_output_links)
        
    
    def backprop(pixels, expected_output):
        """
            * pixels: 2 dim array of pixels of an image, each field in range(0, 255)
            * expected_output:  one dim array with length of output layer, values: (0, 1)

            * updates weight of links based on cost of output
        """
        pass

    def train(dataset, mode):
        """
            * mode: sgd | gd
                gd: gradiant descent
                sgd: stochastic gradiant descent
            * dataset: an object of arrays of names of png files:
                {
                    'a': [1.png, 2.png]
                }

            * returns: nothings
                runs backprop on data based on mode.
        """
        pass

    def test(input):
        """
            * input: a png file (or pixels, not decided yet)
            * output: returns a character between (A, J) with a probability
        """
        pass

def main():
    net = Network(784, 16, 10)

if __name__ == '__main__':
    main()
