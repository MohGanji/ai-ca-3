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

    def __init__(self, input_size, hidden_size, output_size):
        self.input_size= input_size
        self.hidden_size= hidden_size
        self.output_size= output_size 

        self.input_layer= [0*input_size]
        self.hidden_layer= [0*hidden_size]
        self.output_layer= [0*output_size]

        self.input_hidden_links = [ [np.random.uniform(-1, 1) for j in xrange(0, self.hidden_size)] for i in xrange(0, self.input_size)]
        self.hidden_output_links = [ [np.random.uniform(-1, 1) for k in xrange(0, self.output_size)] for j in xrange(0, self.hidden_size)]

        pprint(self.input_hidden_links)
        pprint(self.hidden_output_links)
        
        
def main():
    net = Network(784, 16, 10)

if __name__ == '__main__':
    main()
