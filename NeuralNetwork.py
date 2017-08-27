'''
Created on 2017/08/28

@author: abe
'''
import numpy

class NeuralNetwork:
    
    #コンストラクタ
    def __init__(self, inputnodes, hiddennodes, outputnodes, learningrate):
        self.inodes = inputnodes
        self.hnodes = hiddennodes
        self.onodes = outputnodes
        
        self.wih = numpy.random.normal(0.0, pow(self.hnodes, -0.5), (self.hnodes, self.inodes))
        self.who = numpy.random.normal(0.0, pow(self.hnodes, -0.5), (self.onodes, self.hnodes))
    
        self.learn = learningrate
        
        self.activateion_function = lambda x : 1.0 / (1.0 + numpy.exp(-x))

        pass
    
    #ニューラルネットワークの学習
    def train(self, inputs_list, targets_list):
        inputs = numpy.array(inputs_list, ndmin = 2).T
        targets = numpy.array(targets_list, ndmin = 2).T
        
        #print (self.wih)
        #print (inputs)
        
        hidden_inputs = numpy.dot(self.wih, inputs)
        hidden_outputs = self.activateion_function(hidden_inputs)
        
        final_inputs = numpy.dot(self.who, hidden_outputs)
        final_outputs = self.activateion_function(final_inputs)
        
        output_errors = targets - final_outputs
        
        
        self.who += self.learn * numpy.dot((output_errors * final_outputs * (1.0 - final_outputs)), numpy.transpose(hidden_outputs))
        
        hidden_erros = numpy.dot(self.who.T, output_errors)
        
        self.wih += self.learn * numpy.dot((hidden_erros * hidden_outputs * (1.0 - hidden_outputs)), numpy.transpose(inputs))
        
        pass
    