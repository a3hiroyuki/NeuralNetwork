'''
Created on 2017/08/28

@author: hiroy
'''

from Utils import Utils
from NeuralNetwork import  NeuralNetwork

INPUT_NODES = 784
HIDDEN_NODES = 100
OUTPUT_NODES = 10
LEARNING_RATE = 0.3

#メイン関数
if __name__ == '__main__':
    file_name = 'mnist_train_100.csv'
    data_list = Utils.read_file(file_name)
    #Utils.plot_images(data_list)
    
    network = NeuralNetwork(INPUT_NODES, HIDDEN_NODES, OUTPUT_NODES, LEARNING_RATE)
    
    for i, record in enumerate(data_list):
        print (str(i) + ' \n')
        inputs, targets = Utils.get_init_data(record, OUTPUT_NODES)
        network.train(inputs, targets)