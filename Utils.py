'''
Created on 2017/08/2

@author: abe
'''
import numpy
import matplotlib.pyplot as plt

class Utils:
    
    @classmethod
    def read_file(Class, file_name):
        path =  'C:\\workspace\\python1\\NeuralNetworkTest\\src\\aaa\\'
        file_name = path + file_name
        with open(file_name, encoding='utf-8') as file:
            data_list = file.readlines()
            return data_list
        
    @classmethod
    def get_init_data(Class, record, output_nodes):
        all_values = record.split(',')
        inputs = (numpy.asfarray(all_values[1:]) /255.0 * 0.99 ) + 0.01
        targets = numpy.zeros(output_nodes) + 0.01
        targets[int(all_values[0])] = 0.99
        return inputs, targets
    
    
    @classmethod
    def plot_images(Class, data_list):
        all_values = data_list[0].split(',')
        image_arr = numpy.asfarray(all_values[1:]).reshape((28, 28))
        print (image_arr)
        plt.imshow(image_arr, cmap='Greys', interpolation='None')
        plt.show()
        print ("finish")