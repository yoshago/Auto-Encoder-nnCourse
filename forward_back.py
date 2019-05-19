import numpy as np
from math import exp
from random import random, seed
import random
import prepare_data
import matplotlib.pyplot as plt
# Initialize a network

INPUT_SIZE = 256
OUTPUT_SIZE = 256
HIDEN_SIZE = 16


def initialize_network(n_inputs, n_hidden, n_outputs):
    network = list()
    hidden_layer = [{'weights': [random.random() for i in range(n_inputs + 1)]} for i in range(n_hidden)]
    network.append(hidden_layer)
    output_layer = [{'weights': [random.random() for i in range(n_hidden + 1)]} for i in range(n_outputs)]
    network.append(output_layer)
    return network


# Calculate neuron activation for an input
def activate(weights, inputs):
    activation = weights[-1]
    for i in range(len(weights) - 1):
        activation = activation + weights[i] * inputs[i]
    return activation


# Transfer neuron activation
def transfer(activation):
    return 1.0 / (1.0 + exp(-activation/10))


# Forward propagate input to a network output
def forward_propagate(network, row):
    inputs = row
    for layer in network:
        new_inputs = []
        for neuron in layer:
            activation = activate(neuron['weights'], inputs)
            neuron['output'] = transfer(activation)
            new_inputs.append(neuron['output'])
        inputs = new_inputs
    return inputs


# Calculate the derivative of an neuron output
def transfer_derivative(output):
    return output * (1.0 - output)


# Backpropagate error and store in neurons
def backward_propagate_error(network, expected):
    for i in reversed(range(len(network))):
        layer = network[i]
        errors = list()
        if i != len(network) - 1:
            for j in range(len(layer)):
                error = 0.0
                for neuron in network[i + 1]:
                    error += (neuron['weights'][j] * neuron['delta'])
                errors.append(error)
        else:
            for j in range(len(layer)):
                neuron = layer[j]
                errors.append(expected[j] - neuron['output'])
                #print(str(expected[j])+ " " +str(neuron['output']) + " " + str(expected[j] - neuron['output']))
        for j in range(len(layer)):
            neuron = layer[j]
            x=transfer_derivative(neuron['output'])
            neuron['delta'] = errors[j] * x


# Update network weights with error
def update_weights(network, row, l_rate):
    for i in range(len(network)):
        inputs = row[:-1]
        if i != 0:
            inputs = [neuron['output'] for neuron in network[i - 1]]
        for neuron in network[i]:
            for j in range(len(inputs)):
                neuron['weights'][j] += l_rate * neuron['delta'] * inputs[j]
            neuron['weights'][-1] += l_rate * neuron['delta']




# Train a network for a fixed number of epochs
def train_network(network, train, l_rate, n_epoch, n_outputs):
    error_list = []
    for epoch in range(n_epoch):
        sum_error = 0
        for row in train:
            outputs = forward_propagate(network, row)
            #expected = row
            expected = row
            sum_error += sum([(expected[i] - outputs[i]) ** 2 for i in range(len(expected))])
            backward_propagate_error(network, expected)
            update_weights(network, row, l_rate)
        error_list.append(sum_error)
        print('>epoch=%d, lrate=%.3f, error=%.3f' % (epoch, l_rate, sum_error))
    return error_list

def predict_img(name):
    dataset = prepare_data.prepare_data("Lena.jpg")

# Make a prediction with a network
def predict(network, row):
    outputs = forward_propagate(network, row)
    return outputs

# Test network
print("inital network")
network = initialize_network(INPUT_SIZE, HIDEN_SIZE, OUTPUT_SIZE)
print("done initial network")

# Test training backprop algorithm
seed(1)
print("start to prepare data set")
dataset = prepare_data.prepare_data("Lena.jpg",8)
print(len(dataset))
random.shuffle(dataset)
print("done prepare data set")
print("start train network")
num_epoch=1
#train + error graph
error_list=train_network(network, np.array(dataset)/256.0, 0.1, num_epoch, OUTPUT_SIZE)
x=np.array([x for x in range(num_epoch)]).reshape(num_epoch,1)
plt.plot(x, error_list, label='error function')
plt.title('loss function, '+str(HIDEN_SIZE)+ ' neuron in hidden layer')
plt.xlabel('epochs')
plt.ylabel('error(*256)')
plt.legend()
plt.savefig('morning_n'+str(HIDEN_SIZE)+'_e'+str(num_epoch)+'sec.png')


###################################predict###########################################################
with open('text.txt', 'r') as file:
    data = file.read().replace('\n', '')
test_data=np.array(prepare_data.text_to_data(data[:262144]))/256.0
ret_data=[]
for row in test_data:
    tmp_output=(np.array(predict(network,row))*256)+0.5
    ret_data.append(tmp_output)
test_error=0
for i in range(len(test_data)):
    for j in range(len(test_data[1])):
        test_error+=abs(ret_data[i][j]-256.0*test_data[i][j])
print("baby test error=" +str(test_error))
ret_data=prepare_data.data_to_img(ret_data)

#######################################################################################################################################

test_data=np.array(prepare_data.img_to_data("Lena.jpg"))/256.0
ret_data=[]
for row in test_data:
    tmp_output=(np.array(predict(network,row))*256)+0.5
    ret_data.append(tmp_output)
test_error=0
for i in range(len(test_data)):
    for j in range(len(test_data[1])):
        test_error+=abs(ret_data[i][j]-(256.0*test_data[i][j]))
print("Lena test error=" +str(test_error))
ret_data=prepare_data.data_to_img(ret_data)




