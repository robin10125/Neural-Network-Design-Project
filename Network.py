from NeuralNetwork import Neural_Network
import random
import math


def functCalc(input):


    if input[1] > math.sin(5 * input[0]):
        return [1]
    else:
        return [0]


weights = [2, 20, 20, 1]
network = Neural_Network(weights)

for i in range(10000):
    value = []
    for j in range(2):
        value.append((random.randrange(-100,100)) / 100)
    label = functCalc(value)
    network.feedforward(value)
    network.backpropogation(label)

for k in range(100):
    value2 = []
    for l in range(2):
        value2.append(random.randrange(-100,100) / 100)
    network.feedforward(value2)
    print(network.getOutputs(), functCalc(value2))

