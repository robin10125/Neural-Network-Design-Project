import random
import numpy
import scipy
from scipy.special import expit


class Neural_Network:


    def __init__(self,neuronNumberList):


        self.neuronList = []
        self.neuronActivationList = []
        self.neuronWeightsList = []
        self.biasList = []
        self.numLayers = len(neuronNumberList) - 1
        self.learningRate = 0.1

        numNeurons = [0]
        count = 0
        for layer in neuronNumberList:
            numNeurons.append(layer)
            self.neuronWeightsList.append([])

            self.biasList.append([])
            self.neuronActivationList.append([])
            for neuron in range(layer):
                self.neuronList.append(str(neuron))
                self.neuronActivationList[count].append(0)
                self.neuronWeightsList[count].append([])
                for prevLayerNum in range(numNeurons[count]):
                    self.neuronWeightsList[count][neuron].append(random.randrange(-100,100) / 100)
                if layer > 0:
                    self.biasList[count].append(0)

            count += 1


    def feedforward(self, inputList):


        for inputNeuronsNum in range(len(self.neuronActivationList[0])):
            self.neuronActivationList[0][inputNeuronsNum] = inputList[inputNeuronsNum]

        for eachLayerNum in range(1, len(self.neuronActivationList)):

            for eachNeuronNum in range(len(self.neuronActivationList[eachLayerNum])):
                zValue = 0
                for eachWeightNum in range(len(self.neuronWeightsList[eachLayerNum][eachNeuronNum])):
                    zValue += self.neuronWeightsList[eachLayerNum][eachNeuronNum][eachWeightNum] * \
                              self.neuronActivationList[eachLayerNum - 1][eachWeightNum]
                zValue += self.biasList[eachLayerNum][eachNeuronNum]
                zValue = scipy.asanyarray(zValue)
                self.neuronActivationList[eachLayerNum][eachNeuronNum] = expit(zValue)


    def backpropogation(self, labels):


        _costDict = {}
        _neuronCount = 0

#        #find error of output layer
        _costDict[self.numLayers] = []
        for neuron in self.neuronActivationList[self.numLayers]:
            _costDict[self.numLayers].append((neuron - labels[_neuronCount])
                                             * (expit(neuron) * (1 - expit(neuron))))

            _neuronCount += 1

#        #backpropogate cost
        for layerNum in range(self.numLayers - 1, 0, -1):
            _costDict[layerNum] = []
            for neuronNum in range(len(self.neuronActivationList[layerNum])):
                weightAndErrorProduct = 0

                for eachValue in range(len(self.neuronActivationList[layerNum + 1])):
                    weightAndErrorProduct += self.neuronWeightsList[layerNum + 1][eachValue][neuronNum] * _costDict[layerNum + 1][eachValue]

                z = self.neuronActivationList[layerNum][neuronNum]
                _costDict[layerNum].append(weightAndErrorProduct * (expit(z) * (1 - expit(z))))

#       #adjust weights and biases
        for layerNumber in range(1, len(self.neuronWeightsList)):

            for adjustNeuronNum in range(len(self.neuronWeightsList[layerNumber])):
                self.biasList[layerNumber][adjustNeuronNum] -= self.learningRate * _costDict[layerNumber][adjustNeuronNum]
                for indWeightNum in range(len(self.neuronWeightsList[layerNumber][adjustNeuronNum])):
                    activation = self.neuronActivationList[layerNumber - 1][indWeightNum]
                    self.neuronWeightsList[layerNumber][adjustNeuronNum][indWeightNum] -= self.learningRate * _costDict[layerNumber][adjustNeuronNum] * activation




    def getOutputs(self):


        return self.neuronActivationList[self.numLayers]


