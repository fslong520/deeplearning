#!/usr/bin/env python3
# -*- coding: utf-8 -*-
'利用BP算法解决异或问题'
import random
import time
from math import exp


def random_seed():
    # 重置随机种子：
    random.seed(time.time())
    a = random.random()
    print('随机睡眠{}秒。'.format(a))
    # 重置随机种子:
    random.seed(time.time())

# 初始化神经网络：


class BpNetwork(object):
    def __init__(self, n_inputs, n_hidden, n_outputs):
        self.initialize_network(n_inputs, n_hidden, n_outputs)
        self.n_outputs = n_outputs

    # 初始化神经网络：
    def initialize_network(self, n_inputs, n_hidden, n_outputs):
        random_seed()
        # n_hidden个隐藏层，每个隐藏层有n_inpust+1个权重：
        self.hidden_layer = [{'weights': [random.random() for i in range(n_inputs+1)]}
                             for i in range(n_hidden)]
        random_seed()
        self.output_layer = [{'weights': [random.random() for i in range(
            n_hidden+1)]} for i in range(n_outputs)]
        self.network = [self.hidden_layer, self.output_layer]

    # 计算神经元的激活值：
    def activate(self, weights, inputs):
        activation = weights[-1]
        for i in range(len(weights)-1):
            activation += weights[i]*inputs[i]
        return activation

    # 定义激活函数:
    def transfer(self, activation):
        return 1.0/(1.0+exp(-activation))

    # 计算激活函数的导数：

    def transfer_derivative(self, output):
        return output*(1.0-output)

    # 计算神经网络的正向传播：

    def forward_propagate(self, row):
        inputs = row
        for layer in self.network:
            new_inputs = []
            for neuron in layer:
                activation = self.activate(neuron['weights'], inputs)
                neuron['output'] = self.transfer(activation)
                new_inputs.append(neuron['output'])
            inputs = new_inputs
        return inputs

    # 反向传播误差信息，并将纠偏责任存储在神经元中：
    def backward_propagate_error(self, expected):
        for i in reversed(range(len(self.network))):
            layer = self.network[i]
            errors = []
            if i != len(self.network)-1:
                for j in range(len(layer)):
                    error = 0.0
                    for neuron in self.network[i+1]:
                        error += (neuron['weights'][j] *
                                  neuron['responsbility'])
                    errors.append(error)
            else:
                for j in range(len(layer)):
                    neuron = layer[j]
                    errors.append(expected[j]-neuron['output'])
            for j in range(len(layer)):
                neuron = layer[j]
                neuron['responsbility'] = errors[j] * \
                    self.transfer_derivative(neuron['output'])

    # 根据误差更新网络权重:
    def update_weights(self, row, l_rate):
        for i in range(len(self.network)):
            inputs = row[:-1]
            if i != 0:
                inputs = [neuron['output'] for neuron in self.network[i-1]]
            for neuron in self.network[i]:
                for j in range(len(inputs)):
                    neuron['weights'][j] += l_rate * \
                        neuron['responsbility']*inputs[j]
                neuron['weights'][-1] += l_rate*neuron['responsbility']

    # 根据指定周3期训练网络：
    def train_network(self, train, l_rate, n_epoch):
        for epoch in range(n_epoch):
            sum_error = 0
            for row in train:
                outputs = self.forward_propagate(row)
                expected = [0 for i in range(self.n_outputs)]
                # 这里是重点，偏置或者哑元的值为1：                
                expected[row[-1]] = 1
                sum_error += sum([(expected[i]-outputs[i]) **
                                  2 for i in range(len(expected))])
                self.backward_propagate_error(expected)
                self.update_weights(row, l_rate)
            print('>周期={},误差={:.8}'.format(epoch+1, sum_error))

    def predict(self, row):
        outputs = self.forward_propagate(row)
        return outputs.index(max(outputs))


if __name__ == '__main__':
    dataset = [
        [1, 1, 0],
        [1, 0, 1],
        [0, 1, 1],
        [0, 0, 0]
    ]
    n_inputs = len(dataset[0])-1
    n_outputs = len(set([row[-1] for row in dataset]))
    network = BpNetwork(n_inputs, 2, n_outputs)
    network.train_network(dataset, 0.5, 2000)
    for layer in network.network:
        print(layer)
    for row in dataset:
        prediction = network.predict(row)
        print('预测值={},实际输出值={}'.format(row[-1], prediction))
