#!/usr/bin/env python3
# -*- coding: utf-8 -*-
'BP算法的向前传播'

from math import exp
# 计算神经元的激活值（加权之和)


def activate(weights, inputs):
    activation = weights[-1]
    for i in range(len(weights)-1):
        activation += weights[i]*inputs[i]
    return activation


# 神经元传递函数：
def transfer(activation):
    return 1.0/(1.0+exp(-activation))


# 计算神经网络的正向传播：
def forward_propagate(network, row):
    inputs = row
    for layer in network:
        new_inputs = []
        for neuron in layer:
            activation = activate(neuron['weights'], inputs)
            neuron['outputs'] = transfer(activation)
            new_inputs.append(neuron['outputs'])
            print(neuron['outputs'])
        inputs = new_inputs
    return inputs


if __name__ == '__main__':
    # 测试正向传播：
    network = [
        [{'weights': [0.13436424411240122, 0.8474337369372, 0.763774618976614]},
         {'weights': [0.2550690257394217, 0.49543508709194095, 0.4494910647887381]}],
        [{'weights': [0.651592972722763, 0.7887233511355132, 0.0938595867742349]},
         {'weights': [0.02834747652200631, 0.8357651039198697, 0.43276706790505337]}]]
    row=[1,0,None]
    output=forward_propagate(network,row)
    print(output)