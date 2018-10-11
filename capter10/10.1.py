#!/usr/bin/env python3
# -*- coding: utf-8 -*-
'初始化神经元网络'
from random import seed
from random import random
# 初始化网络:
def initialize_network(n_inputs,n_hidden,n_outputs):
    network=[]
    hidden_layer=[{'weights':[random() for i in range(n_inputs+1)]} for i in range(n_hidden)]
    network.append(hidden_layer)
    outputs_layer=[{'weights':[random() for i in range(n_inputs+1)]} for i in range(n_outputs)]
    network.append(outputs_layer)
    return network
if __name__=='__main__':
    seed(1)
    network=initialize_network(2,3,4)
    for layer in network:
        print(layer)