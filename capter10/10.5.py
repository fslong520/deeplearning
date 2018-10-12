#!/usr/bin/env python3
# -*- coding: utf-8 -*-
'利用BP算法预测小麦分类'
from random import seed
from random import randrange
from csv import reader
from math import exp
import random
import os


def random_seed():
    # 重置随机种子：
    random.seed(time.time())
    a = 2*random.random()
    print('随机睡眠{}秒。'.format(a))
    # 重置随机种子:
    random.seed(time.time())


class Database(object):
    def __init__(self, db_file):
        self.filename = db_file
        self.dataset = []
        self.path = os.path.dirname(os.path.abspath(__file__))

    # 导入csv文件内的数据：

    def load_csv(self):
        with open(os.path.join(self.path, self.filename), 'r', encoding='utf-8') as f:
            csv_reader = reader(f)
            for row in csv_reader:
                # 去掉空行：
                if not row:
                    continue
                self.dataset.append(row)

    # 将n-1列的属性字符串列转换为浮点数，第n列为分类的类别：
    def dataset_str_to_float(self):
        col_len = len(self.dataset[0])-1
        for row in self.dataset:
            for column in range(col_len):
                row[column] = float(row[column].strip())

    # 将最后一列（n）的类别转换为整数，并提取有多少个类：
    def str_column_to_int(self):
        column = len(self.dataset[0])-1
        # 读取分类数据：
        class_values = [row[column] for row in self.dataset]
        # 用列表来去重：
        unique = set(class_values)
        lookup = {}
        for i, value in enumerate(unique):
            lookup[value] = i
        for row in self.dataset:
            row[column] = lookup[row[column]]

    # 找出每一列的最大值和最小值：
    def dataset_min_max(self):
        self.min_max = []
        self.min_max = [[min(column), max(column)]
                        for column in zip(*self.dataset)]  # zip可以将多个列表合成一个列表，此处是把一个二维列表行变成一列，之后直接使用min和max函数

    # 将数据集合中的每个列属性都归一化：
    def normalize_dataset(self):
        self.dataset_min_max()
        for row in self.dataset:
            for i in range(len(row)-1):
                row[i] = (row[i]-self.min_max[i][0]) / \
                    (self.min_max[i][1]-self.min_max[i][0])

    # 构建训练数据：
    def get_dataset(self):
        self.load_csv()
        self.dataset_str_to_float()
        self.str_column_to_int()
        self.normalize_dataset()
        return self.dataset


class BP_Network(object):
    def __init__(self, n_inputs, n_hidden, n_outputs):
        self.n_inputs = n_inputs
        self.n_hidden = n_hidden
        self.n_outputs = n_outputs
        random_seed()
        hidden_layer = [{'weights': [random.random()for i in range(self.n_inputs+1)]}
                        for i in range(self.n_hidden)]
        output_layer = [{'weights': [random.random()for i in range(self.n_hidden+1)]}
                        for i in range(self.n_outputs)]
        self.network = [hidden_layer, output_layer]

    # 计算神经元的激活值(加权之和)：
    def activate(self, weights, inputs):
        activation = weights[-1]
        for i in range(len(weights)-1):
            activation += weights[i]*inputs[i]
        return activation

    # 定义激活函数：
    def transfer(self, activation):
        return 1.0/(1.0+exp(-activation))

    # 计算激活函数的导数：
    def transfer_derivative(self, output):
        return output*(1.0-output)

    # 计算神经网络的正向传播:
    def forward_propagate(self, row):
        inputs = row
        for layer in self.network:
            new_inputs = []
            for neuron in layer:
                activation = self.activate(neuron['weights'], inputs)
                neuron['output'] = self.transfer(activation)
                new_inputs.append(neuron['output'])
            # 将激活函数的输出值作为下一层的输入值：
            inputs = new_inputs
        return inputs

    # 反向传播误差信息，并将纠偏责任存储在神经元当中：
    def backward_propagate_error(self, expected):
        for i in reversed(range(len(self.network))):
            layer = self.network[i]
            errors = []
            if i != len(self.network)-1:
                for j in range(len(layer)):
                    error = 0.0
                    for neuron in self.network[i+1]:
                        error += (neuron['weights'][j]*neuron['responsbility'])
                    errors.append(error)
            else:
                for j in range(len(layer)):
                    neuron = layer[j]
                    errors.append(expected[j]-neuron['output'])
            for j in range(len(layer)):
                neuron = layer[j]
                neuron['responsbility'] = errors[j] * \
                    self.transfer_derivative(neuron['output'])

    # 根据误差，更新权重：
    def _update_weights(self, row):
        for i in range(len(self.network)):
            inputs = row[:-1]
            if i != 0:
                inputs = [neuron['output'] for neuron in self.network[:-1]]
            for neuron in self.network[i]:
                for j in range(len(inputs)):
                    neuron['weights'][j] += self.l_rate * \
                        neuron['responsbility']*inputs[j]
