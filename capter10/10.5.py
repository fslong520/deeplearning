#!/usr/bin/env python3
# -*- coding: utf-8 -*-
'利用BP算法预测小麦分类'
from random import seed
from random import randrange
from csv import reader
from math import exp
import random
import os
import time


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
        self.n_epoch = 2000
        self.l_rate = 0.1
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
                inputs = [neuron['output'] for neuron in self.network[i-1]]
            for neuron in self.network[i]:
                for j in range(len(inputs)):
                    neuron['weights'][j] += self.l_rate * \
                        neuron['responsbility']*inputs[j]
                    neuron['weights'][-1] += self.l_rate * \
                        neuron['responsbility']

    # 根据指定的周期迅雷网络:
    def train_network(self, train):
        for epoch in range(self.n_epoch):
            sum_error = 0
            for row in train:
                outputs = self.forward_propagate(row)
                expected = [0 for i in range(self.n_outputs)]
                # 比如分类是3，那么就把索引为3的输出设置为1，其他的就是0：
                expected[row[-1]] == 1
                sum_error += sum([(expected[i]-outputs[i]) **
                                  2 for i in range(len(expected))])
                self.backward_propagate_error(expected)
                self._update_weights(row)
            print('=>周期={}，误差={:.3}'.format(epoch+1, sum_error*100))

    # 利用训练好的网络，预测新的数据:

    def predict(self, row):
        outputs = self.forward_propagate(row)
        # 因为只有一个输出的值我们设置成了1，而这个值得索引恰好就是分类：
        return outputs.index(max(outputs))

    # 利用随机梯度递减策略训练网络：

    def back_propagation(self, train, test):
        self.train_network(train)
        predictions = []
        for row in test:
            prediction = self.predict(row)
            predictions.append(prediction)
        return predictions

    # 将数据等分成k份：

    def cross_validation_splite(self, n_folds):
        dataset_split = []
        dataset_copy = list(self.dataset)
        fold_size = int(len(self.dataset)/n_folds)
        for i in range(n_folds):
            print('第{}组'.format(i+1))
            fold = []
            while len(fold) < fold_size:
                random_seed()
                index = random.randrange(len(dataset_copy))
                fold.append(dataset_copy.pop(index))
            dataset_split.append(fold)
        return dataset_split

    # 用预测正确百分比来衡量正确率:

    def accuracy_metric(self, actual, predictions):
        correct = 0
        for i in range(len(actual)):
            #print('实际是{}类，预测是{}类'.format(actual[i], predictions[i]))
            if actual[i] == predictions[i]:
                correct += 1
        return 100*correct/float(len(actual))

    # 用每一个交叉分割得块（训练集合，测试集合）来评估BP算法：

    def evaluate_algorithm(self, dataset, n_folds, l_rate, n_epoch):
        self.l_rate = l_rate
        self.n_epoch = n_epoch
        self.dataset = dataset
        folds = self.cross_validation_splite(n_folds)
        scores = []
        for fold in folds:
            train_set = list(folds)
            train_set.remove(fold)
            train_set = sum(train_set, [])
            test_set = []
            actual = [row[-1] for row in fold]
            for row in fold:
                row_copy = list(row)
                test_set.append(row_copy)
                row_copy[-1] = None
            predictions = self.back_propagation(train_set, test_set)            
            accuracy = self.accuracy_metric(actual, predictions)
            scores.append(accuracy)
        return scores


if __name__ == '__main__':
    filename = 'data/seeds_dataset.csv'
    db = Database(filename)
    dataset = db.get_dataset()

    # 设置网络初始化参数：
    n_inputs = len(dataset[0])-1
    n_hidden = 6
    n_outputs = len(set([row[-1] for row in dataset]))
    BP = BP_Network(n_inputs, n_hidden, n_outputs)
    l_rate = 0.1
    n_folds = 5
    n_epoch = 500
    scores = BP.evaluate_algorithm(dataset, n_folds, l_rate, n_epoch)
    print('评估算法正交验证得分:{}'.format(scores))
    print('平均准确率：{:.3}%%'.format(sum(scores)/float(len(scores))))
