#!/usr/bin/env python3
# -*- coding: utf-8 -*-
'山鸢尾K-近邻手动预测'

import csv
import math
import operator
import os
import random

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


class Iris(object):
    # 类的声明以及加载训练要用到的数据
    def __init__(self, filename='data/5.10.iris.data', split=0.3, k=3):
        self.base_path = os.path.dirname(os.path.abspath(__file__))
        self.training_set = []
        self.test_set = []
        self.load_dataset(filename, split)
        self.draw_pic(filename)
        self.predictions = []
        self.k = k
    #　从csv文件中加载数据到类当中：

    def load_dataset(self, filename, split):
        path = os.path.join(self.base_path, filename)
        with open(path, 'r', encoding='utf-8') as csv_file:
            lines = csv.reader(csv_file)
            self.dataset = list(lines)
            for x in range(len(self.dataset)):
                for y in range(4):
                    # 将每组数据的前４个字符串转换为浮点数：
                    self.dataset[x][y] = float(self.dataset[x][y])
                if random.random() < split:
                    self.training_set.append(self.dataset[x])
                else:
                    self.test_set.append(self.dataset[x])

    # 画图：
    def draw_pic(self, filename):
        df = pd.read_csv(os.path.join(self.base_path, filename), header=None)
        Ｘ = df.iloc[0:150, [0, 2]].values  # 表示获取的是0位和2位的数值，也就是每组数据的第1、3个数据
        # 用数据绘图，用的是每组数据的第一个数当x坐标，第二个数当y坐标
        plt.scatter(X[0:50, 0], X[0:50, 1], color='blue',
                    marker='x', label='setosa')
        plt.scatter(X[50:100, 0], X[50:100, 1], color='red',
                    marker='o', label='versicolor')
        plt.scatter(X[100:150, 0], X[100:150, 1], color='green',
                    marker='*', label='virginica')
        plt.xlabel('sepal lenth')
        plt.ylabel('petal width')
        plt.legend(loc='upper left')
        plt.grid()
        fig = plt.gcf()
        plt.show()
        fig.savefig(os.path.join(self.base_path, 'img/5.10.png'))

    # 计算欧式距离，其实就是计算出每个坐标差的平方：
    def euclidist(self, instance1, instance2, length):
        distance = 0
        for x in range(length):
            distance += pow((instance1[x]-instance2[x]), 2)
        return math.sqrt(distance)

    # 寻找邻居，k指的是邻居个数，邻居越多说明范围越宽，精度就越低，但邻居数量少了会受到噪点的影响:
    def get_neighbors(self, testInstance, k):
        distances = []
        length = len(testInstance)-1
        for x in range(len(self.training_set)):
            dist = self.euclidist(testInstance, self.training_set[x], length)
            distances.append((self.training_set[x], dist))
        distances.sort(key=lambda distances: distances[1])
        neighbors = []
        for x in range(k):
            neighbors.append(distances[x][0])
        return neighbors
    # 寻找分类:

    def get_class(self, neighbors):
        class_votes = {}
        for x in range(len(neighbors)):
            instance_class = neighbors[x][-1]
            if instance_class in class_votes:
                class_votes[instance_class] += 1
            else:
                class_votes[instance_class] = 1
            # 逆序排序：
            sorted_votes = sorted(class_votes.items(),
                                  key=operator.itemgetter(1), reverse=True)
            return sorted_votes[0][0]
    # 检查学习率：

    def get_accuracy(self, predictions):
        correct = 0
        for x in range(len(self.test_set)):
            if self.test_set[x][-1] == predictions[x]:
                correct += 1
        return(correct/float(len(self.test_set)))*100

    # 主程序:
    def main(self):
        print('训练集合：'+repr(len(self.training_set))+'个')
        print('测试集合：'+repr(len(self.test_set))+'个')
        for x in range(len(self.test_set)):
            neighbors = self.get_neighbors(self.test_set[x], self.k)
            result = self.get_class(neighbors)
            self.predictions.append(result)
            print('>预测='+repr(result)+'，实际='+repr(self.test_set[x][-1]))
        accuracy = self.get_accuracy(self.predictions)
        print('精确度为：%.3f%%' % accuracy)


if __name__ == '__main__':
    iris = Iris(filename='data/5.10.iris.data', split=0.7, k=3)
    iris.main()
