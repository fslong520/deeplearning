#!/usr/bin/env python3
# -*- coding: utf-8 -*-
'利用SGD解决线性回归问题'
from csv import reader
import os


class LinearUnit(object):
    def __init__(self, input_para_num, acti_func):
        self.activator = acti_func

    # 设置训练数据、学习率、训练轮数：
    def train_sgd(self, dataset, rate, n_epoch):
        # 所有向量权重都初始化为0
        self.weights = [0.0 for i in range(len(dataset[0]))]
        i = 0
        while i < n_epoch:
            i += 1
            for input_vec_label in dataset:
                prediction = self.predict(input_vec_label)
                # 更新权重值：
                self._update_weights(input_vec_label, prediction, rate)

    # 根据输入值计算输出的线性单元的预测结果：
    def predict(self, row_vec):
        act_values = self.weights[0]
        for i in range(len(row_vec)-1):
            act_values += self.weights[i+1]*row_vec[i]
        return self.activator(act_values)

    # 更新权重值：

    def _update_weights(self, input_vec_label, prediction, rate):
        delta = input_vec_label[-1]-prediction
        # 更新权值：
        self.weights[0] = self.weights[0]+rate*delta
        for i in range(len(self.weights)-1):
            self.weights[i+1] = self.weights[i+1]+rate*delta*input_vec_label[i]


class Datasets(object):
    def __init__(self, filename):
        self.datasets = []
        self.filename = filename

    # 读取数据：

    def load_csv(self):
        with open(os.path.join(os.path.dirname(os.path.abspath(__file__)), self.filename), 'r', encoding='utf-8') as f:
            csv_reader = reader(f)
            # 第一行为表头，单独读一下，让指针走到真实数据上：
            headings = next(csv_reader)
            print(headings)
            # 开始读取数据：
            for row in csv_reader:
                # 如果这一行为空则跳过：
                if not row:
                    continue
                else:
                    self.datasets.append(row)

    # 将数据转换为浮点数：

    def datasets_str_to_float(self):
        self.load_csv()
        for row in self.datasets:
            for i in range(len(row)):
                row[i] = float(row[i].strip())

    # 找到每一列的极大值和极小值：

    def __dataset_min_max(self):
        self.datasets_str_to_float()
        self.min_max = []
        for i in range(len(self.datasets[0])):
            col_values = [row[i] for row in self.datasets]
            value_min = min(col_values)
            value_max = max(col_values)
            self.min_max.append([value_min, value_max])

    # 将数据集合中的每个（列）属性都化为0~1：

    def normal_dataset(self):
        self.__dataset_min_max()
        for row in self.datasets:
            for i in range(len(row)):                
                row[i] = (row[i]-self.min_max[i][0]) / \
                    (self.min_max[i][1]-self.min_max[i][0])
        return self.datasets


# 定义激活函数：


def func_activator(input_value):
    return input_value


# 构建训练数据：
def get_training_datasets():
    db = Datasets('data/winequality-white.csv')
    return db.normal_dataset()


def train_linear_unit():
    datasets = get_training_datasets()
    l_rate = 0.001
    n_epoch = 1000
    # 创建线性训练单元，输入参数的特征数：
    linear_unit = LinearUnit(len(datasets[0]), func_activator)
    # 训练，迭代100轮，学习率0.01：
    linear_unit.train_sgd(datasets, l_rate, n_epoch)
    # 返回训练好的线性单元:
    return linear_unit


if __name__ == '__main__':
    # 获取并训练数据：
    LU = train_linear_unit()
    # 打印训练获得的权重：
    print('\nweights = ', LU.weights)
    # 测试：
    test_data = [
        [5.7, 0.21, 0.32, 0.9, 0.038, 38, 121, 0.99074, 3.24, 0.46, 10.6, 6],
        [6.5, 0.23, 0.38, 1.3, 0.032, 29, 112, 0.99298, 3.29, 0.54, 9.7, 5],
        [6.2, 0.21, 0.29, 1.6, 0.039, 24, 92, 0.99114, 3.27, 0.5, 11.2, 6],
        [6.6, 0.32, 0.36, 8, 0.047, 57, 168, 0.9949, 3.15, 0.46, 9.6, 5],
        [6.5, 0.24, 0.19, 1.2, 0.041, 30, 111, 0.99254, 2.99, 0.46, 9.4, 6],
        [5.5, 0.29, 0.3, 1.1, 0.022, 20, 110, 0.98869, 3.34, 0.38, 12.8, 7],
        [6, 0.21, 0.38, 0.8, 0.02, 22, 98, 0.98941, 3.26, 0.32, 11.8, 6],
    ]
    for i in range(len(test_data)):
        pred=LU.predict(test_data[i])
        print('\nexpected={0},predicted={1}'.format(test_data[i][-1],pred))
