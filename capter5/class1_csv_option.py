#!/usr/bin/env python3
# -*- coding: utf-8 -*-
' 深度学习1 '

from random import seed
from random import randrange
from csv import reader
from math import sqrt
import os
from matplotlib import pyplot as plt


class Student(object):
    def __init__(self):
        self.csv = 'class1.csv'
        self.path = os.path.dirname(os.path.abspath(__file__))
        self.dataset = []

    # 导入csv文件：
    def load_csv(self):
        filename = os.path.join(self.path, self.csv)
        with open(filename, 'r', encoding='utf-8') as f:
            csv_reader = reader(f)
            # 读取表头：
            print('文件加载\033[1;32m成功\033[0m，开始读取表头...')
            self.headings = next(csv_reader)
            print('\033[1;31m表头为：\033[0m', self.headings)
            print('开始读取正文数据...')
            # 从第二行开始才是真正的数据：
            for row in csv_reader:
                if not row:  # 如果这一行数据为空则跳过
                    continue
                else:
                    self.dataset.append(row)
            print('数据读取\033[1;32m成功\033[0m，开始解析...')

    # 将读取到的数据从字符串转换为浮点数，这里直接对源数据进行了操作：
    def str_column_to_float(self, column):
        for row in self.dataset:
            row[column] = float(row[column].strip())
    # 将数据分割成训练集合和测试集合两部分：

    def train_test_split(self, percent):
        train = []
        train_size = percent*len(self.dataset)
        dataset_copy = list(self.dataset)
        while len(train) < train_size:
            index = randrange(len(dataset_copy))
            # 在train里添加数据，在dataset_copy里删除同一个数据：
            train.append(dataset_copy.pop(index))
        return train, dataset_copy

    # 计算均值：
    def mean(self, values):
        return sum(values)/float(len(values))

    # 计算x与y协方差的函数
    def covariance(self, x, mena_x, y, mean_y):
        covar = 0.0
        for i in range(len(x)):
            covar += (x[i]-mena_x)*(y[i]-mean_y)
        return covar
    # 计算方差的函数:

    def variance(self, values, mean):
        return sum([(x-mean)**2 for x in values])

    # 计算回归系数的函数：
    def coefficients(self, train):
        x = [row[0] for row in train]
        y = [row[1] for row in train]
        x_mean, y_mean = self.mean(x), self.mean(y)
        w1 = self.covariance(x, x_mean, y, y_mean)/self.variance(x, x_mean)
        w0 = y_mean-w1*x_mean
        return w0, w1
    # 构建简单线性回归:

    def simple_linear_regression(self, train, test_set):
        predictions = []

        w0, w1 = self.coefficients(train)
        for row in test_set:
            y_model = w1*row[0]+w0
            predictions.append((row[0], y_model))
        # 将所有点y值计算一次并将坐标都存入列表中：
        self.predictions = [(row[0], w1*row[0]+w0) for row in self.dataset]
        return predictions
    # 计算均方根误差RMSE：

    def rmse_metric(self, actual, predictions):
        sum_error = 0.0
        for i in range(len(actual)):
            prediction_error = predictions[i][1]-actual[i]
            sum_error += prediction_error**2
        mean_error = sum_error/float(len(actual))
        return sqrt(mean_error)
    # 使用分割开的训练集合和测试集合运行评估算法：

    def evaluate_algorithm(self, algorithm, split_percent, *args):
        train, test = self.train_test_split(split_percent)
        test_set = []
        for row in test:
            row_copy = list(row)
            row_copy[-1] = None
            test_set.append(row_copy)
        predicted = algorithm(train, test_set, *args)
        actual = [row[-1] for row in test]
        rmse = self.rmse_metric(actual, predicted)
        print('线性回归计算成功,','RMSE: %.3f' % rmse)
        print('开始绘图...')
        self.drawPic(self.predictions)
        return rmse

    def drawPic(self, predictions):

        # 设置横纵坐标轴：
        plt.axis([0, 150, 0, 450])
        # 画出散点图:
        x, y = [row[0] for row in self.dataset], [row[1]
                                                  for row in self.dataset]
        plt.plot(x, y, 'bs')
        x, y = [row[0] for row in predictions], [row[1] for row in predictions]
        plt.plot(x, y, 'r^-')
        plt.grid()
        fig = plt.gcf()
        plt.show()
        fig.savefig(os.path.join(self.path,'class1.png'))
        print('图片已保存在当前程序目录下，名称为”class1.png“')

    def main(self):
        # 设置随机种子，为随机挑选训练和测试山上集做准备:
        seed(2)
        # 导入数据：
        self.load_csv()
        for col in range(len(self.dataset[0])):
            self.str_column_to_float(col)
        # 设置数据集合分割百分比：
        percent = 0.6
        self.evaluate_algorithm(self.simple_linear_regression, percent)
       


if __name__ == '__main__':
    student = Student()
    student.main()
