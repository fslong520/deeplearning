#!/usr/bin/env python3
# -*- coding: utf-8 -*-
' 深度学习1 '
__author__ = 'fslong'
__cover__ = '张玉宏'
from matplotlib import pyplot as plt
from math import sqrt


class Class1(object):
    def __init__(self):
        self.dataset = [[1.2, 1.1], [2.4, 3.5],
                        [4.1, 3.2], [3.4, 2.8], [5, 5.4]]
        self.x = [row[0] for row in self.dataset]
        self.y = [row[1] for row in self.dataset]

    def drawPic(self,predict):

        # 设置横纵坐标轴：
        plt.axis([0, 6, 0, 6])
        # 画出散点图:
        x,y=[row[0] for row in self.dataset],[row[1] for row in self.dataset]
        plt.plot(x,y,'bs')
        x,y=[row[0] for row in predict],[row[1] for row in predict]
        plt.plot(x,y,'r^-')
        plt.grid()
        plt.show()
        plt.savefig('class1.png')

    # 计算均值:
    def mean(self, values):
        return sum(values)/float(len(values))

    # 计算方差：
    def variance(self, values, mean):
        return sum([(x-mean)**2 for x in values])

    def convariance(self, x, mean_x, y, mean_y):
        convar=0.0
        for i in range(len(x)):
            convar += (x[i]-mean_x)*(y[i]-mean_y)

        return convar

    # 求回归系数：

    def coefficients(self):
        mean_x, mean_y=self.mean(self.x), self.mean(self.y)
        var_x, var_y=self.variance(
            self.x, mean_x), self.variance(self.y, mean_y)
        convar=self.convariance(self.x, mean_x, self.y, mean_y)
        print('x 统计特性：均值 = %.3f 方差=%.3f' % (mean_x, var_x))
        print('y 统计特性：均值 = %.3f 方差=%.3f' % (mean_y, var_y))
        print('协方差 = :%.3f' % convar)
        # 计算回归系统：
        w1=convar/var_x
        w0=mean_y-w1*mean_x
        print('线性回归系数分别是： w0 = %.3f, w1 = %.3f' % (w0, w1))
        return w0, w1

    # 构建简单线性回归：
    def simple_linear_regression(self, test):
        predict=list()
        w0, w1=self.coefficients()
        for row in test:
            y_model=w1*row[0]+w0
            predict.append((row[0],y_model))
        return predict


    # 计算均方根误差RMSE：
    def rmse_metric(self, actual, predicted):
        sum_error=0.0
        for i in range(len(actual)):
            prediction_error=predicted[i][-1]-actual[i]
            sum_error += (prediction_error)**2
        mean_error=sum_error/float(len(actual))
        return sqrt(mean_error)


    # 评估算法及协调:
    def evaluate_algorithm(self):
        test_set=list()
        # 构建新的列表用于存储预测值：
        for row in self.dataset:
            row_copy=list(row)
            row_copy[-1]=None
            test_set.append(row_copy)            
        predicted=self.simple_linear_regression(test_set)
        for val in predicted:
            print('%.3f\t' % val[-1])
        actual=[row[-1] for row in self.dataset]
        rmse=self.rmse_metric(actual, predicted)
        print(predicted)
        self.drawPic(predicted)
        return rmse
    def main(self):
        print('%.3f'%self.evaluate_algorithm())

if __name__ == '__main__':
    class1=Class1()
    class1.main()
