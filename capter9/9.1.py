#!/usr/bin/env python3
# -*- coding: utf-8 -*-
'基于梯度下降的线性回归'

import os
from matplotlib import pyplot
BREAD_PRICE = [[0.5, 5], [0.6, 5.5], [0.8, 6], [1.1, 6.8], [1.4, 7]]


def draw():
    x = [bread[0] for bread in BREAD_PRICE]
    y = [bread[1] for bread in BREAD_PRICE]
    pyplot.axis([0, 2, 4, 8])
    pyplot.plot(x, y, 'b*')
    learning_rate = 0.001  # 学习率
    num_iter = 1000  # 迭代次数
    w0, w1 = gradient_descent_runner(
        BREAD_PRICE, 1, 1, learning_rate, num_iter)
    y = [predict(w0, w1, weight) for weight in x]
    print(predict(w0,w1,0.9))
    pyplot.plot(x, y, 'r^-')
    pyplot.title('The relationship between weight and price of bread')
    pyplot.xlabel('weight')
    pyplot.ylabel('price')
    pyplot.grid()
    fig = pyplot.gcf()
    pyplot.show()
    fig.savefig(os.path.join(os.path.dirname(
        os.path.abspath(__file__)), 'img/9.1.png'))


def bgd_step_gradient(w0_current, w1_current, points, learning_rate):
    w0_gradient = 0
    w1_gradient = 0
    for i in range(len(points)):
        x = points[i][0]
        y = points[i][1]
        w0_gradient += -1.0*(y-((w1_current*x)+w0_current))
        w1_gradient += -1.0*x*(y-((w1_current*x)+w0_current))
    new_w0 = w0_current-(learning_rate*w0_gradient)
    new_w1 = w1_current-(learning_rate*w1_gradient)
    return [new_w0, new_w1]


def gradient_descent_runner(points, start_w0, start_w1, l_rate, num_iterations):
    w0 = start_w0
    w1 = start_w1
    i = 0
    while i < num_iterations:
        i += 1
        w0, w1 = bgd_step_gradient(w0, w1, points, l_rate)
    return w0, w1


def predict(w0, w1, weight):
    price = w1*weight+w0
    return price


if __name__ == '__main__':
    draw()
