#!/usr/bin/env python3
# -*- coding: utf-8 -*-
'使用感知机来实现and和or'


class Perceptron(object):
    def __init__(self, input_para_num, acti_func):
        self.activator = acti_func
        # 权重向量初始化为0
        self.weights = [0.0 for _ in range(input_para_num)]

    def __str__(self):
        return '\nfinal weights \n\tw0={:.2f}\n\tw1={:.2f}\n\tw2={:.2f} '.format(self.weights[0], self.weights[1], self.weights[2])

    def predict(self, row_vec):
        act_value = 0.0
        for i in range(len(self.weights)):
            act_value += self.weights[i]*row_vec[i]
        return self.activator(act_value)
    # 更新权重的方法：

    def _update_weights(self, input_vec_label, prediction, rate):
        delta = input_vec_label[-1]-prediction
        for i in range(len(self.weights)):
            self.weights[i] += rate*delta*input_vec_label[i]

    def train(self, dataset, iteration, rate):
        i = 0
        while i < iteration:
            i += 1
            for input_vec_label in dataset:
                # 计算机感知当前权重下的输出：
                prediction = self.predict(input_vec_label)
                # 更新权重：
                self._update_weights(input_vec_label, prediction, rate)


def func_act(input_value):
    return 1.0 if input_value >= 0.0 else 0.0


def get_training_dataset(func=0):
    # 构建训练函数所用的数据：
    # and方法：
    dataset_and = [[-1, 1, 1, 1], [-1, 0, 0, 0], [-1, 1, 0, 0], [-1, 0, 1, 0]]
    # or方法:
    dataset_or = [[-1, 1, 1, 1], [-1, 0, 0, 0], [-1, 1, 0, 1], [-1, 0, 1, 1]]
    # func==0表示and，其余都是or：
    if func == 0:
        return dataset_and
    else:
        return dataset_or


def train_preceptron(func):
    p = Perceptron(3, func_act)
    # 获取训练数据：
    dataset = get_training_dataset(func)
    # 迭代10轮，学习率0.1
    p.train(dataset, 10, 0.1)
    # 返回训练好的感知机
    return p


if __name__ == '__main__':
    # 训练and感知机：
    perceptron = train_preceptron(0)
    # 打印训练获得的权重：
    print(perceptron)
    # 测试：
    print('1 and 1 = {}'.format(perceptron.predict([-1, 1, 1])))
    print('0 and 0 = {}'.format(perceptron.predict([-1, 0, 0])))
    print('1 and 0 = {}'.format(perceptron.predict([-1, 1, 0])))
    print('0 and 1 = {}'.format(perceptron.predict([-1, 0, 1])))

    # 训练or感知机：
    perceptron = train_preceptron(1)
    # 打印训练获得的权重：
    print(perceptron)
    # 测试：
    print('1 or 1 = {}'.format(perceptron.predict([-1, 1, 1])))
    print('0 or 0 = {}'.format(perceptron.predict([-1, 0, 0])))
    print('1 or 0 = {}'.format(perceptron.predict([-1, 1, 0])))
    print('0 or 1 = {}'.format(perceptron.predict([-1, 0, 1])))
