#!/usr/bin/env python3
# -*- coding: utf-8 -*-
'sklearn来进行模型评估，还是山鸢尾'
import numpy as np
import pandas as pd
from sklearn import datasets, metrics
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier

# 加载山鸢尾数据集合：
scikit_iris = datasets.load_iris()
pd_iris = pd.DataFrame(
    data=np.c_[scikit_iris['data'], scikit_iris['target']],
    columns=np.append(scikit_iris['feature_names'], 'y')
)
x = pd_iris[scikit_iris['feature_names']]
y = pd_iris['y']
x_train, x_test, y_train, y_test = train_test_split(
    x, y, test_size=0.3, random_state=0)
knn=KNeighborsClassifier(n_neighbors=10)
knn.fit(x_train,y_train)
# 由于训练集合和测试集合并没有本质区别，所以让模型分别测试：
y_predict_on_train=knn.predict(x_train)
y_predict_on_test=knn.predict(x_test)
print('准确率为：{:.2%}'.format(metrics.accuracy_score(y_train,y_predict_on_train)))
print('准确率为：{:.2%}'.format(metrics.accuracy_score(y_test,y_predict_on_test)))