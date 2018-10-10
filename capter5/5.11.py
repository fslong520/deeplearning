#!/usr/bin/env python3
# -*- coding: utf-8 -*-
'sklearn来进行K-近邻，还是山鸢尾'
import numpy as np
import pandas as pd
from sklearn import datasets

from sklearn.neighbors import KNeighborsClassifier

# 加载IRIS数据集合：
scikit_iris = datasets.load_iris()
scikit_iris_str = str(scikit_iris)
with open('capter5/data/5.11.data', 'w+',encoding='utf-8') as f:
    f.write(scikit_iris_str)
# 将数据转换为pandas的DataFrame格式，便于观察：
pd_iris = pd.DataFrame(data=np.c_[scikit_iris['data'], scikit_iris['target']],
                       columns=np.append(scikit_iris['feature_names'], ['Y']))
print(pd_iris.head(3))
# 选择全部特征参与训练模型:
X = pd_iris[scikit_iris['feature_names']]
Y = pd_iris['Y']
# (1)选择模型：
knn = KNeighborsClassifier(n_neighbors=1)
# (2)拟合模型(训练模型))：
knn.fit(X, Y)
# (3)预测新数据:
print(knn.predict([[4, 3, 5, 3], [4, 6, 4, 7]]))
