import  numpy as np #numpy:快速操作结构数组的工具
import  pandas as pd #pandas:数据分析处理工具
import matplotlib.pyplot as plt #matplotlib:画图工具
from sklearn import datasets    #datasets:sklearn的示例数据集

#数据集 0-setosa, 1-versicolor. 2-virginica
scikit_iris = datasets.load_iris()

#转换成pandas的DataFrame数据格式，方便观察数据
iris = pd.DataFrame(
    data = np.c_[scikit_iris['data'], scikit_iris['target']], columns=np.append(scikit_iris.feature_names, ['y'])
)
print(iris.head(2))