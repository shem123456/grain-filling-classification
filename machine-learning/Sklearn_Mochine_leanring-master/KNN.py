"""
    KNN算法也叫做K近邻算法，它的主要思想是：
        计算测试样本与训练集中各个样本之间的距离，选择与测试样本距离最近的K个，然后统计这K个样本中出现标记最多的那个，
        将这个标记作为测试样本的标记
"""

from sklearn.datasets import load_iris
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import MinMaxScaler

def knn():
   # 加载数据集
    titan = pd.read_excel("F:/sly-kernal-1/machine-learning/2018-2019.xlsx")

    # 构造特征值和目标值
    feature = titan[["R", "G", "B", "H","S", "V",
                     "area", "length","radius","equi_diameter", "eccentric",
                     "compact", "rectangle_degree","roundness","correlation", "homogeneity",
                     "energy", "ASM","entropy"
                     ]]
    target = titan["days"]

    # 特征预处理，归一化处理
    scaler = MinMaxScaler()
    feature = scaler.fit_transform(feature)
    # 划分数据集
    x_train, x_test, y_train, y_test = train_test_split(feature, target, test_size=0.2)
    x_train, x_val, y_train, y_val = train_test_split(x_train, y_train, test_size=0.1)
    print("训练集：", x_train.shape, y_train.shape)
    print("验证集：", x_val.shape, y_val.shape)
    print("测试集：", x_test.shape, y_test.shape)
    # 建立KNN模型
    kn = KNeighborsClassifier(n_neighbors=5)
    # 训练
    kn.fit(x_train, y_train)
    # 验证
    score_val = kn.score(x_val, y_val)
    print("在验证集上的得分：", score_val)
    # 测试
    score_test = kn.score(x_test, y_test)
    print("在测试集上的得分：", score_test)
    predict = kn.predict(x_test)
    print("在测试集上的预测结果：", predict)


if __name__ == "__main__":
    knn()
