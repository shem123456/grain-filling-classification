"""
    支持向量机：通过寻找划分超平面来进行分类的算法，这个划分超平面只由支持向量有关，与其他样本无关
"""

import numpy as np
import pandas as pd
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from  sklearn.neural_network import MLPClassifier
from sklearn.metrics import classification_report
from sklearn.preprocessing import MinMaxScaler

def save_classification_report(report,save_path):
    # 将分类报告保存至csv文件
    acc_report_df = pd.DataFrame(report).T
    acc_report_df.iloc[-3,:2]= np.nan
    acc_report_df.iloc[-3,3]= acc_report_df.iloc[-2,3]
    # acc_report_df.iloc[-3,2]= np.nan
    acc_report_df.to_csv(save_path)
    return acc_report_df.round(4)

def MLP():
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

    # 建立模型
    MLP = MLPClassifier()
    # 训练
    MLP.fit(x_train, y_train)
    # 验证
    score_val = MLP.score(x_val, y_val)
    print("在验证集上的得分：", score_val)
    # 测试
    score_test = MLP.score(x_test, y_test)
    print("在测试集上的得分：", score_test)
    # 预测
    predict = MLP.predict(x_test)
    print("预测结果：", predict)

    # 打印召回率、F1
    report = classification_report(y_test, predict, target_names=['6','9','12','15','18','21',
                                                                                '24','27','30','33','36','39'],output_dict=True)
    print(report)
    return report

if __name__ == "__main__":
    report = MLP()
    save_classification_report(report, save_path=r'MLP评价.csv')


