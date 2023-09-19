"""
    贝叶斯分类器的主要思想：
        已知一个含有标记的数据集，此时来了一个测试样本，我们知道测试样本的特征，需要预测标记，
        若我们能够求出这个样本属于各个类别的概率，那么从中选择概率最大的就可以了，那么就是求P(c|x)，
        先用全概率公式P(c|x)=P(x,c)/P(x)，再用条件概率公式P(c|x)=P(x,c)/P(x)=P(x|c)*P(c)/P(x)，
        对于同一个测试样本P(x)都是相同的，因此分母不是我们需要关心的。P(c)很好求，就是在数据集当中某个类别出现的概率
        最难求的就是P(x|c)，朴素贝叶斯的思想就是假设各个特征之间相互独立，那么P(x|c)就等于P(x1|c)*P(x2|c)...，这样就可以求解了
"""

import numpy as np
from sklearn.datasets import fetch_20newsgroups
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.decomposition import PCA
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import MultinomialNB
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import classification_report
import pandas as pd

def save_classification_report(report,save_path):
    # 将分类报告保存至csv文件
    acc_report_df = pd.DataFrame(report).T
    acc_report_df.iloc[-3,:2]= np.nan
    acc_report_df.iloc[-3,3]= acc_report_df.iloc[-2,3]
    # acc_report_df.iloc[-3,2]= np.nan
    acc_report_df.to_csv(save_path)
    return acc_report_df.round(4)

def bayes():
    # 加载数据集（文本数据集）
    # news = fetch_20newsgroups()
    # feature = news.data
    # target = news.target
    # print("特征：", len(feature))
    # print("目标：", len(target))
    # print("目标值的含义：", news.target_names)
    # # 文本特征抽取
    # tf = TfidfVectorizer()
    # feature = tf.fit_transform(feature)
    # feature = feature.toarray()
    # print(feature.shape)
    # print(feature.dtype)
    # feature = feature.astype(np.uint8)
    # print(feature.dtype)
    # # 特征降维
    # # pca
    # pca = PCA(n_components=100)
    # feature = pca.fit_transform(feature)
    # print(feature.shape)

    # 加载数据
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
    x_train, x_test, y_train, y_test = train_test_split(feature, target, test_size=0.25)
    x_train, x_val, y_train, y_val = train_test_split(x_train, y_train, test_size=0.25)
    print("训练集：", x_train.shape)
    print("验证集：", x_val.shape)
    print("测试集：", x_test.shape)
    # 建立贝叶斯模型
    # alapha是拉普拉斯平滑系数，防止计算的概率是0
    mlt = MultinomialNB(alpha=1.0)
    # 训练
    mlt.fit(x_train, y_train)
    # 验证
    score_val = mlt.score(x_val, y_val)
    print("在验证集上的得分：", score_val)
    # 预测
    score_test = mlt.score(x_test, y_test)
    print("在测试集上的得分：", score_test)
    predict = mlt.predict(x_test)
    print("测试结果：", predict)

     # 打印召回率、F1
    report = classification_report(y_test, predict, target_names=['6','9','12','15','18','21',
                                                                                '24','27','30','33','36','39'],output_dict=True)
    print(report)

    return report
if __name__ == "__main__":
    report = bayes()
    save_classification_report(report, save_path=r'朴素贝叶斯精度评价.csv')

