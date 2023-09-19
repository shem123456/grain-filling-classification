"""
    随机森林是一种同质的集成学习算法，通过构建多个决策树，然后结合多个决策树的结果，得到更好的预测
"""

import pandas as pd
from sklearn.feature_extraction import DictVectorizer
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import GridSearchCV
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import classification_report

def forest():
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
    # # 查看有没有缺失值
    # print(pd.isnull(feature).any())
    # # 填充缺失值
    # Age = feature.pop("Age")  # 取出，意思是取出来之后删除原来的
    # Age = Age.fillna(Age.mean())
    # feature.insert(0, "Age", Age)

    # # 字典特征抽取
    # dv = DictVectorizer()
    # feature = dv.fit_transform(feature.to_dict(orient="records"))
    # feature = feature.toarray()
    # print(feature)
    # print(dv.get_feature_names())

    # 划分数据集
    x_train, x_test, y_train, y_test = train_test_split(feature, target, test_size=0.2)
    print("训练集：", x_train.shape, y_train.shape)
    print("测试集：", x_test.shape, y_test.shape)

    # 建立模型
    rf = RandomForestClassifier()

    # 超参数搜索
    param = {"n_estimators":[10, 20, 30, 40], "max_depth":[5, 15, 25]}
    gc = GridSearchCV(rf, param_grid=param, cv=5)

    # 训练
    gc.fit(x_train, y_train)

    # 交叉验证网格搜索的结果
    print("在测试集上的准确率：", gc.score(x_test, y_test))
    print("在验证集上的准确率：", gc.best_score_)
    print("最好的模型参数：", gc.best_params_)
    print("最好的模型：", gc.best_estimator_)

    predict = gc.predict(x_test)

    print(classification_report(y_test, predict, target_names=['6','9','12','15','18','21',
                                                                                '24','27','30','33','36','39']))


if __name__ == "__main__":
    forest()

