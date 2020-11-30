# !usr/bin/env python
# -*- coding:utf-8 -*-
import os
import sys
import glob
import shutil
from tqdm import tqdm  # 显示进度条
import re
import pickle
import pandas as pd
import numpy as np
from scipy import stats
from multiprocessing import Pool
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, MaxAbsScaler
from sklearn.feature_selection import VarianceThreshold, SelectFromModel
from sklearn.preprocessing import Normalizer
from sklearn.ensemble import RandomForestClassifier, ExtraTreesClassifier
from sklearn import svm
from sklearn.model_selection import cross_val_score, StratifiedKFold
from hyperopt import fmin, hp, tpe
from sklearn.metrics import accuracy_score, f1_score, matthews_corrcoef, cohen_kappa_score, confusion_matrix, \
    roc_auc_score
from xgboost import XGBClassifier
from multiprocessing import Pool


def confidence_interval(array_data, confidence=0.95):
    interval = stats.t.interval(confidence, len(array_data) - 1, loc=np.mean(array_data),
                                scale=np.std(array_data))  # 95%置信水平的区间
    return interval


def cal_ef(test_y, pred, pred_prob, *, top=0.01):
    # 合并df
    df = pd.DataFrame(test_y)  # 把真实值转成Dataframe
    df['pred'] = pred
    df['pred_prob'] = pred_prob
    # 根据打分排序，并且给缺失值填充0
    df.sort_values(by='pred_prob', inplace=True, ascending=False)
    # 去除同分异构分子，只保留第一项
    # data.drop_duplicates(subset='title', keep='first', inplace=True)
    # 总分子数
    N_total = len(df)
    # 总活性分子数
    N_active = len(df[df.iloc[:, 0] == 1])
    # 前b%总分子数
    topb_total = int(N_total * top + 0.5)
    # 前b%总分子数据
    topb_data = df.iloc[:topb_total, :]
    # 前b%总活性分子数
    topb_active = len(topb_data[topb_data.iloc[:, 1] == 1])
    # 富集因子
    ef_b = (topb_active / topb_total) / (N_active / N_total)
    return ef_b


def xgb_model(df, *, hyper_opt=True):
    #  划分XY
    x = df.iloc[:, 1:-1]
    y = df.iloc[:, -1].astype('int')
    # 划分数据集
    train_x, test_x, train_y, test_y = train_test_split(x, y, train_size=0.75, stratify=y,
                                                        shuffle=True)  # 随机按照4：1划分训练集和测试集
    # 数据预处理
    scaler = MaxAbsScaler().fit(train_x)  # 数据缩放,均值为0, 方差为1
    train_x_a = scaler.transform(train_x)
    test_x_a = scaler.transform(test_x)
    threshold = VarianceThreshold().fit(train_x_a)  # 低方差过滤
    train_x_b = threshold.transform(train_x_a)
    test_x_b = threshold.transform(test_x_a)
    normalizer = Normalizer(norm='l2').fit(train_x_b)  # 数据归一化, 使数据适合分类,处理后计算速度快了很多: 18min -> 24s
    train_x_c = normalizer.transform(train_x_b)
    test_x_c = normalizer.transform(test_x_b)
    clf = ExtraTreesClassifier(n_estimators=30, n_jobs=28)  # 基于树的特征选择
    clf.fit(train_x_c, train_y)
    selecter = SelectFromModel(clf, prefit=True)
    train_x_d = selecter.transform(train_x_c)
    test_x_d = selecter.transform(test_x_c)
    # 定义用于训练/测试的数据
    train_x = train_x_d
    test_x = test_x_d
    # 判断是否进行寻优
    if hyper_opt:
        # 超参数寻优
        def model(hyper_parameter):  # 待寻优函数
            clf = RandomForestClassifier(**hyper_parameter, oob_score=False, n_jobs=24, random_state=42)
            e = cross_val_score(clf, train_x, train_y, cv=StratifiedKFold(n_splits=10, shuffle=True, random_state=42),
                                scoring='f1', n_jobs=10).mean()
            return -e

        feature_list = ['sqrt', 'log2', None]  # 特征列表
        weight_list = ['balanced', 'balanced_subsample']  # 权重列表
        hyper_parameter = {'n_estimators': hp.uniformint('n_estimators', 80, 150),  # 决策树的数量
                           'max_depth': hp.uniformint('max_depth', 3, 50),  # 树的深度 过多会过拟合
                           'max_features': hp.choice('max_features', feature_list),  # 单个决策树使用的特征
                           'class_weight': hp.choice('class_weight', weight_list)  # 正负集权重
                           }  # 选择要优化的超参数
        best = fmin(model, hyper_parameter, algo=tpe.suggest, max_evals=100,
                    rstate=np.random.RandomState(42))  # 寻找model()函数的最小值，计算次数为100次
        # 训练
        clf = RandomForestClassifier(n_estimators=int(best['n_estimators']), max_depth=int(best['max_depth']),
                                     max_features=feature_list[best['max_features']],
                                     class_weight=weight_list[best['class_weight']], oob_score=False, n_jobs=24,
                                     random_state=42)  # 定义分类器
    else:  # 不进行超参数寻优
        clf = RandomForestClassifier(n_jobs=24, random_state=42)  # 定义分类器
    clf.fit(train_x, train_y)  # 训练
    # 测试
    pred = clf.predict(test_x).tolist()
    pred_pro = clf.predict_proba(test_x)  # 计算0，1的概率
    pred_pro = np.array([i[1] for i in pred_pro])  # 选取小分子为抑制剂的概率
    #
    pred = np.array(pred)
    test_y = np.array([i for i in test_y.values])
    with open(pred_data, 'wb') as f:
        pickle.dump([target, test_y, pred, pred_pro], f)



if __name__ == '__main__':
    path_glob = '/home/xujun/Project_1/revision/target_specific/FREE'
    targets = os.listdir(path_glob)
    for target in targets:
        try:
            # 读取chemdiv数据
            path = r'{}/{}'.format(path_glob, target)
            result_path = '/home/xujun/Project_1/revision/target_specific/Result/FREE'
            data_csv = '{}/{}.csv'.format(path, target)
            pred_data = '{}/{}_rf_pred.data'.format(result_path, target)
            df = pd.read_csv(data_csv, encoding='utf-8').dropna()
            # 建模
            xgb_model(df,
                      hyper_opt=True)  # 进行建模， 超参数寻优100次
            con_ = '{}\n'.format(target)
        except:
            con_ = '{}_error\n'.format(target)
        with open('/home/xujun/Project_1/revision/target_specific/jobs/FREE/rf/progress.txt', 'a') as f:
            f.write(con_)
