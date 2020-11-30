# !usr/bin/env python
# -*- coding:utf-8 -*-
import os
import sys
import glob
import shutil
from tqdm import tqdm  # 显示进度条
import re
import pandas as pd
import numpy as np
from scipy import stats
from scipy.stats import pearsonr
from multiprocessing import Pool
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, MaxAbsScaler
from sklearn.feature_selection import VarianceThreshold, SelectFromModel
from sklearn.preprocessing import Normalizer
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor, ExtraTreesRegressor
from sklearn import svm
from sklearn.model_selection import cross_val_score, StratifiedKFold, KFold
from hyperopt import fmin, hp, tpe
from sklearn.metrics import roc_auc_score, mean_absolute_error, mean_squared_error, mean_squared_log_error
from xgboost import XGBClassifier
from multiprocessing import Pool
import math


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


def xgb_model(train_df, test_df, *, hyper_opt=True):
    #  划分XY
    train_x = train_df.iloc[:, 1:-1]
    train_y = train_df.iloc[:, -1].astype('float')
    #  划分XY
    test_x = test_df.iloc[:, 1:-1]
    test_y = test_df.iloc[:, -1].astype('float')
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
    clf = ExtraTreesRegressor(n_estimators=30, n_jobs=28)  # 基于树的特征选择
    clf.fit(train_x_c, train_y)
    selecter = SelectFromModel(clf, prefit=True)
    train_x_d = selecter.transform(train_x_c)
    test_x_d = selecter.transform(test_x_c)
    # 定义用于训练/测试的数据
    train_x = train_x_d
    test_x = test_x_d
    # 判断是否进行寻优
    best = None
    if hyper_opt:
        # 超参数寻优
        def model(hyper_parameter):  # 待寻优函数
            clf = svm.SVR(**hyper_parameter)
            e = cross_val_score(clf, train_x, train_y, cv=KFold(n_splits=10, shuffle=True),
                                scoring='neg_mean_absolute_error', n_jobs=10).mean()
            return -e

        hyper_parameter = {'C': hp.uniform('C', 0.1, 50),
                           'gamma': hp.uniform('gamma', 0.001, 1)}  # 选择要优化的超参数
        best = fmin(model, hyper_parameter, algo=tpe.suggest, max_evals=100,
                    rstate=np.random.RandomState(42))  # 寻找model()函数的最小值，计算次数为100次
        # 训练
        clf = svm.SVR(C=best['C'], gamma=best['gamma'])  # 定义分类器
    else:  # 不进行超参数寻优
        clf = svm.SVR(random_state=42)  # 定义分类器
    clf.fit(train_x, train_y)  # 训练
    # 测试
    pred = clf.predict(test_x)
    # 计算精度
    cross_score = cross_val_score(clf, train_x, train_y, cv=KFold(n_splits=10, shuffle=True),
                                  scoring='neg_mean_absolute_error', n_jobs=28).mean()  # 计算训练集上十折交叉验证的分数
    mae = mean_absolute_error(y_pred=pred, y_true=test_y)
    mse = mean_squared_error(y_pred=pred, y_true=test_y)
    rmse = math.pow(mse, 0.5)
    mse_log = mean_squared_log_error(y_pred=pred, y_true=test_y)
    rp = pearsonr(pred, test_y)


    # 返回数据
    return [cross_score, mae, mse, rmse, mse_log, rp, best, pred]


if __name__ == '__main__':
    path = '/home/xujun/Project_1/revision/general/FREE'
    for repe in range(3):
        result_path = '/home/xujun/Project_1/revision/general/Result/FREE'
        train_csv = '{}/0.csv'.format(path)
        test_csv = '{}/1.csv'.format(path)
        score_csv = '{}/svr_score.csv'.format(result_path)
        train_df = pd.read_csv(train_csv, encoding='utf-8').dropna()
        test_df = pd.read_csv(test_csv, encoding='utf-8').dropna()
        # 建模
        cross_score, mae, mse, rmse, mse_log, rp, param, pred = xgb_model(
            train_df, test_df,
            hyper_opt=True)  # 进行建模， 超参数寻优100次
        pd.DataFrame([cross_score, mae, mse, rmse, mse_log, rp, param, pred]).T.to_csv(
            score_csv, index=False, mode='a', header=False)  # 输出结果到csv中
        con_ = '{}\n'.format(repe)
        with open('/home/xujun/Project_1/revision/general/jobs/FREE/progress.txt', 'a') as f:
            f.write(con_)
