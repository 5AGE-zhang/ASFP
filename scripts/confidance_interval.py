# !usr/bin/env python
# -*- coding:utf-8 -*-

import os
import sys
import pickle
import pandas as pd
import numpy as np
from scipy import stats
from tqdm import tqdm
from sklearn.utils import resample
from sklearn.metrics import accuracy_score, f1_score, matthews_corrcoef, cohen_kappa_score, confusion_matrix, \
    roc_auc_score
from multiprocessing import Pool


def save_file(file, data):
    with open(file, 'wb') as f:
        pickle.dump(data, f)

def read_data(file):
    with open(file, 'rb') as f:
        data = pickle.load(f)
    return data

def confidence_interval(array_data, confidence=0.95):
    if array_data[0] == array_data.mean():
        return (array_data[0], array_data[0]), 0
    else:
        interval = stats.t.interval(confidence, len(array_data) - 1, loc=np.mean(array_data),
                                    scale=stats.sem(array_data))  # 95%置信水平的区间
        error_ = (interval[1] - interval[0]) / 2
        return interval, error_


def bootstrap_replicate_1d(data, data_index):
    bs_index = np.random.choice(data_index, len(data))
    bs_sample = data[bs_index]
    return bs_sample


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


def my_metric(data, data_index):
    # sample
    bs_sample = bootstrap_replicate_1d(data, data_index)
    #
    bs_sample = bs_sample.T
    # get data
    y_pred = bs_sample[0]
    y_pred_proba = bs_sample[1]
    y_true = bs_sample[2]
    # metric
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred).ravel()  # 计算混淆矩阵
    roc_auc = roc_auc_score(y_true=y_true, y_score=y_pred_proba)
    acc = accuracy_score(y_true, y_pred)  # 计算精度
    f1 = f1_score(y_true, y_pred)  # 计算F1score
    mcc = matthews_corrcoef(y_true, y_pred)  # 计算马修斯系数
    kappa = cohen_kappa_score(y_true, y_pred)  # 计算kappa
    ef0_5 = cal_ef(y_true, y_pred, y_pred_proba, top=0.005)
    ef1 = cal_ef(y_true, y_pred, y_pred_proba, top=0.01)
    ef2 = cal_ef(y_true, y_pred, y_pred_proba, top=0.02)
    ef5 = cal_ef(y_true, y_pred, y_pred_proba, top=0.05)
    return [tn, fp, fn, tp, acc, roc_auc, f1, mcc, kappa, ef0_5, ef1, ef2, ef5]


def bootstrap_(data_list, Round=10000):
    target = data_list[0]
    y_true = data_list[1]
    y_pred = data_list[2]
    y_pred_proba = data_list[3]
    # result
    data = np.array(
        list(zip(y_pred, y_pred_proba, y_true)))
    data_index = np.arange(data.shape[0])
    pool = Pool(24)
    result = pool.starmap(my_metric, zip([data for i in range(Round)], [data_index for i in range(Round)]))
    pool.close()
    pool.join()
    return [target, result]

if __name__ == '__main__':
    path = '/home/xujun/Project_1/revision/target_specific/Result/FREE'
    sf = 'svm'
    # targets
    targets = ['abl1',
               'akt1',
               'akt2',
               'ampc',
               'braf',
               'cdk2',
               'cp3a4',
               'csf1r',
               'cxcr4',
               'egfr',
               'fak1',
               'gcr',
               'hivpr',
               'hivrt',
               'igf1r',
               'jak2',
               'kif11',
               'kit',
               'kpcb',
               'lck',
               'mapk2',
               'met',
               'mk01',
               'mk10',
               'mk14',
               'mp2k1',
               'plk1',
               'rock1',
               'src',
               'tgfr1',
               'vgfr2',
               'wee1']
    dst_file = '{}/{}_bootstrap.data'.format(path, sf)
    result_data = []
    for target in tqdm(targets):
        data_file = '{}/{}_{}_pred.data'.format(path, target, sf)
        if os.path.exists(data_file):
            with open(data_file, 'rb') as f:
                tmp = pickle.load(f)
            result_data.append(bootstrap_(tmp))
    save_file(dst_file, data=result_data)

