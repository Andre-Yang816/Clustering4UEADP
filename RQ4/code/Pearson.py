import copy
import math
import os
import pandas as pd
import numpy as np


def calcMean(x,y):
    sum_x = sum(x)
    sum_y = sum(y)
    n = len(x)
    x_mean = float(sum_x+0.0)/n
    y_mean = float(sum_y+0.0)/n
    return x_mean,y_mean

def sckendall(a, b):
    L = len(a)
    count = 0
    for i in range(L - 1):
        for j in range(i + 1, L):
            count = count + np.sign(a[i] - a[j]) * np.sign(b[i] - b[j])
    kendall_tau = count / (L * (L - 1) / 2)

    return kendall_tau

def calcPearson(x,y):
    x_mean,y_mean = calcMean(x,y)
    n = len(x)
    sumTop = 0.0
    sumBottom = 0.0
    x_pow = 0.0
    y_pow = 0.0
    for i in range(n):
        sumTop += (x[i]-x_mean)*(y[i]-y_mean)
    for i in range(n):
        x_pow += math.pow(x[i]-x_mean,2)
    for i in range(n):
        y_pow += math.pow(y[i]-y_mean,2)
    sumBottom = math.sqrt(x_pow*y_pow)
    p = sumTop/sumBottom
    return p

def read_data(dataset_path, metric, function):

    data_path = '{0}/{1}.xlsx'.format(dataset_path, metric)
    raw_datas = pd.read_excel(data_path)
    raw_datas = raw_datas[function].values

    return raw_datas

if __name__ == '__main__':
    metrics = ['Precision', 'Recall', 'F1', 'Precisionx', 'Recallx', 'F1x', 'PofB', 'PMI', 'IFA']
    dataset_path = '../../Result'


    functions =  ['Kmedoids.LOC', 'KmeansPlus.LOC', 'Cure.LOC']
    for func in functions:
        result_list = []
        for metric in metrics:
            datas = read_data(dataset_path, metric, func)
            result_list.append(list(datas))

        pearson_res = []
        for i in result_list:
            tmp = []
            for j in result_list:
                tmp.append(calcPearson(i, j))
                # tmp.append(sckendall(x, y))
            pearson_res.append(tmp)

            # p = calcPearson(datas, datas)
            # print(p)
        print(pearson_res)
        new_metrics = ['Precision', 'Recall', 'F1', 'Precision@20%', 'Recall@20%', 'F1@20%', 'PofB@20%', 'PMI@20%',
                       'IFA']
        df = pd.DataFrame(data=pearson_res, columns=new_metrics)

        if not os.path.exists("../output/"):
            os.makedirs("../output")
        df.to_csv("../output/{0}.csv".format(func), index=False)


