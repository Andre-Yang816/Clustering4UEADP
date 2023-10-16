# -*- coding: utf-8 -*-
import os
import pandas as pd
import numpy as np
from scipy import stats
from pathlib import Path
import warnings
warnings.filterwarnings("ignore")

def read_data(dataset_path, metric):
    functions = ['Kmedoids', 'Kmeans++', 'Cure', 'ManualUp', 'CBS+', 'RF', 'GB', 'EALR', 'EATT', 'SERS']
    datas = {}
    for function in functions:
        data_path = '{0}/{1}.xlsx'.format(dataset_path,metric)
        raw_datas = pd.read_excel(data_path)

        raw_datas = raw_datas[function].values

        datas[function] = raw_datas

    return datas

def process_data(datas,baseline):
    metric_datas = []
    functions = []

    baseline_data = datas[baseline]
    metric_datas.append(baseline_data)

    for key, value in datas.items():
        if key == baseline:
            continue
        metric_datas.append(value)
        functions.append(key)

    return metric_datas, functions


def wilcoxon(l1, l2):
    w, p_value = stats.wilcoxon(l1, l2, correction=False)
    return p_value

def wdl(l1, l2):
    win = 0
    draw = 0
    loss = 0
    for i in range(len(l1)):
        if l1[i] < l2[i]:
            loss = loss+1
        if l1[i] == l2[i]:
            draw = draw+1
        if l1[i] > l2[i]:
            win = win+1

    return win, draw, loss

def average_improvement(l1, l2):
    avgl1 = round(np.average(l1), 3)
    avgl2 = round(np.average(l2), 3)
    #imp = round((avgl1-avgl2)/avgl2, 4)
    imp = round((avgl1-avgl2), 3)
    return imp

def Wilcoxon_signed_rank_test(metric_datas, functions, metric, b):
    pvalues = []
    sortpvalues = []
    bhpvalues = []
    print('***********{0}***********'.format(metric))
    #improve_ave_dataset = []
    for i in range(1, len(metric_datas)):
        #这里需要改一下
        pvalue = wilcoxon(metric_datas[0], metric_datas[i])
        pvalues.append(pvalue)
        sortpvalues.append(pvalue)
        #improve_ave_dataset.append(average_improvement(metric_datas[i], metric_datas[0]))
        if metric == 'Recall' or metric == 'PofB':
            print('-------------{0}-----------------'.format(functions[i - 1]))
            #print("compute p-value between %s and CBSplus: %s" % (functions[i-1], pvalue))
            #print("compute W/D/L between %s and CBSplus: %s" % (functions[i-1], wdl(metric_datas[i], metric_datas[0])))
            print("compute average improvement between {0} and CBSplus: {1}" .format(functions[i-1],
                                                                             average_improvement(metric_datas[i], metric_datas[0])))

    sortpvalues.sort()

    for i in range(len(pvalues)):
        bhpvalue = pvalues[i]*(len(pvalues))/(sortpvalues.index(pvalues[i])+1)
        bhpvalues.append(bhpvalue)
        print("compute Benjamini—Hochberg p-value between %s and CBSplus: %s" % (functions[i-1], bhpvalue))

    Path('../output/p_{0}/'.format(metric)).mkdir(parents=True, exist_ok=True)
    output_path = '../output/p_{0}/{1}.csv'.format(metric,baseline[b])

    output = pd.DataFrame(data=[pvalues], columns=functions)
    #output = pd.DataFrame(data=[pvalues], columns=functions)
    output.to_csv(output_path, encoding='utf-8')


if __name__ == '__main__':
    #metrics = ['Precision', 'Recall' ,'F1', 'Precisionx', 'Recallx', 'F1x', 'PofB', 'PMI', 'Popt', 'IFA']
    metrics = ['IFA', 'PofB', 'PMI', 'Recallx']
    dataset_path = '../Data'
    for metric in metrics:
        print("Doing Wilcoxon signed rank test in %s ..." % ( metric))
        datas = read_data(dataset_path, metric)
        baseline = ['Kmedoids', 'Kmeans++', 'Cure']
        for b in range(len(baseline)):
            metric_datas, functions = process_data(datas, baseline[b])
            Wilcoxon_signed_rank_test(metric_datas, functions, metric, b)