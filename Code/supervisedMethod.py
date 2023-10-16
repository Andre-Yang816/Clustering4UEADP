# coding=utf-8
import math

import numpy as np
import pandas as pd
from sklearn.model_selection import GridSearchCV
from sklearn.naive_bayes import GaussianNB
from sklearn.linear_model import LogisticRegression, LinearRegression
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
import warnings
import os
from deepforest import CascadeForestClassifier

from Code.Measure import Performance, Popt, evaluate_classify
from Code.PerformanceMeasure import PerformanceMeasure
from Code.Processing import Processing
from Code.functions import KmeansCluster, AgglomerativeCluster, BirchCluster, MeanShiftCluster, KmedoidsCluster, \
    MiniBatchKMeansCluster, AffinityPropagationCluster, BsasCluster, CureCluster, DbscanCluster, MbsasCluster, \
    OpticsCluster, RockCluster, Somsc, SyncsomCluster, BangCluster, Kmeans_plusplusCluster, ClaransCluster, \
    EmaCluster, FcmCluster, GmeansCluster, TtsasCluster, Xmeans, ManualUp

warnings.filterwarnings('ignore')

'''
预测缺陷密度的情况下
'''
header = ["Precision", "Recall", "F1", "Precisionx", "Recallx", "F1x", "IFA", "Pofb", "PMI"]

from sklearn.model_selection import cross_val_score
from sklearn.metrics import precision_recall_fscore_support, roc_auc_score
from sklearn.naive_bayes import GaussianNB
import numpy as np

def feature_evaluation(X, y, classifier):
    scores = cross_val_score(classifier, X, y, cv=10)
    precision, recall, _, _ = precision_recall_fscore_support(y, classifier.predict(X), average='binary')
    accuracy = np.mean(scores)
    roc_auc = roc_auc_score(y, classifier.predict_proba(X)[:, 1])
    return precision, recall, accuracy, 2 * (precision * recall) / (precision + recall), roc_auc

def wrapper_method(X, y):
    F = list(range(X.shape[1]))
    selF = F.copy()
    all_feature_results = []

    min_features = 1  # 设置一个最小特征数
    max_iterations = 10  # 设置最大迭代次数

    for _ in range(max_iterations):
        results = []

        for feature in selF:
            classifier = GaussianNB()  # Using Naive Bayes for Wrapper method

            # Create a subset of features
            subset_X = X[:, [feature]]

            # Fit the classifier
            classifier.fit(subset_X, y)

            # Evaluate performance
            precision, recall, accuracy, f_measure, roc_auc = feature_evaluation(subset_X, y, classifier)
            results.append((f_measure, feature))

        # Record the results for all features in this iteration
        all_feature_results.extend(results)

        # Determine the best feature
        best_result = max(results, key=lambda x: x[0])
        if len(selF) < min_features:
            break  # Break the loop if there are too few features

        best_feature = best_result[1]

        # Recompute feature scores using Naive Bayes (if needed)

        # Identify removeF, the 50% of features of selF with the lowest feature evaluation
        remove_count = int(len(selF) * 0.5)
        removeF = [x[1] for x in sorted(results, key=lambda x: x[0])[:remove_count]]

        # Update selF
        selF = list(set(selF) - set(removeF))

    # Select the top half of features based on the overall results
    top_half_features = [x[1] for x in sorted(all_feature_results, key=lambda x: x[0], reverse=True)[:len(F)//2]]

    return top_half_features



def ClassificationBySERS(X, Y, testX, testingcodeN):
    selected_features = wrapper_method(X, Y)
    selected_features = sorted(selected_features)
    print("Selected Features:", selected_features)
    selected_X = np.array(X)[:, selected_features]
    selected_testX = np.array(testX)[:, selected_features]
    NB = GaussianNB()
    NB_pred = NB.fit(selected_X, Y).predict_proba(selected_testX)
    NB_pred = [p[1] for p in NB_pred]  # 概率当做缺陷个数
    NB_pred3 = []
    for j in range(len(NB_pred)):
        if testingcodeN[j] != 0:
            if NB_pred[j] > 0.5:
                NB_pred3.append((NB_pred[j] / testingcodeN[j]) + 100000)
            else:
                NB_pred3.append((NB_pred[j] / testingcodeN[j]) - 100000)
        else:
            NB_pred3.append(-100000000)
    return NB_pred3


def optimizeParameter(classifier, x, y, params):

    model = GridSearchCV(estimator=classifier, param_grid=params, scoring='f1', cv=10)
    model.fit(x, y)
    best_model = model.best_estimator_

    print(model.best_params_)
    print('best f1:%f' % model.best_score_)
    return best_model

def ClassificationByLR(X, Y, testX, testingcodeN):
    LinearR = LinearRegression()
    LR_tuned_parameters = {'normalize': [True, False]}

    model = optimizeParameter(LinearR, X, Y, LR_tuned_parameters)
    LR_pred = model.predict(testX)
    LR_pred3 = []
    for j in range(len(LR_pred)):
        if testingcodeN[j] != 0:
            if LR_pred[j] > 0.5:
                LR_pred3.append((LR_pred[j] / testingcodeN[j]) + 100000)
            else:
                LR_pred3.append((LR_pred[j] / testingcodeN[j]) - 100000)
        else:
            LR_pred3.append(-100000000)


    return LR_pred3

def ClassificationByEASC(X, Y, testX, testingcodeN):
    NB = GaussianNB()
    LR_pred = NB.fit(X, Y).predict_proba(testX)
    LR_pred = [p[1] for p in LR_pred]
    LR_pred3 = []
    for j in range(len(LR_pred)):
        if testingcodeN[j] != 0:
            if LR_pred[j] > 0.5:
                LR_pred3.append((LR_pred[j] / testingcodeN[j]) + 100000)
            else:
                LR_pred3.append((LR_pred[j] / testingcodeN[j]) - 100000)
        else:
            LR_pred3.append(-100000000)

    return LR_pred3

def ClassificationByCbs(X, Y, testX, testingcodeN):
    # # 逻辑回归
    LR = LogisticRegression()
    LR_tuned_parameters = [{'tol': [0.1, 0.01, 0.001, 0.0001, 0.00001],
                           'penalty':['l1','l2', 'elasticnet', 'none'],
                           'solver':['liblinear','lbfgs', 'sag', 'newton-cg', 'saga']}
                           ]

    model = optimizeParameter(LR, X, Y, LR_tuned_parameters)
    LR_pred = model.predict_proba(testX)


    LR_pred = [p[1] for p in LR_pred]  # 概率当做缺陷个数
    LR_pred3 = []
    for j in range(len(LR_pred)):
        if testingcodeN[j] != 0:
            if LR_pred[j] > 0.5:
                LR_pred3.append((LR_pred[j] / testingcodeN[j]) + 100000)
            else:
                LR_pred3.append((LR_pred[j] / testingcodeN[j]) - 100000)
        else:
            LR_pred3.append(-100000000)
    return LR_pred3

def ClassificationByDeepForest(X, Y, testX, testingcodeN):
    model = CascadeForestClassifier()
    model.fit(X, Y)
    LR_pred = model.predict_proba(testX)

    LR_pred = [p[1] for p in LR_pred]  # 概率当做缺陷个数
    LR_pred3 = []
    for j in range(len(LR_pred)):
        if testingcodeN[j] != 0:
            if LR_pred[j] > 0.5:
                LR_pred3.append((LR_pred[j] / testingcodeN[j]) + 100000)
            else:
                LR_pred3.append((LR_pred[j] / testingcodeN[j]) - 100000)
        else:
            LR_pred3.append(-100000000)

    return LR_pred3

def ClassificationByRandomForest(X, Y, testX, testingcodeN):
    model = RandomForestClassifier()
    #调优参数，需要补充
    RFC_tuned_parameters = {'n_estimators': [i for i in range(10,150,10)]}

    model = optimizeParameter(model, X, Y, RFC_tuned_parameters)
    LR_pred = model.predict_proba(testX)

    LR_pred = [p[1] for p in LR_pred]  # 概率当做缺陷个数
    LR_pred3 = []
    for j in range(len(LR_pred)):
        if testingcodeN[j] != 0:
            if LR_pred[j] > 0.5:
                LR_pred3.append((LR_pred[j] / testingcodeN[j]) + 100000)
            else:
                LR_pred3.append((LR_pred[j] / testingcodeN[j]) - 100000)
        else:
            LR_pred3.append(-100000000)

    return LR_pred3

def ClassificationByGradientBoosting(X, Y, testX, testingcodeN):
    model = GradientBoostingClassifier()
    GB_tuned_parameters={'n_estimators':[i for i in range(10,150,10)]
                         }
    model = optimizeParameter(model, X, Y, GB_tuned_parameters)

    model.fit(X, Y)
    LR_pred = model.predict_proba(testX)

    LR_pred = [p[1] for p in LR_pred]  # 概率当做缺陷个数
    LR_pred3 = []
    for j in range(len(LR_pred)):
        if testingcodeN[j] != 0:
            if LR_pred[j] > 0.5:
                LR_pred3.append((LR_pred[j] / testingcodeN[j]) + 100000)
            else:
                LR_pred3.append((LR_pred[j] / testingcodeN[j]) - 100000)
        else:
            LR_pred3.append(-100000000)

    return LR_pred3

def transform_data(original_data):
    original_data = original_data.iloc[:, :]

    original_data = np.array(original_data)

    k = len(original_data[0])

    original_data = np.array(original_data)
    original_data_X = original_data[:, 0:k - 1]

    original_data_y = original_data[:, k - 1]
    y_list = []
    for i in original_data_y:
        if i >= 1:
            y_list.append(1)
        else:
            y_list.append(0)
    return original_data_X, y_list

def transform_data1(original_data):
    #取出特征和标签，保留标签表示的bug数目
    original_data = original_data.iloc[:, :]

    original_data = np.array(original_data)

    k = len(original_data[0])

    original_data = np.array(original_data)
    original_data_X = original_data[:, 0:k - 1]

    original_data_y = original_data[:, k - 1]

    return original_data_X, original_data_y

def calculateASFM(X):
    # 返回各行之和
    features_sum = np.sum(X)
    asfm = features_sum / (len(X)*len(X[0]))
    return asfm

def devideCluster(y_predict,X):
    n_ = len(set(y_predict))
    res = [0] * n_
    count = [0] * n_
    index = [[]] * n_
    for i in set(y_predict):
        temp = []
        for j in range(len(y_predict)):
            if i == y_predict[j]:
                res[i] += np.sum(X[j])
                count[i] += 1
                temp.append(j)
        index[i] = temp
    asfm = [res[i] / count[i] for i in range(len(res))]
    mean = np.mean(asfm)
    defectX = []
    undefectX = []
    for i in range(len(asfm)):
        if asfm[i] >= mean:
            defectX += np.array(X)[index[i]].tolist()
        else:
            undefectX += np.array(X)[index[i]].tolist()
    defectY = [1]*len(defectX)
    undefectY = [0]*len(undefectX)
    return defectX, defectY, undefectX, undefectY

if __name__ == '__main__':

    """
    这个是EASC等模型的实验
    """
    dataset_train = pd.core.frame.DataFrame()
    dataset_test = pd.core.frame.DataFrame()

    folder_path = '../CrossversionData/'

    #读取文件
    for root, dirs, files, in os.walk(folder_path):
        if root == folder_path:
            thisroot = root
            for dir in dirs:
                dir_path = os.path.join(thisroot, dir)

                for root, dirs, files, in os.walk(dir_path):
                    if (files[0][-7:-4] < files[1][-7:-4]):
                        file_path_train = os.path.join(dir_path, files[0])
                        file_path_test = os.path.join(dir_path, files[1])
                        trainingfile = files[0]
                        testingfile = files[1]
                    else:
                        file_path_train = os.path.join(dir_path, files[1])
                        file_path_test = os.path.join(dir_path, files[0])
                        trainingfile = files[1]
                        testingfile = files[0]

                    #print('files[0][-7:-4]', files[0][-7:-4])
                    #print('files[1][-7:-4]', files[1][-7:-4])
                    #print(files[0][-7:-4], '>', files[1][-7:-4])
                    print('train', file_path_train)
                    print('test', file_path_test)
                    print('***********************************')

                    dataset_train = pd.read_csv(file_path_train)
                    dataset_test = pd.read_csv(file_path_test)
                    training_data_x, training_data_y = transform_data(dataset_train)
                    testing_data_x, testing_data_y = transform_data(dataset_test)
                    # model = RandomUnderSampler()
                    # training_data_x, training_data_y=model.fit_resample(training_data_x, training_data_y)

                    testingcodeN = testing_data_x[:, 10]

                    functions = {'CBS+': ClassificationByCbs,
                                 'EALR':ClassificationByLR,
                                 'EASC':ClassificationByEASC,
                                 'RF':ClassificationByRandomForest,
                                 'GB':ClassificationByGradientBoosting,
                                 'DF':ClassificationByDeepForest,
                                 'SERS':ClassificationBySERS
                                 }
                    #functions_name = ['EALR',  'CBS+', 'RF', 'GB', 'SERS']
                    functions_name = ['SERS']
                    #functions_name = ['EALR']
                    for fname in functions_name:
                        print('=========================================')
                        print(fname)
                        print('=========================================')
                        resultlist = []
                        y_predict = functions[fname](training_data_x, training_data_y, testing_data_x, testingcodeN)

                        sort_y = np.argsort(y_predict)[::-1]
                        testY = np.array(testing_data_y)[sort_y].tolist()
                        sorted_code = testingcodeN[sort_y]
                        Precisionx, Recallx, F1x, IFA, PofB, PMI = Performance(testY,
                                                                                                   y_predict,
                                                                                                   sorted_code)
                        precision, recall, f1 = evaluate_classify(testY, y_predict)
                        header = ["Precision", "Recall", "F1", "Precisionx", "Recallx", "F1x", "IFA", "Pofb",
                                  "PMI"]
                        performance_result = [precision, recall, f1, Precisionx, Recallx, F1x, IFA, PofB, PMI]
                        resultlist.append(performance_result)
                        datasetfile = trainingfile[:-4] + '_' + testingfile[:-4]
                        df = pd.DataFrame(data=resultlist, columns=header)
                        if not os.path.exists("../Output/{0}".format(datasetfile)):
                            os.makedirs("../Output/{0}".format(datasetfile))
                        df.to_csv("../Output/{0}/{1}.csv".format(datasetfile, fname), index=False)

