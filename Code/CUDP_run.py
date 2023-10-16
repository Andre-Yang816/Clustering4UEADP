# coding=utf-8
import numpy as np
import pandas as pd
from sklearn.naive_bayes import GaussianNB
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
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




def ClassificationByLR(X, Y, testX, testingcodeN):
    LR = LogisticRegression()
    LR_pred = LR.fit(X, Y).predict_proba(testX)

    LR_pred = [p[1] for p in LR_pred]
    LR_pred1=[]

    return LR_pred1

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
    LR_pred = LR.fit(X, Y).predict_proba(testX)
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

def ClassificationByFeature(feature):
    result = []
    for f in feature:
        result.append(1/f)
    return np.array(result)

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
                    training_data_x, training_data_y = transform_data(
                        dataset_train)
                    testing_data_x, testing_data_y = transform_data(
                        dataset_test)
                    # model = RandomUnderSampler()
                    # training_data_x, training_data_y=model.fit_resample(training_data_x, training_data_y)


                    # step1：测试集聚类 （30种方法）
                    functions = [(ManualUp, testing_data_x)]

                    # functions = [
                    #              (KmeansCluster, 2, testing_data_x),
                    #              (AgglomerativeCluster, 2, testing_data_x),
                    #              (BirchCluster, 2, testing_data_x),
                    #              (KmedoidsCluster, 2, testing_data_x),
                    #              (MiniBatchKMeansCluster, 2, testing_data_x),
                    #              (MeanShiftCluster, testing_data_x),
                    #              (AffinityPropagationCluster, testing_data_x),
                    #
                    #              (BsasCluster, 2, 1.0, testing_data_x),
                    #              (CureCluster, 2, testing_data_x),
                    #              (DbscanCluster, 3, 0.5, testing_data_x),
                    #              (MbsasCluster, 2, 1.0, testing_data_x),
                    #              (OpticsCluster, 3, 0.5, testing_data_x),
                    #              (RockCluster, 2, 1, testing_data_x),
                    #              (Somsc, 2, testing_data_x),
                    #              (SyncsomCluster, testing_data_x),
                    #
                    #              (BangCluster, testing_data_x),
                    #              (Kmeans_plusplusCluster, testing_data_x),
                    #              (EmaCluster, testing_data_x),
                    #              (FcmCluster, testing_data_x),
                    #              (GmeansCluster, testing_data_x),
                    #              (TtsasCluster, testing_data_x),
                    #              (Xmeans, testing_data_x)
                    #              ]



                    for func, *args in functions:
                        y_predict, func_name = func(*args)
                        performance_result = [[0] * 9] * 20
                        iterations = 1
                        print('方法：{0}'.format(func_name))
                        for p in range(iterations):
                            # 对聚类结果划分 有缺陷还是无缺陷 使用ASFM
                            # step2：对有缺陷的按照 特征值（20维）从小到大排序
                            # 1.提取出有缺陷的模块的特征
                            defectX, defectY, undefectX, undefectY = devideCluster(y_predict, testing_data_x)
                            # 2.对有缺陷的模块每个特征排序，这里暂时用第一列替代
                            # sort_axis = np.argsort(defectX[:,0])
                            # sorted_defectX = np.array(defectX)[sort_axis]
                            # 排完序之后，检查LOC的前20%的模块；无缺陷需要排序，需要拼接
                            # sorted_undefectX = np.array(undefectX)[sort_axis]
                            # 3.合并有缺陷及无缺陷
                            # predtestX = np.vstack(sorted_defectX,undefectX)
                            # index_real = [i for i in range(len(y_predict)) if testing_data_y[i] > 0]
                            # 提取代码行数，用于工作量的截取
                            testingcodeN = testing_data_x[:, 10]
                            # print(testing_data_x)
                            # preDensity = ClassificationByCbs(training_data_x, training_data_y, testing_data_x, testingcodeN)
                            # 特征循环
                            defectX = np.array(defectX)
                            undefectX = np.array(undefectX)
                            #print(undefectX)

                            for f in range(len(defectX[0])):
                                if undefectX.any():
                                    feature_d = defectX[:, f]
                                    feayure_u = undefectX[:, f]
                                    density_d = ClassificationByFeature(feature_d)
                                    density_u = ClassificationByFeature(feayure_u)
                                    sort_axis_d = np.argsort(-density_d)
                                    sorted_defectX = defectX[sort_axis_d]
                                    sort_axis_u = np.argsort(-density_u)
                                    sorted_undefectX = undefectX[sort_axis_u]
                                    # 合并标签对y排序
                                    # predtestX = np.append(sorted_defectX, sorted_undefectX, axis=0)
                                    sort_y = np.append(sort_axis_d, sort_axis_u)
                                else:
                                    feature = testing_data_x[:, f]
                                    density = ClassificationByFeature(feature)
                                    sort_y = np.argsort(-density)

                                testY = np.array(testing_data_y)[sort_y].tolist()
                                predY = np.append(defectY, undefectY)
                                sorted_code = testingcodeN[sort_y]
                                Precisionx, Recallx, F1x, IFA, PofB, PMI= Performance(testY,
                                                                                                            predY,
                                                                                                            sorted_code)
                                #popt = Popt(testY, sorted_code, sort_y)

                                precision, recall, f1 = evaluate_classify(testY, predY)
                                header = ["Precision", "Recall", "F1", "Precisionx", "Recallx", "F1x", "IFA", "Pofb",
                                          "PMI"]
                                temp = [precision, recall, f1, Precisionx, Recallx, F1x, IFA, PofB, PMI]
                                performance_result[f] = list(np.add(performance_result[f], temp))

                        performance_result = np.array(performance_result) / iterations

                        datasetfile = trainingfile[:-4] + '_' + testingfile[:-4]
                        df = pd.DataFrame(data=performance_result, columns=header)
                        if not os.path.exists("../Output/{0}".format(datasetfile)):
                            os.makedirs("../Output/{0}".format(datasetfile))
                        df.to_csv("../output/{0}/{1}.csv".format(datasetfile, func_name), index=False)