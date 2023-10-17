import numpy as np
import pandas as pd
import warnings
import os
from Code.Measure import Performance, Popt, evaluate_classify
from Code.functions import KmeansCluster, AgglomerativeCluster, BirchCluster, MeanShiftCluster, KmedoidsCluster, \
    MiniBatchKMeansCluster, AffinityPropagationCluster, BsasCluster, CureCluster, DbscanCluster, MbsasCluster, \
    OpticsCluster, RockCluster, Somsc, SyncsomCluster, BangCluster, Kmeans_plusplusCluster, ClaransCluster, \
    EmaCluster, FcmCluster, GmeansCluster, TtsasCluster, Xmeans, ManualUp

warnings.filterwarnings('ignore')

header = ["Precision", "Recall", "F1", "Precisionx", "Recallx", "F1x", "IFA", "Pofb", "PMI"]


def ClassificationByFeature(feature):
    result = []
    for f in feature:
        result.append(1/f)
    return np.array(result)


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

# def transform_data1(original_data):
#     original_data = original_data.iloc[:, :]
#
#     original_data = np.array(original_data)
#
#     k = len(original_data[0])
#
#     original_data = np.array(original_data)
#     original_data_X = original_data[:, 0:k - 1]
#
#     original_data_y = original_data[:, k - 1]
#
#     return original_data_X, original_data_y

def calculateASFM(X):
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
    dataset_train = pd.core.frame.DataFrame()
    dataset_test = pd.core.frame.DataFrame()
    folder_path = '../CrossversionData/'
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

                    functions = [(ManualUp, testing_data_x),
                                 (KmeansCluster, 2, testing_data_x),
                                 (AgglomerativeCluster, 2, testing_data_x),
                                 (BirchCluster, 2, testing_data_x),
                                 (KmedoidsCluster, 2, testing_data_x),
                                 (MiniBatchKMeansCluster, 2, testing_data_x),
                                 (MeanShiftCluster, testing_data_x),
                                 (AffinityPropagationCluster, testing_data_x),

                                 (BsasCluster, 2, 1.0, testing_data_x),
                                 (CureCluster, 2, testing_data_x),
                                 (DbscanCluster, 3, 0.5, testing_data_x),
                                 (MbsasCluster, 2, 1.0, testing_data_x),
                                 (OpticsCluster, 3, 0.5, testing_data_x),
                                 (RockCluster, 2, 1, testing_data_x),
                                 (Somsc, 2, testing_data_x),
                                 (SyncsomCluster, testing_data_x),

                                 (BangCluster, testing_data_x),
                                 (Kmeans_plusplusCluster, testing_data_x),
                                 (EmaCluster, testing_data_x),
                                 (FcmCluster, testing_data_x),
                                 (GmeansCluster, testing_data_x),
                                 (TtsasCluster, testing_data_x),
                                 (Xmeans, testing_data_x)
                                 ]



                    for func, *args in functions:
                        y_predict, func_name = func(*args)
                        performance_result = [[0] * 9] * 20
                        iterations = 1
                        print('Function：{0}'.format(func_name))
                        for p in range(iterations):
                            # Use ASFM to classify clustering results as defective or defect-free
                            # step2：Sort the defects according to their eigenvalues (20 dimensions) from small to large.
                            # 1.Extract features of defective modules
                            defectX, defectY, undefectX, undefectY = devideCluster(y_predict, testing_data_x)
                            # 2.Sort each feature of the defective module, temporarily replacing it with the first column.
                            # sort_axis = np.argsort(defectX[:,0])
                            # sorted_defectX = np.array(defectX)[sort_axis]
                            # After sorting, check the first 20% of the modules in LOC; if there are no defects, they need to be sorted and spliced.
                            # sorted_undefectX = np.array(undefectX)[sort_axis]
                            # 3.Merge defective and non-defective
                            # predtestX = np.vstack(sorted_defectX,undefectX)
                            # index_real = [i for i in range(len(y_predict)) if testing_data_y[i] > 0]
                            # Extract the number of lines of code for workload interception
                            testingcodeN = testing_data_x[:, 10]
                            # print(testing_data_x)
                            # preDensity = ClassificationByCbs(training_data_x, training_data_y, testing_data_x, testingcodeN)
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
                                    # Merge labels to sort y
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
