import os
import pandas as pd

header = [
    'Kmeans', 'Kmedoids', 'Xmeans', 'FCM', 'Gmeans', 'MBKmeans',
    'Kmeans++',
    'BIRCH', 'CURE', 'ROCK', "AHC",
    "DBSCAN", 'OPTICS', 'MeanShift',
    'SOMAC', 'SYNCSOM', 'EMA',
    "AP",
    "BSAS", 'MBSAS', 'TTSAS',
    "BANG"
]
#functions = ["AP", "Agglomerative", "Bang", "Birch", "Bsas", "Cure", "Dbscan",
          #'EMA','Fcm','Gmeans','KmeansPlus', 'Kmeans', 'Kmedoids', 'Mbsas', 'MeanShift', 'MiniBatchKmeans','Optics',
          #'Rock', 'Somsc', 'Syncsom', 'Ttsas', 'Xmeans']
functions = ['ManualUp',
             'Kmeans', 'Kmedoids', 'Xmeans', 'Fcm', 'Gmeans', 'MiniBatchKmeans', 'KmeansPlus',
             'Birch', 'Cure', 'Rock', "Agglomerative",
             "Dbscan", 'Optics', 'MeanShift',
             'Somsc', 'Syncsom', 'EMA',
             "AP",
             "Bsas", 'Mbsas', 'Ttsas',
             "Bang"
                      ]
features = ['WMC','DIT','NOC','CBO','RFC','LCOM','Ca','Ce','NPM','LCOM3',
            'LOC','DAM','MOA','MFA','CAM','IC','CBM','AMC','Max_CC','Avg_CC']

def writeToFile(file, name, argMax):
    new_header = ['Cross-version']
    for h in functions:
        new_header.append(h + '.' + features[argMax[h]])
    dataf = pd.DataFrame(columns=new_header, data=file)
    path = "../Result_rq1"
    if not os.path.exists(path):
        os.makedirs(path)
    file_path = "{0}/{1}.xlsx".format(path, name)
    dataf.to_excel(file_path, index=False)

def getArgMax(path):
    folder_path = path
    init_list = [[0.]*22]*20
    result_data = pd.DataFrame(columns=functions, data=init_list)
    # 读取文件
    for root, dirs, files, in os.walk(folder_path):
        if root == folder_path:
            thisroot = root
            for dir in dirs:
                dir_path = os.path.join(thisroot, dir)
                for root, dirs, files, in os.walk(dir_path):
                    for f in functions:
                        #funcname = f[:-4]
                        file_path = os.path.join(dir_path, f+'.csv')
                        dataset = pd.read_csv(file_path)
                        pofb = dataset.loc[:, "Pofb"]
                        result_data[f] += pofb
    argmax = result_data.idxmax()
    return argmax

def getMeasure(argMax):
    resultlist_pofb = []
    resultlist_ifa = []
    resultlist_pmi = []
    resultlist_precision = []
    resultlist_recall = []
    resultlist_f1 = []
    resultlist_precision20 = []
    resultlist_recall20 = []
    resultlist_f120 = []

    for root, dirs, files, in os.walk(folder_path):
        if root == folder_path:
            thisroot = root
            for dir in dirs:
                dir_path = os.path.join(thisroot, dir)
                for root, dirs, files, in os.walk(dir_path):
                    temp_ifa = []
                    temp_pofb = []
                    temp_pmi = []
                    temp_precision = []
                    temp_recall = []
                    temp_f1 = []
                    temp_precision20 = []
                    temp_recall20 = []
                    temp_f120 = []

                    temp_ifa.append(dir)
                    temp_pofb.append(dir)
                    temp_pmi.append(dir)
                    temp_precision.append(dir)
                    temp_recall.append(dir)
                    temp_f1.append(dir)
                    temp_precision20.append(dir)
                    temp_recall20.append(dir)
                    temp_f120.append(dir)


                    for f in functions:
                        #funcname = f[:-4]
                        feature_n = argMax[f]
                        file_path = os.path.join(dir_path, f+'.csv')
                        dataset = pd.read_csv(file_path)
                        pofb = dataset["Pofb"].loc[feature_n]
                        ifa = dataset['IFA'].loc[feature_n]
                        pmi = dataset['PMI'].loc[feature_n]
                        precision = dataset['Precision'].loc[feature_n]
                        recall = dataset['Recall'].loc[feature_n]
                        f1 = dataset['F1'].loc[feature_n]
                        precision20 = dataset['Precisionx'].loc[feature_n]
                        recall20 = dataset['Recallx'].loc[feature_n]
                        f120 = dataset['F1x'].loc[feature_n]

                        temp_ifa.append(ifa)
                        temp_pofb.append(pofb)
                        temp_pmi.append(pmi)
                        temp_precision.append(precision)
                        temp_recall.append(recall)
                        temp_f1.append(f1)
                        temp_precision20.append(precision20)
                        temp_recall20.append(recall20)
                        temp_f120.append(f120)


                    resultlist_ifa.append(temp_ifa)
                    resultlist_pofb.append(temp_pofb)
                    resultlist_pmi.append(temp_pmi)
                    resultlist_precision.append(temp_precision)
                    resultlist_recall.append(temp_recall)
                    resultlist_f1.append(temp_f1)
                    resultlist_precision20.append(temp_precision20)
                    resultlist_recall20.append(temp_recall20)
                    resultlist_f120.append(temp_f120)


    writeToFile(resultlist_ifa, "IFA", argMax)
    writeToFile(resultlist_pofb, "Pofb", argMax)
    writeToFile(resultlist_pmi, "PMI", argMax)
    writeToFile(resultlist_precision, "Precision", argMax)
    writeToFile(resultlist_recall, "Recall", argMax)
    writeToFile(resultlist_f1, "F1", argMax)
    writeToFile(resultlist_precision20, "Precisionx", argMax)
    writeToFile(resultlist_recall20, "Recallx", argMax)
    writeToFile(resultlist_f120, "F1x", argMax)



if __name__ == '__main__':
    dataset = pd.core.frame.DataFrame()
    folder_path = '../Output/'

    argmax = getArgMax(folder_path)
    #print(argmax)
    getMeasure(argmax)

