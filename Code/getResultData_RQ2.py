import os
import pandas as pd

functions = ['Kmedoids','KmeansPlus', 'Cure', 'ManualUp', 'RF', 'GB', 'CBS+', 'SERS']


features = ['WMC','DIT','NOC','CBO','RFC','LCOM','Ca','Ce','NPM','LCOM3',
            'LOC','DAM','MOA','MFA','CAM','IC','CBM','AMC','Max_CC','Avg_CC']

def writeToFile(file, name, argMax):
    new_header = ['Cross-version']
    for h in functions:
        if h == 'KmeansPlus':
            h='Kmeans++'
        new_header.append(h)
    dataf = pd.DataFrame(columns=new_header, data=file)
    path = "../Result_rq2"
    if not os.path.exists(path):
        os.makedirs(path)
    file_path = "{0}/{1}.xlsx".format(path, name)
    dataf.to_excel(file_path, index=False)
    # dataf.to_csv(file_path, index=False)



def getArgMax(path):
    folder_path = path
    init_list = [[0.]*8]*20
    result_data = pd.DataFrame(columns=functions, data=init_list)
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
                        temp_pofb.append(round(pofb,3))
                        temp_pmi.append(round(pmi,3))
                        temp_precision.append(round(precision,3))
                        temp_recall.append(round(recall,3))
                        temp_f1.append(round(f1,3))
                        temp_precision20.append(round(precision20,3))
                        temp_recall20.append(round(recall20,3))
                        temp_f120.append(round(f120,3))


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

