import math

from pandas import np


def Performance(real, pred, loc, percentage=0.2):
    if (len(pred) != len(real)) or (len(pred) != len(loc) or (len(loc) != len(real))):
        print("预测缺陷数目或密度与真实缺陷数目或密度，输入长度不一致！")
        exit()

    M = len(real)  # 总的模块数目M
    L = sum(loc)  # 总的代码LOC：
    P = sum([1 if i > 0 else 0 for i in real])  # 获取真实有缺陷的模块数P,也就是看真实缺陷列中大于0的有多少个
    m = None
    Q = sum(real)  # 缺陷个数总数Q

    # 获取前percentage的LOC所占的模块数目m，这里的m是个数，而不是下标
    locOfPercentage = percentage * L
    sum_ = 0
    for i in range(len(loc)):
        sum_ += loc[i]
        if (sum_ > locOfPercentage):
            m = i
            break
        elif (sum_ == locOfPercentage):
            m = i + 1
            break
    # 获得前percentage的LOC所占的模块，含有的cc总数

    PMI = m / M

    tp = sum([1 if real[j] > 0 and pred[j] > 0 else 0 for j in range(m)])
    fn = sum([1 if real[j] > 0 and pred[j] <= 0 else 0 for j in range(m)])
    fp = sum([1 if real[j] <= 0 and pred[j] > 0 else 0 for j in range(m)])
    tn = sum([1 if real[j] <= 0 and pred[j] <= 0 else 0 for j in range(m)])
    # print('tp:{0},fn:{1},fp:{2},tn:{3}'.format(tp,fn,fp,tn))
    if (tp + fn + fp + tn == 0):
        Precisionx = 0
    else:
        Precisionx = (tp + fn) / (tp + fn + fp + tn)

    if (P == 0):
        Recallx = 0
    else:
        Recallx = (tp + fn) / P

    if (PMI == 0):
        recallPmi = 0
    else:
        recallPmi = Recallx / PMI

    if (Recallx + Precisionx == 0):
        F1x = 0
    else:
        F1x = 2 * Recallx * Precisionx / (Recallx + Precisionx)

    # 获取当我们在m个模块中第一次检测出一个真实缺陷的时候，已经检测了的LOC数目(IFLA)和module数目(IFMA)，和cc数目(IFCCA)
    IFLA = 0
    IFMA = 0

    for i in range(m):
        if (real[i] > 0):
            break
        else:
            IFLA += loc[i]

            IFMA += 1
    PofB = sum([real[j] if real[j] > 0 else 0 for j in range(m)]) / Q
    # 2021.10.09加入一个新指标PoFbPmi
    # if (Q == 0 and PMI == 0):
    #     PofBPmi = 0
    # else:
    #     PofBPmi = PofB / PMI
    return Precisionx, Recallx, F1x, IFMA, PofB, PMI

def Popt(Yt,loc,pred_index):
    N = sum(Yt)
    xcost = loc
    xcostsum = sum(xcost)
    optimal_index = [j / i if j != 0 and i != 0 else 0 for i, j in zip(xcost, Yt)]
    optimal_index = list(np.argsort(optimal_index))
    optimal_index.reverse()

    optimal_X = [0]
    optimal_Y = [0]
    for i in optimal_index:
        optimal_X.append(xcost[i] / xcostsum + optimal_X[-1])
        optimal_Y.append(Yt[i] / N + optimal_Y[-1])

    wholeoptimal_auc = 0.
    prev_x = 0
    prev_y = 0
    for x, y in zip(optimal_X, optimal_Y):
        if x != prev_x:
            wholeoptimal_auc += (x - prev_x) * (y + prev_y) / 2.
            prev_x = x
            prev_y = y
    pred_X = [0]
    pred_Y = [0]
    for i in pred_index:
        pred_X.append(xcost[i] / xcostsum + pred_X[-1])
        pred_Y.append(Yt[i] / N + pred_Y[-1])

    wholepred_auc = 0.
    prev_x = 0
    prev_y = 0
    for x, y in zip(pred_X, pred_Y):
        if x != prev_x:
            wholepred_auc += (x - prev_x) * (y + prev_y) / 2.
            prev_x = x
            prev_y = y
    optimal_index.reverse()
    mini_X = [0]
    mini_Y = [0]
    for i in optimal_index:
        mini_X.append(xcost[i] / xcostsum + mini_X[-1])
        mini_Y.append(Yt[i] / N + mini_Y[-1])

    wholemini_auc = 0.
    prev_x = 0
    prev_y = 0
    for x, y in zip(mini_X, mini_Y):
        if x != prev_x:
            wholemini_auc += (x - prev_x) * (y + prev_y) / 2.
            prev_x = x
            prev_y = y
    wholemini_auc = 1 - (wholeoptimal_auc - wholemini_auc)
    wholenormOPT = ((1 - (wholeoptimal_auc - wholepred_auc)) - wholemini_auc) / (1 - wholemini_auc)
    return wholenormOPT

def AUC(label, pre):
    pos = []
    neg = []
    auc = 0
    for index,l in enumerate(label):
        if l == 0:
            neg.append(index)
        else:
            pos.append(index)
    for i in pos:
        for j in neg:
            if pre[i] > pre[j]:
                auc += 1
            elif pre[i] == pre[j]:
                auc += 0.5
    if len(pos)==0 or len(neg)==0:
        return 0
    else:
        return auc * 1.0 / (len(pos)*len(neg))

def evaluate_classify(Yt, pred):
    TN = 0
    FP = 0
    FN = 0
    TP = 0
    for i in range(len(Yt)):
        if Yt[i] == 0 and pred[i] == 0:
            TN += 1
        elif Yt[i] == 0 and pred[i] > 0:
            FP += 1
        elif Yt[i] > 0 and pred[i] == 0:
            FN += 1
        elif Yt[i] > 0 and pred[i] > 0:
            TP += 1
    if TN + FP + FN + TP != 0:
        Accuracy = (TN + TP) / (TN + FP + FN + TP)
    else:
        Accuracy = 0.0
    if FN + TP != 0:
        Recall = TP / (TP + FN)
    else:
        Recall = 0.0
    if FP + TP != 0:
        Precision = TP / (TP + FP)
    else:
        Precision = 0.0
    if Precision + Recall !=0:
        F1_measure = 2 * Precision * Recall / (Precision + Recall)
    else:
        F1_measure = 0.0
    auc = AUC(Yt, pred)
    if (TP+FP)*(TP+FN)*(TN+FP)*(TN+FN) != 0:
        mcc = (TP*TN-FP*FN)/math.sqrt((TP+FP)*(TP+FN)*(TN+FP)*(TN+FN))
    else:
        mcc = 0.
    return Precision, Recall, F1_measure