# coding=utf-8
import numpy as np


class PerformanceMeasure():
    # real_list和pred_list是真实缺陷个数和预测缺陷个数，loc是代码行数，cc是代码复杂度。
    def __init__(self, real_list=None, pred_list=None, loc=None,  percentage=0.2, ranking="density", cost="loc"):
        self.real = real_list
        self.pred = pred_list
        self.loc = loc
        self.percentage = percentage
        self.ranking = ranking
        self.cost=cost

    def Performance(self):
        if (len(self.pred) != len(self.real)) or (len(self.pred) != len(self.loc) or (len(self.loc) != len(self.real))):
            print("预测缺陷数目或密度与真实缺陷数目或密度，输入长度不一致！")
            exit()

        M = len(self.real)  # 总的模块数目M
        L = sum(self.loc)  # 总的代码LOC：
        P = sum([1 if i > 0 else 0 for i in self.real])  # 获取真实有缺陷的模块数P,也就是看真实缺陷列中大于0的有多少个
        m=None
        Q = sum(self.real)  # 缺陷个数总数Q

        # 获取前percentage的LOC所占的模块数目m，这里的m是个数，而不是下标
        locOfPercentage = self.percentage * L
        sum_ = 0
        for i in range(len(self.loc)):
            sum_ += self.loc[i]
            if (sum_ > locOfPercentage):
                m = i
                break
            elif (sum_ == locOfPercentage):
                m = i + 1
                break
        # 获得前percentage的LOC所占的模块，含有的cc总数

        PMI = m / M


        tp = sum([1 if self.real[j] > 0 and self.pred[j] > 0 else 0 for j in range(m)])
        fn = sum([1 if self.real[j] > 0 and self.pred[j] <= 0 else 0 for j in range(m)])
        fp = sum([1 if self.real[j] <= 0 and self.pred[j] > 0 else 0 for j in range(m)])
        tn = sum([1 if self.real[j] <= 0 and self.pred[j] <= 0 else 0 for j in range(m)])
        # print('tp:{0},fn:{1},fp:{2},tn:{3}'.format(tp,fn,fp,tn))
        if (tp + fn + fp + tn == 0):
            Precisionx = 0
        else:
            Precisionx = (tp + fn) / (tp + fn + fp + tn)

        if (P == 0):
            Recallx = 0
        else:
            Recallx = (tp + fn) / P

        if (P == 0 and PMI == 0):
            recallPmi = 0
        else:
            recallPmi = Recallx / PMI

        if (Recallx + Precisionx==0):
            F1x = 0
        else:
            F1x = 2 * Recallx * Precisionx / (Recallx + Precisionx)

        # 获取当我们在m个模块中第一次检测出一个真实缺陷的时候，已经检测了的LOC数目(IFLA)和module数目(IFMA)，和cc数目(IFCCA)
        IFLA = 0
        IFMA = 0

        for i in range(m):
            if (self.real[i] > 0):
                break
            else:
                IFLA += self.loc[i]

                IFMA += 1

        PofB = sum([self.real[j] if self.real[j] > 0 else 0 for j in range(m)]) / Q
        #2021.10.09加入一个新指标PoFbPmi
        if (Q == 0 and PMI == 0):
            PofBPmi = 0
        else:
            PofBPmi = PofB / PMI



        return   Precisionx, Recallx,F1x, IFMA,PMI, recallPmi,PofB,PofBPmi

    def POPT(self):
        '''
        有四个模块m1,m2,m3,m4，真实缺陷个数分别为5，2，1，0,self.real=[5，2，1，0],
        代码行数为10，2，100，50，真实的缺陷密度为[0.5,1,0.01,0]
        预测m1缺陷个数为1，m2缺陷个数为0，m3缺陷个数为1，m4缺陷个数为50,self.pred=[1,0,1,50],
        预测的缺陷密度为[0.1,0,0.01,1]
        optimal model’s curve (0,0),（2/162,0.25）,(12/162,0.875),(112/162,1),(1,1)
        worst model’s curve (0,0),(50/162,0,(150/162,0.125),(160/162,0.75),(1,1)
        prediction model’s curve (0,0),(50/162,0),(60/162,0.625),(160/162,0.75),(1,1)
        from sklearn import metrics
        optimalx = np.array([0,2/162, 12/162, 112/162, 1])
        optimaly = np.array([0,0.25, 0.875, 1, 1])
        optimalauc=metrics.auc(optimalx, optimaly)
        worsetx = np.array([0,50/162, 150/162, 160/162, 1])
        worsety = np.array([0,0.0, 0.125, 0.75, 1])
        worsetauc=metrics.auc(worsetx, worsety)
        predx = np.array([0,50/162, 60/162, 160/162, 1])
        predy = np.array([0,0.0, 0.625, 0.75, 1])
        predauc=metrics.auc(predx, predy)
        popt=1-(optimalauc-predauc)
        minpopt=1-(optimalauc-worsetauc)
        normpopt=(popt-minpopt)/(1-minpopt)
        print (normpopt)
        输出得 0.446265938069
        '''


        Q = sum(self.real)  # 缺陷个数总数Q

        if (self.ranking == "density" and self.cost == 'loc'):  # 按照缺陷密度从大到小排序module，x轴的坐标是检测的loc的百分比
            #print("此时是按照预测{0}从大到小对模块进行排序，x轴的坐标是{1}".format(self.ranking, self.cost))
            pred_index = [j / i if j != 0 and i != 0 else 0 for i, j in zip(self.loc, self.pred)]
            pred_index = list(np.argsort(pred_index))
            pred_index.reverse()
            xcost=self.loc
            xcostsum=sum(xcost)


        elif (self.ranking == "defect" and self.cost == 'module'): #按照缺陷个数从大到小排序module，检测前percentage的loc
            #print("此时是按照预测{0}从大到小对模块进行排序，选取前{1}的{2}".format(self.ranking, self.percentage, self.cost))
            pred_index = self.pred
            pred_index = list(np.argsort(pred_index))
            pred_index.reverse()
            ###这里不会
            xcost = [1 for i in range(len(self.pred))]
            xcostsum = sum(xcost)


        elif (self.ranking == "complexitydensity" and self.cost == 'cc'):  # 按照缺陷复杂度密度从大到小排序module，x轴的坐标是检测的cc的百分比
            #print("此时是按照预测{0}从大到小对模块进行排序，x轴的坐标是{1}".format(self.ranking, self.cost))
            pred_index = [j / i if j != 0 and i != 0 else 0 for i, j in zip(self.cc, self.pred)]
            pred_index = list(np.argsort(pred_index))
            pred_index.reverse()
            xcost = self.cc
            xcostsum = sum(xcost)

        else:
            print("参数传入错误")
            exit()

        optimal_index = [j / i if j != 0 and i != 0 else 0 for i, j in zip(xcost, self.real)]
        optimal_index = list(np.argsort(optimal_index))
        optimal_index.reverse()

        optimal_X = [0]
        optimal_Y = [0]
        for i in optimal_index:
            optimal_X.append(xcost[i] / xcostsum + optimal_X[-1])
            optimal_Y.append(self.real[i] / Q + optimal_Y[-1])

        wholeoptimal_auc = 0.
        prev_x = 0
        prev_y = 0
        for x, y in zip(optimal_X, optimal_Y):
            if x != prev_x:
                wholeoptimal_auc += (x - prev_x) * (y + prev_y) / 2.
                #print('wholeoptimalx', (x - prev_x))
                #print('wholeoptimaly', (y + prev_y))
                prev_x = x
                prev_y = y

        #print('wholeoptimalauc', wholeoptimal_auc)

        pred_X = [0]
        pred_Y = [0]
        for i in pred_index:
            pred_X.append(xcost[i]/ xcostsum + pred_X[-1])
            pred_Y.append(self.real[i] / Q + pred_Y[-1])

        wholepred_auc = 0.
        prev_x = 0
        prev_y = 0
        for x, y in zip(pred_X, pred_Y):
            if x != prev_x:
                wholepred_auc += (x - prev_x) * (y + prev_y) / 2.
                #print('predx', (x - prev_x))
                #print('predy', (y + prev_y))
                prev_x = x
                prev_y = y

        #print('wholepredauc',wholepred_auc)

        optimal_index.reverse()
        mini_X = [0]
        mini_Y = [0]
        for i in optimal_index:
            mini_X.append(xcost[i]/ xcostsum + mini_X[-1])
            mini_Y.append(self.real[i] / Q + mini_Y[-1])

        wholemini_auc = 0.
        prev_x = 0
        prev_y = 0
        for x, y in zip(mini_X, mini_Y):
            if x != prev_x:
                wholemini_auc += (x - prev_x) * (y + prev_y) / 2.
                #print('wholeworstpredx', (x - prev_x))
                #print('wholeworstpredy', (y + prev_y))
                prev_x = x
                prev_y = y

        #print('worstwholeauc',wholemini_auc)

        wholemini_auc = 1 - (wholeoptimal_auc - wholemini_auc)
        wholenormOPT = ((1 - (wholeoptimal_auc - wholepred_auc)) - wholemini_auc) / (1 - wholemini_auc)

        return wholenormOPT


if __name__ == '__main__':
    real = [0, 4, 1, 0, 1, 0, 1, 0, 0, 0] #真实缺陷个数
    pred = [60, 20, 22.5, 9, 9.9, 0, -80, -270, -100, -150]  #预测缺陷个数
    loc = [200, 100, 150, 90, 110, 500, 400, 900, 250, 300]




    #当y值是缺陷个数时，论文主要要这些指标,和解释指标的word文档进行测试的时候，percentage是取的0.4.
    Precision20module, Recall20module, F120module, Precisionx20module, Recallx20module, F1x20module, PF20module, falsealarmrate20module, \
    IFMA20module, IFLA20module, IFCCA20module, PMI20module, PLI20module, PCCI20module, PofB20module=PerformanceMeasure(
        real, pred, loc, 0.2,'defect','module').getSomePerformance()

