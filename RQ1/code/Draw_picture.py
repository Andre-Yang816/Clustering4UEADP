import matplotlib.pyplot as plt
import pandas as pd
import os
plt.rc('font', family='Times New Roman')
class BoxPlot:
    def __init__(self, phome):
        self.color_dict = {1: 'red', 2: "green", 3: 'blue', 4: 'orange', 5: 'purple', 6: 'yellow', 7: 'pink', 8: 'gray',
                           9: 'gray', 10: 'gray', 11: 'gray', 12: 'gray', 13: 'gray', 14: 'gray', 15: 'gray',
                           16: 'gray', 17: 'gray', 18: 'gray', 19: 'gray', 20: 'gray', 21: 'gray', 22: 'gray',
                           }

        # 9: 'olive', 10: 'plum', 11: 'c', 12: 'aqua',13: 'cyan', 14: 'teal', 15: 'skyblue', 16: 'darkblue',
        # 17: 'deepskyblue', 18: 'indigo', 19: 'darkorange', 20: 'lightcoral'}
        self.home = phome
        self.parse()
    # After passing the r program will get a txt version of which group each algorithm belongs to,
    # here the txt is converted to a csv file
    def parseTxtToCSV(self, header, txt_path):
        # The function is to convert the grouped txt to a color csv and return the path to generate the color csv
        # Note that the first line of this txt is invalid data, the data starts from the second line,
        # and each line is a key-value pair
        if not os.path.exists(txt_path):
            print("txt file：{0} does not exit!".format(txt_path))
            exit()
        method = []
        rank = []
        lines = open(txt_path, "r", encoding="utf-8").readlines()
        for line in lines:
            l = line.replace("\n", "").replace('"', "").split(" ")
            if l[0] != 'x':
                method.append(l[0])
                rank.append(int(l[1]))
        if len(header) != len(method) or len(header) != len(rank):
            print("长度不一致！")
            exit()
        header_rank = []
        for head in header:
            if head in method:
                index = method.index(head)
                header_rank.append(rank[index])
        data = []
        data.append(header)
        data.append(header_rank)
        return data
        pass

    def parse(self):
        results_path = self.home
        for file in os.listdir(results_path):
            # PBC: k-means/k-modoids/x-means/fcm/g-means/minibatchkmeans/kmeans++
            # HBC: birch/cure/rock/agglomerative
            # DBC: DBSCAN/optics/meanshift
            # MBC: somsc / syncsom / Expectation - Maximization
            # GTBC: AP
            # SBC: bsas/mbsas/ttsas
            # GBC: bang
            header = [
                      'Kmeans.LOC', 'Kmedoids.LOC', 'Xmeans.NPM', 'Fcm.Ce', 'Gmeans.LCOM3', 'MiniBatchKmeans.LOC', 'KmeansPlus.LOC',
                      'Birch.Ce', 'Cure.LOC', 'Rock.AMC', "Agglomerative.LOC",
                      "Dbscan.LOC", 'Optics.LOC', 'MeanShift.LCOM',
                      'Somsc.RFC', 'Syncsom.LCOM', 'EMA.NPM',
                      "AP.LOC",
                      "Bsas.LCOM3", 'Mbsas.LCOM3', 'Ttsas.NPM',
                      "Bang.LOC"
                      ]

            txt_file_path = self.home + '{0}'.format(file)
            color_data = self.parseTxtToCSV(header, txt_file_path)
            csv_name = file[:-4]
            csv_path = '../DrawPicData/{0}'.format(csv_name)
            all_data = pd.read_excel(csv_path)
            all_data = all_data.iloc[1:, 1:].values
            colors_nums = color_data[1]
            print("colors_nums from color_csv:", colors_nums)
            fig, ax = plt.subplots(
                figsize=(12, 2))
            ax.tick_params(direction='in')
            figure = ax.boxplot(all_data,
                                notch=False,  # notch shape
                                sym='r+',  # blue squares for outliers
                                vert=True,  # vertical box aligmnent
                                meanline=True,
                                showmeans=True,
                                patch_artist=False,
                                showfliers=False
                                )
            colors = [self.color_dict[int(i)] for i in colors_nums]
            # dict_tmp = {'red': 1, "green": 2, 'blue': 3, 'orange': 4, 'purple': 5, 'yellow': 6, 'pink': 7, 'gray': 8}
            # for color,c_value in dict_tmp.items():
            #     if color in colors:
            #         plt.scatter([], [], c=color, marker='s', s=60, label=c_value)
            #plt.legend(bbox_to_anchor=(0.5, -0.7),loc=8,ncol=9)
            for i in range(0, len(colors)):
                k = figure['boxes'][i]
                k.set(color=colors[i])
                k = figure['means'][i]
                k.set(color=colors[i], linewidth=0)
                k = figure['medians'][i]
                k.set(color=colors[i], linewidth=2)
                k = figure['whiskers'][2 * i:2 * i + 2]
                for w in k:
                    w.set(color=colors[i], linestyle='--')
                k = figure['caps'][2 * i:2 * i + 2]
                for w in k:
                    w.set(color=colors[i])
            plt.xlim((0, 23))
            if csv_name.strip('.xlsx') == 'IFA' or csv_name.strip('.xlsx') == 'Popt':
                plt.ylabel("{0}".format(csv_name.strip('.xlsx')), fontsize=10)
                if csv_name.strip('.xlsx') == 'IFA':
                    plt.axhline(y=10, color='blue',lw=1)
            elif csv_name=='Pofb.xlsx':
                plt.ylabel("PofB@20%", fontsize=10)
            elif csv_name=='PMI.xlsx':
                plt.ylabel("PMI@20%", fontsize=10)
            elif csv_name=='F1x.xlsx':
                plt.ylabel("F1@20%", fontsize=10)
            elif csv_name=='Precisionx.xlsx':
                plt.ylabel("Precision@20%", fontsize=10)
            elif csv_name == 'Recallx.xlsx':
                plt.ylabel("Recall@20%", fontsize=10)

            lenheader = len(header) + 1

            new_header = [
                'K-means.LOC', 'K-medoids.LOC', 'X-means.NPM', 'FCM.Ce', 'G-means.LCOM3', 'MiniBatchKmeans.LOC',
                'Kmeans++.LOC',
                'BIRCH.Ce', 'CURE.LOC', 'ROCK.AMC', "AHC.LOC",
                "DBSCAN.LOC", 'OPTICS.LOC', 'MeanShift.LCOM',
                'SOMAC.RFC', 'SYNC-SOM.LCOM', 'EMA.NPM',
                "AP.LOC",
                "BSAS.LCOM3", 'MBSAS.LCOM3', 'TTSAS.NPM',
                "BANG.LOC"
            ]
            plt.xticks([y for y in range(1, lenheader)], new_header, rotation=45, weight='heavy', fontsize=7.5)
            plt.yticks(fontsize=10)
            plt.rcParams['xtick.direction'] = 'in'
            plt.rcParams['ytick.direction'] = 'in'
            plt.axvline(7.5, color='black', linestyle=':')
            plt.axvline(11.5, color='black', linestyle=':')
            plt.axvline(14.5, color='black', linestyle=':')
            plt.axvline(17.5, color='black', linestyle=':')
            plt.axvline(18.5, color='black', linestyle=':')
            plt.axvline(21.5, color='black', linestyle=':')



            # PBC: k-means/k-modoids/x-means/fcm/g-means/minibatchkmeans/kmeans++
            # HBC: birch/cure/rock/agglomerative
            # DBC: DBSCAN/optics/meanshift
            # MBC: somsc / syncsom / Expectation - Maximization
            # GTBC: AP
            # SBC: bsas/mbsas/ttsas
            # GBC: bang
            plt.title("                                       PBC                             "
                      "                    HBC             "
                      "               DBC            "
                      "            MBC          "
                      "  GTBC        "
                      "     SBC     "
                      "          GBC "
                      , fontsize=11, loc='left')

            if not os.path.exists('../pictures/'):
                os.makedirs('../pictures/')
            output_file_path = '../pictures/RQ1_{0}.png'.format(csv_name[:-5])
            foo_fig = plt.gcf()
            foo_fig.savefig(output_file_path, format='png', dpi=1000, bbox_inches='tight')
            plt.clf()
            plt.close()


if __name__ == '__main__':
    #classifiers = ['LR', 'NB', 'KNN', 'RF', 'DT', 'MLP']

    BoxPlot(r"../output/")
