from pathlib import Path
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.datasets import load_wine

wine = load_wine()
data = wine.data
lables = wine.target
feaures = wine.feature_names
df = pd.DataFrame(data, columns=feaures)
def ShowGRAHeatMap(DataFrame, func):
    ylabels = DataFrame.columns.values.tolist()
    f, ax = plt.subplots(figsize=(10, 10))
    #ax.set_title('{0}'.format(func),fontsize=18,weight='heavy',family='Times New Roman')
    with sns.axes_style("white"):
        sns.heatmap(DataFrame,
                    cmap="YlGnBu",
                    annot=True,
                    yticklabels=ylabels,
                    annot_kws={'size':18,'family':'Times New Roman'},
                    cbar_kws={"orientation": "horizontal", "shrink": 1}
                    )
    ax.set_xticklabels(ax.get_xticklabels(), rotation=35, fontsize=15, family='Times New Roman',weight='heavy')
    ax.set_yticklabels(ax.get_yticklabels(), fontsize=12, family='Times New Roman',weight='heavy')
    Path('../picture/'.format(func)).mkdir(parents=True, exist_ok=True)
    plt.savefig('../picture/RQ4_{0}.png'.format(func),bbox_inches='tight')


if __name__ == '__main__':
    functions =  ['Kmedoids.LOC', 'KmeansPlus.LOC', 'Cure.LOC']

    for function in functions:
        data = pd.read_csv("../output/{0}.csv".format(function))
        ShowGRAHeatMap(data, function)
