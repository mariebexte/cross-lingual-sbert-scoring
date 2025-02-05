import matplotlib
import os

import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

from config import EPIRLS, ASAP_T, ASAP_M


def plot_dataset(score_distribution, dataset_name, no_y=False, add_legend=True):


    short_names = {'ePIRLS': 'ePIRLS', 'ASAP_translated': 'ASAP_T', 'ASAP_multilingual': 'ASAP_M'}
    dataset_name = short_names[dataset_name]

    df_scores = pd.read_csv(score_distribution)
    df_scores.sort_values(by=['0', '1'], inplace=True)
    plt.rcParams['figure.figsize'] = 0.2*len(df_scores),3

    colors = ["#9955F7", "#52C8D9", "#ACD35C", "#5B8B5D"]

    # colors = ['#39648A', '#7C80EF', '#AFCEF7', 'lightgreen', 'blue', 'lightblue', 'red']
    #colors = ['#c53c69', '#1e0e3f', '#4f4086', '#9a72aa']
    # colors = ['#ab544d', '#112f2c', '#099197', '#96d1aa']
    # colors = ['#f15a30', '#00b49b', '#bce4e5', '#cab356']
    df_scores.set_index('prompt').plot(kind='bar', stacked=True, color=colors, width=0.9, legend=add_legend)

    plt.ticklabel_format(style='plain', useOffset=False, axis='y')
    plt.rcParams['savefig.dpi'] = 500
    # plt.tight_layout()

    # latex_names = {'ASAP_M': r'$\mathbf{{ASAP}_{M}}}$', 'ASAP_T': r'$\mathbf{{ASAP_{T}}}$', 'ePIRLS': r'$\mathbf{{ePIRLS}}$'}
    latex_names = {'ASAP_M': r'$\mathregular{ASAP}_{\mathregular{M}}}$', 'ASAP_T': r'$\mathregular{ASAP_{\mathregular{T}}}$', 'ePIRLS': r'$\mathregular{ePIRLS}$'}
    plt.title(latex_names[dataset_name], fontsize=15)
    plt.xlabel('Prompt', fontsize=14)

    plt.xticks([])
    plt.box(False)

    if no_y:

        plt.yticks([])

    else:

        plt.ylabel('Score Frequency', fontsize=14)

    if add_legend:
    
        plt.legend(title='Score', reverse=True, loc=(1.6, 0.004), title_fontsize=14)

    plt.savefig(score_distribution[:score_distribution.rindex('.')] + '.png', transparent=True, bbox_inches="tight")

    plt.clf()
    plt.cla()
    plt.close()


for dataset in EPIRLS, ASAP_T, ASAP_M:

    no_y = True
    
    if dataset['dataset_name'] == 'ePIRLS':

        no_y = False
    
    add_legend = False

    if dataset['dataset_name'] == 'ASAP_multilingual':

        add_legend = True

    plot_dataset(score_distribution=os.path.join('/data', dataset['dataset_name'] + '_scores.csv'), no_y=no_y, dataset_name=dataset['dataset_name'], add_legend=add_legend)

