import os
import pandas as pd
import numpy as np
import sys
import seaborn as sns
import matplotlib.pyplot as plt
import matplotlib as mpl
from matplotlib.transforms import TransformedBbox
import copy


# Plot cross-lingual results as heatmap

mpl.rcParams.update({'font.size': 10})
# plt.rcParams["font.family"] = "Times New Roman"
# plt.rc('text', usetex=True)

def plot_heat(df_matrix, target_path, model, metric, show_cbar=True):

    lang_order = ['da', 'nb', 'sv', 'en', 'it', 'pt', 'ar', 'he', 'ka', 'sl', 'zh']
    df_matrix = df_matrix[lang_order]
    df_matrix = df_matrix.reindex(lang_order)

    if not os.path.exists(target_path):

        os.makedirs(target_path)

    plt.rcParams['figure.figsize'] = 4, 3

    df_annos = copy.deepcopy(df_matrix)
    annotations = df_annos.to_numpy()
    df_annos = pd.DataFrame(annotations)

    for col in df_annos.columns:

        df_annos[col] = df_annos[col].apply(lambda x: str(x)[1:4] if x>0 else '')

    annotations = df_annos.values.tolist()

    cmap = sns.color_palette("blend:#FFFFFF,#ebeb6e,#82D2D6,#0198A7", as_cmap=True)
    ax = sns.heatmap(df_matrix, vmin=0, vmax=1, annot=annotations, fmt='', cmap='YlGn', cbar=show_cbar, linewidth=.5, cbar_kws={'ticks': [0.2, 0.4, 0.6, 0.8], 'label': ''}, annot_kws={"size":6})
    ax.xaxis.tick_top()
    ax.xaxis.set_label_position('top')

    plt.yticks(rotation=0, ha='right')

    model_name = model

    if '_' in model:

        model_name = model[:model.index('_')]
        
    plt.title(model_name, y=-0.15)
    # plt.title(model + '('+metric+')', y=-0.15)

    plt.rcParams['savefig.dpi'] = 500
    plt.tight_layout()
    plt.savefig(os.path.join(target_path, model + '_' + metric + ".pdf"), transparent=True)

    plt.clf()
    plt.cla()
    plt.close()
