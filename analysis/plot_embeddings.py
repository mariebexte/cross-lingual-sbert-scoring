import os
import sys

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import umap.umap_ as umap

from config import EPIRLS, ASAP_M, ASAP_T
from copy import deepcopy
from sentence_transformers import SentenceTransformer
from sklearn.manifold import TSNE
from tqdm import tqdm
from transformers import XLMRobertaModel, XLMRobertaTokenizer
from utils import read_data

plt.rcParams['figure.figsize'] = 4.9,4.9


## Visualize embedding space

def plot_embeddings_scatterplot(df_overall, target_path, vis_column, model_name, prompt_name, suffix='', legend=False):

    if vis_column == 'Language':

        # colors = sns.color_palette('husl', 11)
        colors = ['#D0A2F8', '#8141AD', '#BB0B0F', '#DC30BD', '#0096FF', '#011993', '#ACD35C', '#5B8B5D', '#E09035', '#81D6B9', '#8D8D92']
    
    else:

        colors = ["#9955F7", "#52C8D9", "#ACD35C", "#5B8B5D"]
        # colors = ["#307099", "#868BF7", "#B0D5FC", "#7CF393"]

        # colors = ["#00499D", "#9282FF", "#4BD2FF", "#5ECE61"]
        # colors = ['#82241f', '#112f2c', '#099197', '#96d1aa']
        # colors = ["#60C6D1", "#327AAE", "#1C435F", "#122040"]
        # colors = ['#f15a30', '#00b49b', '#bce4e5', '#cab356']
        # colors = ['#ab544d', '#112f2c', '#099197', '#96d1aa']
        # sns.set_palette(colors)

    fig = sns.scatterplot(data=df_overall, x="x", y="y", hue=vis_column, legend=legend, palette=colors)
    # fig = sns.scatterplot(data=df_overall, x="x", y="y", hue=vis_column, legend=False, palette=colors)
    # sns.move_legend(fig, "upper left", bbox_to_anchor=(1, 1))

    # if vis_column == 'Language':

    #     plt.legend(title='Language', loc=(0.0, -0.45), ncol=6, columnspacing=0.1, handletextpad=0.01, fontsize=17.4, title_fontsize=18)
    
    # else:

    #     plt.legend(title='Label', loc=(0.0, -0.35), ncol=3, columnspacing=0.1, handletextpad=0.01, fontsize=17.4, title_fontsize=18)

    if legend:

        if vis_column == 'Language':

            plt.legend(title='Language', loc='upper left', ncol=1, labelspacing=0.29, columnspacing=0.1, handletextpad=0.01, fontsize=17, title_fontsize=22, bbox_to_anchor=(1.0, 0.03, 1,1))
        
        else:

            plt.legend(title='Score       ', loc='upper left', ncol=1, labelspacing=0.29, columnspacing=0.1, handletextpad=0.01, fontsize=17, title_fontsize=22, bbox_to_anchor=(1.0, 0.03, 1, 1))

    plt.title(model_name + ' (' + prompt_name + ')', fontsize=24)

    plt.xlabel(None)
    plt.ylabel(None)

    plt.xticks([])
    plt.yticks([])

    plt.rcParams['savefig.dpi'] = 500
    # plt.tight_layout()

    if legend:

        suffix = suffix + '_legend'

    # plt.savefig(os.path.join(target_path, model_name + suffix + ".pdf"), transparent=True)
    plt.savefig(os.path.join(target_path, model_name + suffix + ".pdf"), transparent=True, bbox_inches="tight")

    plt.clf()
    plt.cla()
    plt.close()


def get_xlmr_embedding(answer, answer_col, tokenizer, bert_model):

    inputs = tokenizer(answer, return_tensors='pt', padding=True, truncation=True)
    inputs = inputs.to('cuda')
    outputs = bert_model(**inputs)
    embedding = outputs[0][:, 0, :]

    return embedding.detach().cpu().numpy().squeeze()


def get_df_with_embeddings_xlmr(df_overall, answer_col, model_name='xlm-roberta-base', use_umap=False):

    df_overall = deepcopy(df_overall)
    tokenizer = XLMRobertaTokenizer.from_pretrained(model_name)
    bert_model = XLMRobertaModel.from_pretrained(model_name)
    # bert_model = XLMRobertaForSequenceClassification.from_pretrained(model_name)
    bert_model.to('cuda')
    bert_model.eval()

    df_overall['embedding_xlmr'] = df_overall[answer_col].apply(get_xlmr_embedding, args=(answer_col, tokenizer, bert_model))
    embeddings = list(df_overall['embedding_xlmr'])

    # embeddings = []

    # for idx, row in tqdm(df_overall.iterrows(), total=len(df_overall)):

    #     answer = row[answer_col]
    #     inputs = tokenizer(answer, return_tensors='pt', padding=True, truncation=True)
    #     outputs = bert_model.roberta(**inputs)
    #     embedding = outputs[0][:, 0, :]
    #     embeddings.append(embedding.detach().numpy().squeeze())

    # df_overall['embedding_xmlr'] = embeddings
    embeddings = np.array(embeddings)
    # print(embeddings.shape)

    if use_umap:

        reducer = umap.UMAP()
        df_overall['x'], df_overall['y'] = zip(*reducer.fit_transform(np.array(embeddings)))
    
    else:

        df_overall['x'], df_overall['y'] = zip(*TSNE().fit_transform(np.array(embeddings)))

    return df_overall


def get_df_with_embeddings_sbert(df_overall, answer_col, model_name='paraphrase-multilingual-MiniLM-L12-v2', use_umap=False):


    df_overall = deepcopy(df_overall)
    model = SentenceTransformer(model_name)
    sbert_embeddings = model.encode(list(df_overall[answer_col]))
    # print(sbert_embeddings.shape)
    df_overall['sbert_embedding'] = list(sbert_embeddings)

    if use_umap:

        reducer = umap.UMAP()
        df_overall['x'], df_overall['y'] = zip(*reducer.fit_transform(np.array(sbert_embeddings)))
    
    else:

        df_overall['x'], df_overall['y'] = zip(*TSNE().fit_transform(sbert_embeddings))

    return df_overall


def plot_embeddings(prompt, dataset_path, dataset_name, answer_column, target_column, language_column, languages, target_path='/results/emb_vis', use_umap=False):

    target_path = os.path.join(target_path, dataset_name, prompt)

    if not os.path.exists(target_path):

        os.makedirs(target_path)

    dfs = []

    for lang in languages:

        df_test = read_data(os.path.join(dataset_path, prompt, lang, 'test.csv'), answer_column=answer_column, target_column=target_column)

        if language_column is None:

            df_test['Language'] = lang

        dfs.append(df_test)
    
    df_overall = pd.concat(dfs).reset_index()
    df_overall_sbert = get_df_with_embeddings_sbert(df_overall=df_overall, answer_col=answer_column, use_umap=use_umap)
    df_overall_xlmr = get_df_with_embeddings_xlmr(df_overall=df_overall, answer_col=answer_column, use_umap=use_umap)

    if language_column is None:

        language_column = 'Language'

    extra_suffix = ''

    if use_umap:

        extra_suffix = '_umap'

    for legend in [True, False]:

        plot_embeddings_scatterplot(df_overall=df_overall_sbert, target_path=target_path, vis_column=language_column, model_name='SBERT', prompt_name=prompt, suffix='_lang' + extra_suffix, legend=legend)
        plot_embeddings_scatterplot(df_overall=df_overall_xlmr, target_path=target_path, vis_column=language_column, model_name='XLMR', prompt_name=prompt, suffix='_lang' + extra_suffix, legend=legend)

        plot_embeddings_scatterplot(df_overall=df_overall_sbert, target_path=target_path, vis_column=target_column, model_name='SBERT', prompt_name=prompt, suffix='_score' + extra_suffix, legend=legend)
        plot_embeddings_scatterplot(df_overall=df_overall_xlmr, target_path=target_path, vis_column=target_column, model_name='XLMR', prompt_name=prompt, suffix='_score' + extra_suffix, legend=legend)


# for dataset in [ASAP_T]:
for dataset in [EPIRLS, ASAP_T]:

    for prompt in os.listdir(dataset['dataset_path']):

        print(prompt)

        if prompt == '1':

            plot_embeddings(target_path='/results/emb_vis_final_', dataset_path=dataset['dataset_path'], dataset_name=dataset['dataset_name'], prompt=prompt, answer_column=dataset['answer_column'], target_column=dataset['target_column'], language_column=dataset['language_column'], use_umap=False, languages=dataset['languages'])
