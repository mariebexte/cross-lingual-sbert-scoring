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


## Visualize embedding space

def plot_embeddings_scatterplot(df_overall, target_path, vis_column, model_name, prompt_name, suffix):

    fig = sns.scatterplot(data=df_overall, x="x", y="y", hue=vis_column)
    sns.move_legend(fig, "upper left", bbox_to_anchor=(1, 1))

    plt.title(model_name + '(' + prompt_name + ')')

    plt.rcParams['savefig.dpi'] = 500
    plt.tight_layout()
    plt.savefig(os.path.join(target_path, model_name + suffix + ".pdf"), transparent=True)

    plt.clf()
    plt.cla()
    plt.close()


def get_xlmr_embedding(answer, answer_col, tokenizer, bert_model):

    inputs = tokenizer(answer, return_tensors='pt', padding=True, truncation=True)
    outputs = bert_model(**inputs)
    embedding = outputs[0][:, 0, :]
    return embedding.detach().numpy().squeeze()


def get_df_with_embeddings_xlmr(df_overall, answer_col, model_name='xlm-roberta-base', use_umap=False):

    df_overall = deepcopy(df_overall)
    tokenizer = XLMRobertaTokenizer.from_pretrained(model_name)
    bert_model = XLMRobertaModel.from_pretrained(model_name)
    # bert_model = XLMRobertaForSequenceClassification.from_pretrained(model_name)
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

    if umap:

        reducer = umap.UMAP()
        df_overall['x'], df_overall['y'] = zip(*reducer.fit_transform(np.array(embeddings)))
    
    else:

        df_overall['x'], df_overall['y'] = zip(*TSNE(n_components=2).fit_transform(np.array(embeddings)))

    return df_overall


def get_df_with_embeddings_sbert(df_overall, answer_col, model_name='paraphrase-multilingual-MiniLM-L12-v2', use_umap=False):

    df_overall = deepcopy(df_overall)
    model = SentenceTransformer(model_name)
    sbert_embeddings = model.encode(list(df_overall[answer_col]))
    # print(sbert_embeddings.shape)
    df_overall['sbert_embedding'] = list(sbert_embeddings)

    if umap:

        reducer = umap.UMAP()
        df_overall['x'], df_overall['y'] = zip(*reducer.fit_transform(np.array(sbert_embeddings)))
    
    else:

        df_overall['x'], df_overall['y'] = zip(*TSNE(n_components=2).fit_transform(sbert_embeddings))

    return df_overall


def plot_embeddings(prompt, dataset_path, dataset_name, answer_column, target_column, language_column, target_path='/results/emb_vis', use_umap=False):

    target_path = os.path.join(target_path, dataset_name, prompt)

    if not os.path.exists(target_path):

        os.makedirs(target_path)

    dfs = []

    for lang in ['ar', 'da', 'en', 'he', 'it', 'ka', 'nb', 'pt', 'sl', 'sv', 'zh']:

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

    plot_embeddings_scatterplot(df_overall=df_overall_sbert, target_path=target_path, vis_column=language_column, model_name='SBERT', prompt_name=prompt, suffix='_lang' + extra_suffix)
    plot_embeddings_scatterplot(df_overall=df_overall_xlmr, target_path=target_path, vis_column=language_column, model_name='XLMR', prompt_name=prompt, suffix='_lang' + extra_suffix)

    plot_embeddings_scatterplot(df_overall=df_overall_sbert, target_path=target_path, vis_column=target_column, model_name='SBERT', prompt_name=prompt, suffix='_score' + extra_suffix)
    plot_embeddings_scatterplot(df_overall=df_overall_xlmr, target_path=target_path, vis_column=target_column, model_name='XLMR', prompt_name=prompt, suffix='_score' + extra_suffix)


for dataset in [EPIRLS, ASAP_T]:

    for prompt in os.listdir(dataset['dataset_path']):

        print(prompt)

        plot_embeddings(dataset_path=dataset['dataset_path'], dataset_name=dataset['dataset_name'], prompt=prompt, answer_column=dataset['answer_column'], target_column=dataset['target_column'], language_column=dataset['language_column'], use_umap=True)
