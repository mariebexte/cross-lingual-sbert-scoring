import os
import pandas as pd
from sklearn.manifold import TSNE
from utils import read_data
from sentence_transformers import SentenceTransformer
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np
import sys
from transformers import XLMRobertaForSequenceClassification, XLMRobertaTokenizer
from copy import deepcopy
from tqdm import tqdm


## Visualize embedding space

def plot_embeddings_scatterplot(df_overall, target_path, model_name):

    fig = sns.scatterplot(data=df_overall, x="x", y="y", hue="Language", style='score')
    sns.move_legend(fig, "upper left", bbox_to_anchor=(1, 1))

    plt.rcParams['savefig.dpi'] = 500
    plt.tight_layout()
    plt.savefig(os.path.join(target_path, model_name + ".pdf"), transparent=True)

    plt.clf()
    plt.cla()
    plt.close()


def get_df_with_embeddings_xlmr(df_overall, model_name='xlm-roberta-base'):

    df_overall = deepcopy(df_overall)
    tokenizer = XLMRobertaTokenizer.from_pretrained('xlm-roberta-base')
    bert_model = XLMRobertaForSequenceClassification.from_pretrained('xlm-roberta-base')
    bert_model.eval()

    # inputs = tokenizer(list(df_overall['Value']), return_tensors='pt', padding=True, truncation=True)
    embeddings = []

    for idx, row in tqdm(df_overall.iterrows(), total=len(df_overall)):

        answer = row['Value']
        inputs = tokenizer(answer, return_tensors='pt', padding=True, truncation=True)
        outputs = bert_model.roberta(**inputs)
        embedding = outputs[0][:, 0, :]
        # print(type(embedding.detach()))
        embeddings.append(embedding.detach().numpy().squeeze())
        # print(embedding.detach().numpy().squeeze().shape)

    # inputs = tokenizer("Hello, my dog is cute", return_tensors='pt', padding=True, truncation=True)
    # print(inputs)
    # sys.exit(0)
    # outputs = bert_model.roberta(**inputs, output_hidden_states=True)
    # # print(outputs[0].shape)
    # # print(outputs[0])
    # embedding = outputs[0][:, 0, :]
    # print(embedding)
    # print(embedding.shape)
    # sys.exit(0)

    df_overall['embedding_xmlr'] = embeddings
    embeddings = np.array(embeddings)
    print(embeddings.shape)
    df_overall['x'], df_overall['y'] = zip(*TSNE(n_components=2).fit_transform(np.array(embeddings)))

    return df_overall


def get_df_with_embeddings_sbert(df_overall, model_name='paraphrase-multilingual-MiniLM-L12-v2'):

    df_overall = deepcopy(df_overall)
    model = SentenceTransformer(model_name)
    sbert_embeddings = model.encode(list(df_overall['Value']))
    print(sbert_embeddings.shape)
    df_overall['sbert_embedding'] = list(sbert_embeddings)
    df_overall['x'], df_overall['y'] = zip(*TSNE(n_components=2).fit_transform(sbert_embeddings))
    return df_overall


def plot_embeddings(prompt, data_path='/data/exp', answer_col='', target_path='/results/emb_vis'):

    target_path = os.path.join(target_path, prompt)

    if not os.path.exists(target_path):

        os.makedirs(target_path)

    dfs = []

    for lang in ['ar', 'da', 'en', 'he', 'it', 'ka', 'nb', 'pt', 'sl', 'sv', 'zh']:

        df_test = read_data(os.path.join(data_path, prompt, lang, 'test.csv'))
        dfs.append(df_test)
    
    df_overall = pd.concat(dfs).reset_index()
    df_overall_sbert = get_df_with_embeddings_sbert(df_overall=df_overall)
    df_overall_xlmr = get_df_with_embeddings_xlmr(df_overall=df_overall)

    plot_embeddings_scatterplot(df_overall=df_overall_sbert, target_path=target_path, model_name='SBERT')
    plot_embeddings_scatterplot(df_overall=df_overall_xlmr, target_path=target_path, model_name='XLMR')


for prompt in os.listdir('/data/exp'):

    plot_embeddings(prompt)
