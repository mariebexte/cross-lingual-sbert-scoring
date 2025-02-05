import os
import sys

import numpy as np
import pandas as pd

from config import EPIRLS, ASAP_T, ASAP_M
from heatmap import plot_heat
from sklearn.metrics import accuracy_score, cohen_kappa_score
from utils import average_qwk, average_runs_exp3
from pathlib import Path

import matplotlib.pyplot as plt


## Build dataframe that aggregates results from entire directory
## Columns: Prompt, train_lang, test_lang, model, acc, qwk

def aggregate_results(result_dir, prompts, target_column, languages):

    results = {}
    results_idx = 0

    for prompt in prompts:
        
        if os.path.isdir(os.path.join(result_dir, prompt)):

            for test_lang in languages:

                # for model in os.listdir(os.path.join(result_dir, prompt, train_lang))
                for model in [('XLMR', ''), ('SBERT', '_avg'), ('SBERT', '_max'), ('XLMR_SBERTcore', ''), ('SBERT_XLMRcore', '_avg'), ('SBERT_XLMRcore', '_max'), ('NPCR_XLMR', ''), ('NPCR_SBERT', ''), ('pretrained', '_avg'), ('pretrained', '_max')]:

                    # try:

                    df_preds = pd.read_csv(os.path.join(result_dir, prompt, test_lang, model[0], 'preds.csv')) 

                    gold=list(df_preds[target_column])
                    pred=list(df_preds['pred'+model[1]])

                    acc = accuracy_score(y_true=gold, y_pred=pred)
                    qwk = cohen_kappa_score(y1=gold, y2=pred, weights='quadratic')

                    results[results_idx] = {
                        'prompt': prompt,
                        'test_lang': test_lang,
                        'model': model[0] + model[1],
                        'acc': acc,
                        'qwk': qwk
                    }

                    results_idx += 1

                    if qwk < .05:

                        print('Concerningly low qwk', qwk, prompt, model, test_lang)
                    
                    # except:

                    #     print('MISSING RESULTS', prompt, model, test_lang, result_dir)

        else:

            print('MISSING PROMPT', prompt, result_dir)
            sys.exit(0)


    df_results = pd.DataFrame.from_dict(results, orient='index')
    df_results.to_csv(os.path.join(result_dir, 'overall.csv'))


def aggregate_results_cv(result_dir, prompts, target_column, languages, num_folds):

    results = {}
    results_idx = 0

    for prompt in prompts:
        
        if os.path.isdir(os.path.join(result_dir, prompt)):
            
            print(prompt)

            for test_lang in languages:

                # for model in os.listdir(os.path.join(result_dir, prompt, train_lang))
                for model in [('XLMR', ''), ('SBERT', '_avg'), ('SBERT', '_max'), ('XLMR_SBERTcore', ''), ('SBERT_XLMRcore', '_avg'), ('SBERT_XLMRcore', '_max'), ('NPCR_XLMR', ''), ('NPCR_SBERT', ''), ('pretrained', '_avg'), ('pretrained', '_max')]:

                    # try:

                    df_preds_list = []

                    for fold in range(1, num_folds+1):

                        df_preds_list.append(pd.read_csv(os.path.join(result_dir, prompt, test_lang, model[0], 'fold_'+str(fold), 'preds.csv')))
                    
                    df_preds = pd.concat(df_preds_list)

                    gold=list(df_preds[target_column])
                    pred=list(df_preds['pred'+model[1]])

                    acc = accuracy_score(y_true=gold, y_pred=pred)
                    qwk = cohen_kappa_score(y1=gold, y2=pred, weights='quadratic')

                    results[results_idx] = {
                        'prompt': prompt,
                        'test_lang': test_lang,
                        'model': model[0] + model[1],
                        'acc': acc,
                        'qwk': qwk
                    }

                    results_idx += 1

                    if qwk < .1:

                        print('Concerningly low qwk', qwk, prompt, model, test_lang)

                    # except:

                    #     print('MISSING', prompt, model, test_lang, result_dir)

        else:

            print('MISSING PROMPT', prompt, result_dir)
            sys.exit(0)


    df_results = pd.DataFrame.from_dict(results, orient='index')
    df_results.to_csv(os.path.join(result_dir, 'overall.csv'))


def split_results(overall_results_path):

    df_overall = pd.read_csv(overall_results_path)

    for model, df_model in df_overall.groupby('model'):

        dict_averaged = {}
        
        for test_lang, df_lang in df_model.groupby('test_lang'):

            dict_averaged[test_lang] = {'qwk': average_qwk(df_lang[['qwk']]), 'acc': df_lang['acc'].mean(), 'support': len(df_lang)}
    
        df_averaged = pd.DataFrame.from_dict(dict_averaged, orient='index')
        dict_averaged['avg'] = {'qwk': average_qwk(df_averaged[['qwk']]), 'acc': df_averaged['acc'].mean(), 'support': df_averaged['support'].mean()}

        df_averaged = pd.DataFrame.from_dict(dict_averaged, orient='index')
        df_averaged.index.name = 'test_lang'
        df_averaged.to_csv(os.path.join(Path(overall_results_path).parent.absolute(), model + '.csv'))


def get_cross_avg(row):

    row = row[[(('cross' in c) and not('translated' in c)) for c in row.index]]
    row = pd.DataFrame(row)
    row.columns = ['qwk']

    return average_qwk(row)


def get_translated_avg(row):

    row = row[['translated' in c for c in row.index]]
    row = pd.DataFrame(row)
    row.columns = ['qwk']

    return average_qwk(row)



# res_name = '/results/exp_3_lolo'
res_name = '/results/FINAL_PAPER/exp_3_lolo'

for dataset in [EPIRLS, ASAP_T]:

    for condition in ['combine_downsampled']:
        
#         for run in ['_RUN1', '_RUN2', '_RUN3']:

#             aggregate_results(result_dir=os.path.join(res_name + run, condition, dataset['dataset_name']), prompts=dataset['prompts'], target_column=dataset['target_column'], languages=dataset['languages'])
#             split_results(os.path.join(res_name + run, condition, dataset['dataset_name'], 'overall.csv'))

#         average_runs_exp3(result_file_list=[os.path.join(res_name+'_RUN1', condition, dataset['dataset_name'], 'overall.csv'),
#         os.path.join(res_name+'_RUN2', condition, dataset['dataset_name'], 'overall.csv'),
#         os.path.join(res_name+'_RUN3', condition, dataset['dataset_name'], 'overall.csv')],
#         target_folder=os.path.join(res_name+'_AVG', condition, dataset['dataset_name']))
        split_results(os.path.join(res_name + '_AVG', condition, dataset['dataset_name'], 'overall.csv'))


for dataset in [ASAP_M]:

    for condition in ['combine_downsampled', 'combine_downsampled_translated']:

        # for run in ['_RUN1', '_RUN2', '_RUN3']:

        #     aggregate_results_cv(result_dir=os.path.join(res_name + run, condition, dataset['dataset_name']), prompts=dataset['prompts'], target_column=dataset['target_column'], languages=dataset['languages'], num_folds=dataset['num_folds'])
        #     split_results(os.path.join(res_name + run, condition, dataset['dataset_name'], 'overall.csv'))
            
        # average_runs_exp3(result_file_list=[os.path.join(res_name+'_RUN1', condition, dataset['dataset_name'], 'overall.csv'),
        # os.path.join(res_name+'_RUN2', condition, dataset['dataset_name'], 'overall.csv'),
        # os.path.join(res_name+'_RUN3', condition, dataset['dataset_name'], 'overall.csv')], target_folder=os.path.join(res_name+'_AVG',  condition, dataset['dataset_name']))
        split_results(os.path.join(res_name + '_AVG', condition, dataset['dataset_name'], 'overall.csv'))
