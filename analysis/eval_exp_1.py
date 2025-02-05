import os
import sys

import numpy as np
import pandas as pd

from config import EPIRLS, ASAP_T, ASAP_M
from heatmap import plot_heat
from sklearn.metrics import accuracy_score, cohen_kappa_score
from utils import average_qwk, average_runs_exp1

import matplotlib.pyplot as plt


## Build dataframe that aggregates results from entire directory
## Columns: Prompt, train_lang, test_lang, model, acc, qwk

def aggregate_results(result_dir, target_column, languages, translate_test):

    results = {}
    results_idx = 0

    for prompt in os.listdir(result_dir):
        
        if os.path.isdir(os.path.join(result_dir, prompt)):
            
            print(prompt)

            for train_lang in languages:

                # for model in os.listdir(os.path.join(result_dir, prompt, train_lang))
                for model in [('XLMR', ''), ('SBERT', '_avg'), ('SBERT', '_max'), ('XLMR_SBERTcore', ''), ('SBERT_XLMRcore', '_avg'), ('SBERT_XLMRcore', '_max'), ('NPCR_XLMR', ''), ('NPCR_SBERT', ''), ('pretrained', '_avg'), ('pretrained', '_max')]:

                    for test_lang in languages:

                        try:

                            df_preds = pd.read_csv(os.path.join(result_dir, prompt, train_lang, model[0], test_lang, 'preds.csv')) 

                            gold=list(df_preds[target_column])
                            pred=list(df_preds['pred'+model[1]])

                            acc = accuracy_score(y_true=gold, y_pred=pred)
                            qwk = cohen_kappa_score(y1=gold, y2=pred, weights='quadratic')

                            results[results_idx] = {
                                'prompt': prompt,
                                'train_lang': train_lang,
                                'test_lang': test_lang,
                                'model': model[0] + model[1],
                                'acc': acc,
                                'qwk': qwk
                            }

                            results_idx += 1

                            if qwk < .1 and train_lang == test_lang:

                                print('Concerningly low qwk', qwk, prompt, model, train_lang, test_lang)
                        
                        except:
                            # print('MISSING', prompt, model, train_lang, test_lang)
                            pass

                        if translate_test:

                            try:

                                df_preds = pd.read_csv(os.path.join(result_dir, prompt, train_lang, model[0], test_lang+'_translated', 'preds.csv')) 

                                gold=list(df_preds[target_column])
                                pred=list(df_preds['pred'+model[1]])

                                acc = accuracy_score(y_true=gold, y_pred=pred)
                                qwk = cohen_kappa_score(y1=gold, y2=pred, weights='quadratic')

                                results[results_idx] = {
                                    'prompt': prompt,
                                    'train_lang': train_lang,
                                    'test_lang': test_lang+'_translated',
                                    'model': model[0] + model[1],
                                    'acc': acc,
                                    'qwk': qwk
                                }

                                results_idx += 1

                                if qwk < .1:

                                    print('Concerningly low qwk', qwk, prompt, model, train_lang, test_lang + '_translated')
                            
                            except:
                                # print('MISSING', prompt, model, train_lang, test_lang)
                                pass


    df_results = pd.DataFrame.from_dict(results, orient='index')
    df_results.to_csv(os.path.join(result_dir, 'overall.csv'))


def aggregate_results_cv(result_dir, target_column, languages, num_folds, translate_test):

    results = {}
    results_idx = 0

    for prompt in os.listdir(result_dir):
        
        if os.path.isdir(os.path.join(result_dir, prompt)):
            
            print(prompt)

            for train_lang in languages:

                # for model in os.listdir(os.path.join(result_dir, prompt, train_lang))
                for model in [('XLMR', ''), ('SBERT', '_avg'), ('SBERT', '_max'), ('XLMR_SBERTcore', ''), ('SBERT_XLMRcore', '_avg'), ('SBERT_XLMRcore', '_max'), ('NPCR_XLMR', ''), ('NPCR_SBERT', ''), ('pretrained', '_avg'), ('pretrained', '_max')]:

                    for test_lang in languages:

                        try:

                            df_preds_list = []

                            for fold in range(1, num_folds+1):

                                df_preds_list.append(pd.read_csv(os.path.join(result_dir, prompt, train_lang, model[0], 'fold_'+str(fold), test_lang, 'preds.csv')))

                            # TODO: Perform voting if there are multiple predictions
                            df_preds = pd.concat(df_preds_list)

                            gold=list(df_preds[target_column])
                            pred=list(df_preds['pred'+model[1]])

                            acc = accuracy_score(y_true=gold, y_pred=pred)
                            qwk = cohen_kappa_score(y1=gold, y2=pred, weights='quadratic')

                            results[results_idx] = {
                                'prompt': prompt,
                                'train_lang': train_lang,
                                'test_lang': test_lang,
                                'model': model[0] + model[1],
                                'acc': acc,
                                'qwk': qwk
                            }

                            results_idx += 1

                            if qwk < .1 and train_lang == test_lang:

                                print('Concerningly low qwk', qwk, prompt, model, train_lang, test_lang)

                        except:

                            # print('MISSING', prompt, model, train_lang, test_lang)
                            pass

                        if translate_test:

                            try:

                                df_preds_list = []

                                for fold in range(1, num_folds+1):

                                    df_preds_list.append(pd.read_csv(os.path.join(result_dir, prompt, train_lang, model[0], 'fold_'+str(fold), test_lang+'_translated', 'preds.csv')))
                                
                                df_preds = pd.concat(df_preds_list)

                                gold=list(df_preds[target_column])
                                pred=list(df_preds['pred'+model[1]])

                                acc = accuracy_score(y_true=gold, y_pred=pred)
                                qwk = cohen_kappa_score(y1=gold, y2=pred, weights='quadratic')

                                results[results_idx] = {
                                    'prompt': prompt,
                                    'train_lang': train_lang,
                                    'test_lang': test_lang+'_translated',
                                    'model': model[0] + model[1],
                                    'acc': acc,
                                    'qwk': qwk
                                }

                                results_idx += 1

                                if qwk < .1:

                                    print('Concerningly low qwk', qwk, prompt, model, train_lang, test_lang + '_translated')
                            
                            except:

                                # print('MISSING', prompt, model, train_lang, test_lang)
                                pass

    df_results = pd.DataFrame.from_dict(results, orient='index')
    df_results.to_csv(os.path.join(result_dir, 'overall.csv'))


def get_cross_avg(row, languages):

    row = row[['cross_' + lang + '_avg' for lang in languages]]
    row = pd.DataFrame(row)
    row.columns = ['qwk']

    return average_qwk(row)


def get_best_cross_avg(row, languages):

    row = row[['cross_' + lang + '_best' for lang in languages]]
    # row = row[[(('cross' in c) and ('best' in c)) and not ('translated' in c) for c in row.index]]
    row = pd.DataFrame(row)
    row.columns = ['qwk']

    return average_qwk(row)


def get_translated_avg(row, languages):

    row = row[['cross_' + lang + '_translated_avg' for lang in languages]]
    row = pd.DataFrame(row)
    row.columns = ['qwk']

    return average_qwk(row)


def calculate_model_matrixes(result_df_path, lang_order, translate_test=False):

    dir_for_results = os.path.dirname(result_df_path)

    df_results = pd.read_csv(result_df_path)

    dict_averaged = {}

    # For each model, calculate a cross-matrix and its support
    for model, df_model in df_results.groupby('model'):

        # Plot within language performances for each prompt
        df_results_within = df_model[df_model['train_lang'] == df_model['test_lang']]

        plt.rcParams['figure.figsize'] = 0.2*len(df_results_within),2
        plt.xticks(fontsize=3)
        plt.rcParams['savefig.dpi'] = 200

        df_results_within = df_results_within.sort_values(by=['qwk'])
        df_results_within['prompt_lang'] = df_results_within['prompt'].astype(str) + df_results_within['train_lang']
        df_results_within.plot.bar(x='prompt_lang', y='qwk', width=0.7, legend=True)

        plt.savefig(os.path.join(dir_for_results, model + '_within_performance.png'), transparent=True, bbox_inches="tight")
        plt.clf()
        plt.cla()
        plt.close()

        aggregated_dict = {}
        dict_idx = 0

        for lang_train, df_lang_train in df_model.groupby('train_lang'):

            for lang_test, df_lang_test in df_lang_train.groupby('test_lang'):

                aggregated_dict[dict_idx] = {'train_lang': lang_train, 'test_lang': lang_test, 'support': len(df_lang_test), 'qwk_fisher': average_qwk(df_lang_test[['qwk']]), 'acc': df_lang_test['acc'].mean()}
                dict_idx += 1
        
        df_model = pd.DataFrame.from_dict(aggregated_dict, orient='index')

        df_support = pd.pivot_table(df_model, values='support', index=['train_lang'], columns=['test_lang'])
        df_qwk_fisher = pd.pivot_table(df_model, values='qwk_fisher', index=['train_lang'], columns=['test_lang'])
        df_acc = pd.pivot_table(df_model, values='acc', index=['train_lang'], columns=['test_lang'])
        
        df_support.to_csv(os.path.join(dir_for_results, model + '_support.csv'))
        df_qwk_fisher.to_csv(os.path.join(dir_for_results, model + '_qwk_fisher.csv'))
        df_acc.to_csv(os.path.join(dir_for_results, model + '_acc.csv'))

        df_qwk_fisher.index.names=['training language']
        df_qwk_fisher.columns.names=['test language']

        plot_heat(df_matrix=df_acc, target_path=dir_for_results, model=model, metric="acc", lang_order=lang_order, translated=False)
        plot_heat(df_matrix=df_qwk_fisher, target_path=dir_for_results, model=model, metric="qwk_fisher", lang_order=lang_order, translated=False)

        if translate_test:

            lang_order_translated = [lang + '_translated' for lang in lang_order]
            df_acc_translated = df_acc[lang_order_translated]
            df_qwk_fisher_translated = df_qwk_fisher[lang_order_translated]

            df_acc_translated.columns = lang_order
            df_qwk_fisher_translated.columns = lang_order

            # Add within language performance to main diagonal
            for lang in lang_order:

                df_acc_translated.loc[lang, lang] = df_acc.loc[lang, lang]
                df_qwk_fisher_translated.loc[lang, lang] = df_qwk_fisher.loc[lang, lang]

            df_acc_translated.columns.names=['test language']
            df_qwk_fisher_translated.columns.names=['test language']
            plot_heat(df_matrix=df_acc_translated, target_path=dir_for_results, model=model, metric="acc", lang_order=lang_order, translated=True)
            plot_heat(df_matrix=df_qwk_fisher_translated, target_path=dir_for_results, model=model, metric="qwk_fisher", lang_order=lang_order, translated=True)


        df_model['condition'] = df_model.apply(lambda row: 'cross_' + row['test_lang'] if row['train_lang'] != row['test_lang'] else 'within', axis=1)
        model_dict = {}

        for condition, df_condition in df_model.groupby('condition'):

            model_dict[condition + '_avg'] = average_qwk(df_condition, qwk_col='qwk_fisher')

            if condition == 'within':

                for _, row in df_condition.iterrows():

                    model_dict['within_' + row['test_lang']] = row['qwk_fisher']
            
            # Is cross: Get result for best cross training language
            else:

                if not 'translated' in condition:
                
                    model_dict[condition + '_best'] = df_condition['qwk_fisher'].max()

        dict_averaged[model] = model_dict
    
    df_averaged = pd.DataFrame.from_dict(dict_averaged, orient='index')
    df_averaged['cross_avg'] = df_averaged.apply(get_cross_avg, axis=1, args=(lang_order,))
    df_averaged['best_cross_avg'] = df_averaged.apply(get_best_cross_avg, axis=1, args=(lang_order,))
    
    has_translated = False

    for col in df_averaged.columns:

        if 'translated' in col:

            has_translated = True
    
    if has_translated:

        df_averaged['cross_translated_avg'] = df_averaged.apply(get_translated_avg, axis=1, args=(lang_order,))

    df_averaged.to_csv(os.path.join(dir_for_results, 'averages.csv'))


res_name = '/results/FINAL_PAPER/exp_1_zero_shot'

for dataset in [EPIRLS, ASAP_T]:

    # for run in ['_RUN1', '_RUN2', '_RUN3']:

    #     aggregate_results(result_dir=os.path.join(res_name + run, dataset['dataset_name']), target_column=dataset['target_column'], languages=dataset['languages'], translate_test=dataset['translate_test'])
    #     calculate_model_matrixes(os.path.join(res_name + run, dataset['dataset_name'], 'overall.csv'), lang_order=dataset['languages'])

    # average_runs_exp1(result_file_list=[os.path.join(res_name+'_RUN1', dataset['dataset_name'], 'overall.csv'),
    # os.path.join(res_name+'_RUN2', dataset['dataset_name'], 'overall.csv'),
    # os.path.join(res_name+'_RUN3', dataset['dataset_name'], 'overall.csv')], target_folder=os.path.join(res_name+'_AVG', dataset['dataset_name']))

    calculate_model_matrixes(os.path.join(res_name+'_AVG', dataset['dataset_name'], 'overall.csv'), lang_order=dataset['languages'], translate_test=dataset['translate_test'])


for dataset in [ASAP_M]:

    # for run in ['_RUN1', '_RUN2', '_RUN3']:

    #     aggregate_results_cv(result_dir=os.path.join(res_name + run, dataset['dataset_name']), target_column=dataset['target_column'], languages=dataset['languages'], num_folds=dataset['num_folds'], translate_test=dataset['translate_test'])
    #     calculate_model_matrixes(os.path.join(res_name + run, dataset['dataset_name'], 'overall.csv'), lang_order=dataset['languages'])

    # average_runs_exp1(result_file_list=[os.path.join(res_name+'_RUN1', dataset['dataset_name'], 'overall.csv'),
    # os.path.join(res_name+'_RUN2', dataset['dataset_name'], 'overall.csv'),
    # os.path.join(res_name+'_RUN3', dataset['dataset_name'], 'overall.csv')], target_folder=os.path.join(res_name+'_AVG', dataset['dataset_name']))

    calculate_model_matrixes(os.path.join(res_name+'_AVG', dataset['dataset_name'], 'overall.csv'), lang_order=dataset['languages'], translate_test=dataset['translate_test'])
