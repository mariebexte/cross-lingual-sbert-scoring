import os
import sys

import numpy as np
import pandas as pd

from copy import deepcopy


def read_data(path, answer_column, target_column):

    df = pd.read_csv(path)
    df = df.fillna('')
    df[answer_column] = df[answer_column].astype(str)
    df[target_column] = df[target_column].astype(int)

    return df


def average_qwk(df, qwk_col='qwk'):

    df = deepcopy(df)

    high = 0.999

    df.loc[df[qwk_col] > high] = high 
    df = df[qwk_col]

    # Arctanh == FISHER
    df_preds_fisher = np.arctanh(df)
    # print(df_preds_fisher)
    test_scores_mean_fisher = np.nanmean(df_preds_fisher, axis=0)
    # Tanh == FISHERINV
    test_scores_mean = np.tanh(test_scores_mean_fisher)

    return test_scores_mean


def get_qwk_sd(df, qwk_col='qwk'):

    df = deepcopy(df)

    high = 0.999

    df.loc[df[qwk_col] > high] = high 
    df = df[qwk_col]

    # Arctanh == FISHER
    df_preds_fisher = np.arctanh(df)
    # print(df_preds_fisher)
    sd_fisher = np.std(df_preds_fisher)
    # Tanh == FISHERINV
    sd = np.tanh(sd_fisher)

    return sd


# ONLY processes those where all runs were finalized!
def average_runs_exp1(result_file_list, target_folder):

    overall_results_dict = {}
    overall_results_dict_idx = 0

    if not os.path.exists(target_folder):

        os.makedirs(target_folder)

    result_file_dfs = []
    
    for file in result_file_list:

        result_file_dfs.append(pd.read_csv(file))

    df_all_results = pd.concat(result_file_dfs)

    for prompt, df_prompt in df_all_results.groupby('prompt'):

        for train_lang, df_train_lang in df_prompt.groupby('train_lang'):

            for test_lang, df_test_lang in df_train_lang.groupby('test_lang'):

                for model, df_model in df_test_lang.groupby('model'):

                    # All runs were completed: Proceed with this data point
                    if len(df_model) == len(result_file_list):

                        overall_results_dict[overall_results_dict_idx] = {
                            'prompt': prompt,
                            'train_lang': train_lang,
                            'test_lang': test_lang,
                            'model': model,
                            'acc': df_model['acc'].mean(),
                            'acc_sd': df_model['acc'].std(),
                            'qwk': average_qwk(df_model[['qwk']]),
                            'qwk_sd': get_qwk_sd(df_model[['qwk']])
                        }

                        overall_results_dict_idx += 1
                
                    else:

                        print('Only', len(df_model), 'runs for', model, prompt, train_lang, test_lang)
                
    df_averaged_results = pd.DataFrame.from_dict(overall_results_dict, orient='index')
    df_averaged_results.to_csv(os.path.join(target_folder, 'overall.csv'))
    

# ONLY processes those where all runs were finalized!
def average_runs_exp3(result_file_list, target_folder):

    overall_results_dict = {}
    overall_results_dict_idx = 0

    if not os.path.exists(target_folder):

        os.makedirs(target_folder)

    result_file_dfs = []
    
    for file in result_file_list:

        result_file_dfs.append(pd.read_csv(file))

    df_all_results = pd.concat(result_file_dfs)

    for prompt, df_prompt in df_all_results.groupby('prompt'):

        for test_lang, df_test_lang in df_prompt.groupby('test_lang'):

            for model, df_model in df_test_lang.groupby('model'):

                # All runs were completed: Proceed with this data point
                if len(df_model) == len(result_file_list):

                    overall_results_dict[overall_results_dict_idx] = {
                        'prompt': prompt,
                        'test_lang': test_lang,
                        'model': model,
                        'acc': df_model['acc'].mean(),
                        'acc_sd': df_model['acc'].std(),
                        'qwk': average_qwk(df_model[['qwk']]),
                        'qwk_sd': get_qwk_sd(df_model[['qwk']])
                    }

                    overall_results_dict_idx += 1
            
                else:

                    print('Only', len(df_model), 'runs for', model, prompt, test_lang)
                
    df_averaged_results = pd.DataFrame.from_dict(overall_results_dict, orient='index')
    df_averaged_results.to_csv(os.path.join(target_folder, 'overall.csv'))
