import pandas as pd
import numpy as np
import os, sys
from sklearn.metrics import accuracy_score, cohen_kappa_score
from heatmap import plot_heat

## Build dataframe that aggregates results from entire directory
## Columns: Prompt, train_lang, test_lang, model, acc, qwk

def aggregate_results(result_dir, languages=['ar', 'da', 'en', 'he', 'it', 'ka', 'nb', 'pt', 'sv', 'sl', 'zh']):

    results = {}
    results_idx = 0

    for prompt in os.listdir(result_dir):
        if os.path.isdir(os.path.join(result_dir, prompt)):

            # for train_lang in os.listdir(os.path.join(result_dir, prompt)):
            for train_lang in languages:

                for model in os.listdir(os.path.join(result_dir, prompt, train_lang)):
                # for model in ['SBERT', 'MBERT']:

                    for test_lang in languages:

                        print(prompt, model, train_lang, test_lang)

                        df_preds = pd.read_csv(os.path.join(result_dir, prompt, train_lang, model, test_lang, 'preds.csv')) 
                        gold=list(df_preds['score'])
                        try:
                            pred=list(df_preds['pred'])
                        except:
                            pred=list(df_preds['pred_avg'])

                        acc = accuracy_score(y_true=gold, y_pred=pred)
                        qwk = cohen_kappa_score(y1=gold, y2=pred, weights='quadratic')

                        acc_within_val = -1
                        qwk_within_val = -1
                    
                        if model == 'SBERT':

                            test_lang_val_within = test_lang + '_target_val'

                            print("SBERT CONDITION", prompt, model, train_lang, test_lang_val_within)

                            df_preds = pd.read_csv(os.path.join(result_dir, prompt, train_lang, model, test_lang_val_within, 'preds.csv')) 
                            gold=list(df_preds['score'])
                            try:
                                pred=list(df_preds['pred'])
                            except:
                                pred=list(df_preds['pred_avg'])

                            acc_within_val = accuracy_score(y_true=gold, y_pred=pred)
                            qwk_within_val = cohen_kappa_score(y1=gold, y2=pred, weights='quadratic')

                        results[results_idx] = {
                            'prompt': prompt,
                            'train_lang': train_lang,
                            'test_lang': test_lang,
                            'model': model,
                            'acc': acc,
                            'qwk': qwk,
                            'qwk_within_val': qwk_within_val,
                            'acc_within_val': acc_within_val
                        }

                        results_idx += 1


    df_results = pd.DataFrame.from_dict(results, orient='index')
    df_results.to_csv(os.path.join(result_dir, 'overall.csv'))


def average_qwk(df):

    high = 0.999
    try:
        df['qwk_smooth'] = df['qwk'].apply(lambda x: x if x < high else high)
    except:
        df['qwk_smooth'] = df['qwk_within_val'].apply(lambda x: x if x < high else high)
    # Arctanh == FISHER
    df_preds_fisher = np.arctanh(df)
    print(df_preds_fisher)
    test_scores_mean_fisher = np.nanmean(df_preds_fisher, axis=0)
    # Tanh == FISHERINV
    test_scores_mean = np.tanh(test_scores_mean_fisher)
    return test_scores_mean


def calculate_model_matrixes(result_df_path):

    dir_for_results = os.path.dirname(result_df_path)

    df_results = pd.read_csv(result_df_path)

    # For each model, calculate a cross-matrix and its support
    for model, df_model in df_results.groupby('model'):

        aggregated_dict = {}
        dict_idx = 0

        for lang_train, df_lang_train in df_model.groupby('train_lang'):

            for lang_test, df_lang_test in df_lang_train.groupby('test_lang'):

                aggregated_dict[dict_idx] = {'train_lang': lang_train, 'test_lang': lang_test, 'support': len(df_lang_test), 'qwk_fisher': average_qwk(df_lang_test[['qwk']])[1], 'qwk': df_lang_test['qwk'].mean(), 'acc': df_lang_test['acc'].mean(),
                'acc_within_val': df_lang_test['acc_within_val'].mean(), 'qwk_within_val': df_lang_test['qwk_within_val'].mean(), 'qwk_within_val_fisher': average_qwk(df_lang_test[['qwk_within_val']])[1]}
                dict_idx += 1
        
        df_model = pd.DataFrame.from_dict(aggregated_dict, orient='index')

        df_support = pd.pivot_table(df_model, values='support', index=['train_lang'], columns=['test_lang'])
        df_qwk_fisher = pd.pivot_table(df_model, values='qwk_fisher', index=['train_lang'], columns=['test_lang'])
        df_qwk = pd.pivot_table(df_model, values='qwk', index=['train_lang'], columns=['test_lang'])
        df_acc = pd.pivot_table(df_model, values='acc', index=['train_lang'], columns=['test_lang'])
        df_qwk_fisher_within_val = pd.pivot_table(df_model, values='qwk_within_val_fisher', index=['train_lang'], columns=['test_lang'])
        df_qwk_within_val = pd.pivot_table(df_model, values='qwk_within_val', index=['train_lang'], columns=['test_lang'])
        df_acc_within_val = pd.pivot_table(df_model, values='acc_within_val', index=['train_lang'], columns=['test_lang'])
        
        df_support.to_csv(os.path.join(dir_for_results, model + '_support.csv'))
        df_qwk_fisher.to_csv(os.path.join(dir_for_results, model + '_qwk_fisher.csv'))
        df_qwk.to_csv(os.path.join(dir_for_results, model + '_qwk.csv'))
        df_acc.to_csv(os.path.join(dir_for_results, model + '_acc.csv'))
        df_qwk_fisher_within_val.to_csv(os.path.join(dir_for_results, model + '_qwk_fisher_within_val.csv'))
        df_qwk_within_val.to_csv(os.path.join(dir_for_results, model + '_qwk_within_val.csv'))
        df_acc_within_val.to_csv(os.path.join(dir_for_results, model + '_acc_within_val.csv'))

        plot_heat(df_matrix=df_qwk_fisher, target_path=dir_for_results, model=model, metric="qwk_fisher")
        plot_heat(df_matrix=df_qwk, target_path=dir_for_results, model=model, metric="qwk")
        plot_heat(df_matrix=df_acc, target_path=dir_for_results, model=model, metric="acc")
        plot_heat(df_matrix=df_qwk_fisher_within_val, target_path=dir_for_results, model=model, metric="qwk_fisher_within_val")
        plot_heat(df_matrix=df_qwk_within_val, target_path=dir_for_results, model=model, metric="qwk_within_val")
        plot_heat(df_matrix=df_acc_within_val, target_path=dir_for_results, model=model, metric="acc_within_val")


aggregate_results('/results/exp_1_zero_shot')
calculate_model_matrixes('/results/exp_1_zero_shot/overall.csv')

