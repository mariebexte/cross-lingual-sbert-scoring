import pandas as pd
import os
from sklearn.metrics import accuracy_score, cohen_kappa_score

## Build dataframe that aggregates results from entire directory
## Columns: Prompt, train_lang, test_lang, model, acc, qwk

def aggregate_results(result_dir):

    results = {}
    results_idx = 0

    for prompt in os.listdir(result_dir):
        if os.path.isdir(os.path.join(result_dir, prompt)):

            for target_lang in os.listdir(os.path.join(result_dir, prompt)):

                for model in os.listdir(os.path.join(result_dir, prompt, target_lang)):

                    df_preds = pd.read_csv(os.path.join(result_dir, prompt, target_lang, model, 'preds.csv')) 
                    gold=list(df_preds['score'])
                    pred=list(df_preds['pred'])

                    acc = accuracy_score(y_true=gold, y_pred=pred)
                    qwk = cohen_kappa_score(y1=gold, y2=pred, weights='quadratic')

                    results[results_idx] = {
                        'prompt': prompt,
                        'lang': target_lang,
                        'model': model,
                        'acc': acc,
                        'QWK': qwk
                    }

                    results_idx += 1

    df_results = pd.DataFrame.from_dict(results, orient='index')
    df_results.to_csv(os.path.join(result_dir, 'overall.csv'))

    for model, df_model in df_results.groupby('model'):

        print('model\t' + model)
        print('acc\t' + str(df_model['acc'].mean()))
        print('QWK\t' + str(df_model['QWK'].mean()))


aggregate_results('/results/results_dev/dev')

