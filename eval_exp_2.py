import os
import sys
import pandas as pd
from sklearn.metrics import cohen_kappa_score
from analysis.utils import average_qwk
from copy import deepcopy
from config import RESULT_PATH_EXP_1, RESULT_PATH_EXP_2

## Plot curves for tradeoff between base and target language

target_folder = 'curves'
languages = ['ar', 'da', 'en', 'he', 'it', 'ka', 'nb', 'pt', 'sl', 'sv', 'zh']
sizes = [0, 15, 35, 75, 150, 300, 600]


def eval(cross_results_path, results_path, update_results=True):

    if update_results:

        results = {}
        results_idx = 0

        for prompt in os.listdir(results_path):
        
            if (not '.' in prompt) and len(prompt)==8:

                # print('tradeoff', prompt)

                for target_language in languages:
                # for target_language in os.listdir(os.path.join(results_path, prompt)):

                    for model in [('XLMR', ''),('XLMR_SBERTcore', ''), ('SBERT', ''), ('SBERT', '_max')]:
                        if os.path.isdir(os.path.join(results_path, prompt, target_language, model[0])):

                            for base_language in languages:
                            # for base_language in os.listdir(os.path.join(results_path, prompt, target_language, model[0])):
                                if os.path.isdir(os.path.join(results_path, prompt, target_language, model[0], base_language)):

                                    for target_amount in sizes[1:-1]:
                                    # for target_amount in os.listdir(os.path.join(results_path, prompt, target_language, model[0], base_language)):

                                        try:
                                            print(os.path.join(results_path, prompt, target_language, model[0], base_language, str(target_amount), 'preds.csv'))
                                            preds = pd.read_csv(os.path.join(results_path, prompt, target_language, model[0], base_language, str(target_amount), 'preds.csv'))
                                            qwk = cohen_kappa_score(list(preds['score']), list(preds['pred'+model[1]]), weights='quadratic')

                                            results[results_idx] = {
                                                'prompt': prompt,
                                                'base_lang': base_language,
                                                'target_lang': target_language,
                                                'num_target': target_amount,
                                                'model': model[0] + model[1],
                                                'qwk': qwk,
                                            }

                                            results_idx += 1
                                        except:
                                            print('missing', prompt, model, base_language, target_language, target_amount)
                                            # sys.exit(0)

        # sys.exit(0)


        for prompt in os.listdir(cross_results_path):
        
            if not '.' in prompt:

                print(prompt)

                for training_language in languages:
                # for training_language in os.listdir(os.path.join(cross_results_path, prompt)):

                    for model in [('XLMR', ''),('XLMR_SBERTcore', ''), ('SBERT', ''), ('SBERT', '_max')]:
                        if os.path.isdir(os.path.join(cross_results_path, prompt, training_language, model[0])):

                            for test_language in languages:
                            # for test_language in os.listdir(os.path.join(cross_results_path, prompt, training_language, model[0])):
                                if os.path.isdir(os.path.join(cross_results_path, prompt, training_language, model[0], test_language)):

                                    preds = pd.read_csv(os.path.join(cross_results_path, prompt, training_language, model[0], test_language, 'preds.csv'))
                                    qwk = cohen_kappa_score(list(preds['score']), list(preds['pred'+model[1]]), weights='quadratic')

                                    # If we are training and testing on the same language, set number of target langauge answers in training accordingly
                                    num_target = 0
                                    if training_language == test_language:
                                        num_target = 600

                                    results[results_idx] = {
                                        'prompt': prompt,
                                        'base_lang': training_language,
                                        'target_lang': test_language,
                                        'num_target': num_target,
                                        'model': model[0] + model[1],
                                        'qwk': qwk,
                                    }

                                    results_idx += 1


        df_results = pd.DataFrame.from_dict(results, orient='index')
        df_results.to_csv(os.path.join(results_path, 'overall.csv'))
    
    else:
        df_results = pd.read_csv(os.path.join(results_path, 'overall.csv'))

    print(df_results)
    df_results = df_results[['target_lang', 'prompt', 'base_lang', 'model', 'num_target', 'qwk']]
    df_results = df_results.sort_values(['target_lang', 'prompt', 'base_lang', 'model', 'num_target'])


    if not os.path.exists(os.path.join(results_path, target_folder)):
        os.mkdir(os.path.join(results_path, target_folder))

    # Support df for overview of whether all results are there
    support = {}
    support_idx = 0

    for model in ['XLMR', 'SBERT_max', 'SBERT', 'XLMR_SBERTcore']:

        for target_language in languages:

            target_dict = {}

            df_of_interest = df_results.loc[(df_results['model'] == model) & (df_results['target_lang'] == target_language) & 
                                                    (df_results['base_lang'] == target_language) & (df_results['num_target'] == sizes[-1])]
        
            support[support_idx] = {
                        'model': model,
                        'target_lang': target_language,
                        'base_lang': target_language,
                        'num_target': sizes[-1],
                        'num_prompts': len(df_of_interest)
            }
            support_idx += 1

            ## Can put 34 here if we only want results for full support
            if len(df_of_interest) > 0:
                avg_within_performance=average_qwk(pd.DataFrame(df_of_interest['qwk']))
            else:
                avg_within_performance=-1
            # avg within performance is endpoint for all other languages
            
            other_languages = deepcopy(languages)
            other_languages.remove(target_language)
            for base_language in other_languages:

                lang_dict = {}

                for num_target in sizes[:-1]:

                    df_of_interest = df_results.loc[(df_results['model'] == model) & (df_results['target_lang'] == target_language) & 
                                                    (df_results['base_lang'] == base_language) & (df_results['num_target'] == num_target)]

                    support[support_idx] = {
                        'model': model,
                        'target_lang': target_language,
                        'base_lang': base_language,
                        'num_target': num_target,
                        'num_prompts': len(df_of_interest)
                    }
                    support_idx += 1

                    ## Can put 34 here if we only want results for full support
                    if len(df_of_interest) > 0:
                        avg_performance=average_qwk(pd.DataFrame(df_of_interest['qwk']))
                    else:
                        avg_performance=-1
                    
                    lang_dict[num_target] = avg_performance


                lang_dict[sizes[-1]] = avg_within_performance
                target_dict[base_language] = lang_dict

            # Need this column for easier processing
            target_lang_dict = {}
            for size in sizes:
                target_lang_dict[size] = -1
            
            target_dict[target_language] = target_lang_dict

            df_target = pd.DataFrame.from_dict(target_dict) 
            df_target.index.name='n'
            print(df_target)

            average_dict = {}
            df_to_average = df_target[other_languages].T
            for column in df_to_average.columns:
                df_current = df_to_average[[column]]
                df_current.columns = ['qwk']
                # print(df_current)
                this_result = average_qwk(df_current)
                average_dict[column] = this_result
            print(average_dict)
            df_average = pd.DataFrame.from_dict(average_dict, orient='index')
            df_average.columns = ['average'] 
            df_average.index.name='n'
            print(df_average)
            df_target = pd.merge(df_target, df_average, left_index=True, right_index=True)               
            df_target.to_csv(os.path.join(results_path, target_folder, model + '_' + target_language + '.csv'))


    df_support = pd.DataFrame.from_dict(support, orient='index')
    print(df_support)
    df_support.to_csv(os.path.join(results_path, target_folder, 'support.csv'))


eval(cross_results_path=os.path.join(RESULT_PATH_EXP_1,'ePIRLS'), results_path=os.path.join(RESULT_PATH_EXP_2,'ePIRLS'), update_results=True)