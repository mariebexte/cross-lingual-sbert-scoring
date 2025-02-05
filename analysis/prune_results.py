import os, sys
from config import ASAP_M, ASAP_T, EPIRLS


def prune_exp1(results_path, datasets=[ASAP_T, EPIRLS], datasets_folds=[ASAP_M]):

    for dataset in datasets:

        if os.path.exists(os.path.join(results_path, dataset['dataset_name'])):

            for prompt in dataset['prompts']:
            # for prompt in os.listdir(os.path.join(results_path, dataset['dataset_name'])):

                if os.path.isdir(os.path.join(results_path, dataset['dataset_name'], prompt)):

                    for language in dataset['languages']:

                        if os.path.exists(os.path.join(results_path, dataset['dataset_name'], prompt, language)):

                            for model in os.listdir(os.path.join(results_path, dataset['dataset_name'], prompt, language)):

                                ## See if there are predictions for all languages
                                for test_language in dataset['languages']:

                                    if not os.path.exists(os.path.join(results_path, dataset['dataset_name'], prompt, language, model, test_language, 'preds.csv')):

                                        print('MISSING', dataset['dataset_name'], prompt, language, model, test_language, 'predictions!')
                                    
                                    if dataset['translate_test'] and language != test_language:
                                    
                                        if not os.path.exists(os.path.join(results_path, dataset['dataset_name'], prompt, language, model, test_language + '_translated', 'preds.csv')):

                                            print('MISSING', dataset['dataset_name'], prompt, language, model, test_language + '_translated', 'predictions!')
                        
                        else:

                            print('MISSING TRAINING LANGUAGE', dataset['dataset_name'], prompt, language)

                else:

                    print('MISSING PROMPT', prompt)
        
        else:

            # print('MISSING dataset', dataset['dataset_name'])
            pass

        print()

    for dataset in datasets_folds:

        if os.path.exists(os.path.join(results_path, dataset['dataset_name'])):

            for prompt in dataset['prompts']:
            # for prompt in os.listdir(os.path.join(results_path, dataset['dataset_name'])):

                if os.path.isdir(os.path.join(results_path, dataset['dataset_name'], prompt)):

                    for language in dataset['languages']:

                        if os.path.exists(os.path.join(results_path, dataset['dataset_name'], prompt, language)):

                            for model in os.listdir(os.path.join(results_path, dataset['dataset_name'], prompt, language)):

                                for fold in range(1, dataset['num_folds']+1):

                                    if os.path.exists(os.path.join(results_path, dataset['dataset_name'], prompt, language, model, 'fold_' + str(fold))):
                                    
                                        ## See if there are predictions for all languages
                                        for test_language in dataset['languages']:

                                            if not os.path.exists(os.path.join(results_path, dataset['dataset_name'], prompt, language, model, 'fold_' + str(fold), test_language, 'preds.csv')):

                                                print('MISSING', dataset['dataset_name'], prompt, language, model, 'fold_' + str(fold), test_language, 'predictions!')

                                            if dataset['translate_test'] and language != test_language:

                                                if not os.path.exists(os.path.join(results_path, dataset['dataset_name'], prompt, language, model, 'fold_' + str(fold), test_language + '_translated', 'preds.csv')):

                                                    print('MISSING', dataset['dataset_name'], prompt, language, model, 'fold_' + str(fold), test_language + '_translated', 'predictions!')

                                    else:

                                        print('MISSING FOLD', os.path.join(results_path, dataset['dataset_name'], prompt, language, model, 'fold_' + str(fold)))
                        
                        else:

                            print('DOES NO EXIST:', dataset['dataset_name'], prompt, language)
                
                else:

                    print('MISSING PROMPT', prompt)
        
        else:

            # print('MISSING dataset', dataset['dataset_name'])
            pass


def prune_exp3(results_path, datasets=[ASAP_T, EPIRLS], datasets_folds=[ASAP_M]):

    for dataset in datasets:

        if os.path.exists(os.path.join(results_path, dataset['dataset_name'])):

            for prompt in dataset['prompts']:
            # for prompt in os.listdir(os.path.join(results_path, dataset['dataset_name'])):

                if os.path.isdir(os.path.join(results_path, dataset['dataset_name'], prompt)):

                    for language in dataset['languages']:

                        if os.path.exists(os.path.join(results_path, dataset['dataset_name'], prompt, language)):

                            for model in os.listdir(os.path.join(results_path, dataset['dataset_name'], prompt, language)):

                                if not os.path.exists(os.path.join(results_path, dataset['dataset_name'], prompt, language, model, 'preds.csv')):

                                    print('MISSING', dataset['dataset_name'], prompt, language, model, 'predictions!')
                                
                                # else:

                                #     print('FOUND', dataset['dataset_name'], prompt, language, model, 'predictions!')
                                                                                       
                        else:

                            print('MISSING TRAINING LANGUAGE', dataset['dataset_name'], prompt, language)

                else:

                    print('MISSING PROMPT', prompt)
        
        else:

            # print('MISSING dataset', dataset['dataset_name'])
            pass

        print()

    for dataset in datasets_folds:

        if os.path.exists(os.path.join(results_path, dataset['dataset_name'])):

            for prompt in dataset['prompts']:
            # for prompt in os.listdir(os.path.join(results_path, dataset['dataset_name'])):

                if os.path.isdir(os.path.join(results_path, dataset['dataset_name'], prompt)):

                    for language in dataset['languages']:

                        if os.path.exists(os.path.join(results_path, dataset['dataset_name'], prompt, language)):

                            for model in os.listdir(os.path.join(results_path, dataset['dataset_name'], prompt, language)):

                                for fold in range(1, dataset['num_folds']+1):

                                    if os.path.exists(os.path.join(results_path, dataset['dataset_name'], prompt, language, model, 'fold_' + str(fold))):

                                        if not os.path.exists(os.path.join(results_path, dataset['dataset_name'], prompt, language, model, 'fold_' + str(fold), 'preds.csv')):

                                            print('MISSING', dataset['dataset_name'], prompt, language, model, 'fold_' + str(fold), 'predictions!')
                                        
                                        # else:

                                        #     print('FOUND', dataset['dataset_name'], prompt, language, model, 'fold_' + str(fold), 'predictions!')

                                    else:

                                        print('MISSING FOLD', os.path.join(results_path, dataset['dataset_name'], prompt, language, model, 'fold_' + str(fold)))
                        
                        else:

                            print('DOES NO EXIST:', dataset['dataset_name'], prompt, language)
                
                else:

                    print('MISSING PROMPT', prompt)
        
        else:

            # print('MISSING dataset', dataset['dataset_name'])
            pass


# prune_exp1('/results/exp_1_zero_shot_RUN1')
# prune_exp1('/results/exp_1_zero_shot_RUN2')
# prune_exp1('/results/exp_1_zero_shot_RUN3')


for condition in ['combine_downsampled', 'combine_downsampled_translated', 'combined_all_other']:

    prune_exp3(os.path.join('/results/exp_3_lolo_RUN1/', condition))
    prune_exp3(os.path.join('/results/exp_3_lolo_RUN2/', condition))
    prune_exp3(os.path.join('/results/exp_3_lolo_RUN3/', condition))