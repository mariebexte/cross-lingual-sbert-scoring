import os
import sys
import torch

import pandas as pd

from datetime import datetime
from config import EPIRLS, ASAP_T, ASAP_M, SBERT_BASE_MODEL, RESULT_PATH_EXP_1, ANSWER_LENGTH
from copy import deepcopy
from model_training.utils import read_data, get_device, eval_sbert, write_classification_statistics
from sentence_transformers import SentenceTransformer


def run_pretrained(dataset_path, dataset_name, id_column, answer_column, target_column, languages, translate_test, run_suffix=''):

    device = get_device()

    for prompt in os.listdir(dataset_path):

        # For each prompt - language pair, train a model
        for language in languages:

            start = datetime.now()

            torch.cuda.empty_cache()

            print(prompt, language)

            df_train = read_data(os.path.join(dataset_path, prompt, language, 'train.csv'), target_column=target_column, answer_column=answer_column)
            df_val = read_data(os.path.join(dataset_path, prompt, language, 'val.csv'), target_column=target_column, answer_column=answer_column)
            df_test = read_data(os.path.join(dataset_path, prompt, language, 'test.csv'), target_column=target_column, answer_column=answer_column)
            
            run_path_sbert = os.path.join(RESULT_PATH_EXP_1 + run_suffix, dataset_name, prompt, language, 'pretrained')

            if not os.path.exists(os.path.join(run_path_sbert)):

                os.makedirs(run_path_sbert)

                # Load model
                model = SentenceTransformer(SBERT_BASE_MODEL)
                model.max_seq_length=ANSWER_LENGTH
                df_ref = deepcopy(df_train)
                # df_ref = pd.concat([df_train, df_val])
                df_ref['embedding'] = df_ref[answer_column].apply(model.encode)
                
                # Zero-shot evaluation of model on all languages
                for test_lang in languages:

                    run_path_test_sbert = os.path.join(run_path_sbert, test_lang)

                    if not os.path.exists(run_path_test_sbert):

                        os.mkdir(run_path_test_sbert)
                    
                    df_test_sbert = read_data(os.path.join(dataset_path, prompt, test_lang, 'test.csv'), target_column=target_column, answer_column=answer_column)
                    df_test_sbert['embedding'] = df_test_sbert[answer_column].apply(model.encode)
                    gold, pred_max, pred_avg = eval_sbert(run_path_test_sbert, df_test=df_test_sbert, df_ref=df_ref, id_column=id_column, answer_column=answer_column, target_column=target_column)

                    write_classification_statistics(filepath=run_path_test_sbert, y_true=gold, y_pred=pred_avg, suffix='')
                    write_classification_statistics(filepath=run_path_test_sbert, y_true=gold, y_pred=pred_max, suffix='_max')
                    write_classification_statistics(filepath=run_path_test_sbert, y_true=gold, y_pred=pred_hybrid, suffix='_hybrid')

                    if translate_test and test_lang != language:

                        run_path_test_sbert_translated = os.path.join(run_path_sbert, test_lang + '_translated')

                        if not os.path.exists(run_path_test_sbert_translated):

                            os.mkdir(run_path_test_sbert_translated)
                        
                        df_test_sbert_translated = read_data(os.path.join(dataset_path, prompt, test_lang, 'test_translated_m2m_100_1.2B_' + language + '.csv'), target_column=target_column, answer_column=answer_column)
                        df_test_sbert_translated['embedding'] = df_test_sbert_translated[answer_column].apply(model.encode)
                        gold, pred_max_translated, pred_avg_translated, pred_hybrid_translated = eval_sbert(run_path_test_sbert_translated, df_test=df_test_sbert_translated, df_ref=df_ref, id_column=id_column, answer_column=answer_column, target_column=target_column)

                        write_classification_statistics(filepath=run_path_test_sbert_translated, y_true=gold, y_pred=pred_avg_translated, suffix='')
                        write_classification_statistics(filepath=run_path_test_sbert_translated, y_true=gold, y_pred=pred_max_translated, suffix='_max')
                        write_classification_statistics(filepath=run_path_test_sbert_translated, y_true=gold, y_pred=pred_hybrid_translated, suffix='_hybrid')
                    
                end = datetime.now()

                with open(os.path.join(run_path_sbert, datetime.now().strftime('logs_%H_%M_%d_%m_%Y.log')), 'w') as out_file:

                    out_file.write('Total duration:\t' + str(end - start))



def run_pretrained_cross_validated(dataset_path, dataset_name, id_column, answer_column, target_column, languages, translate_test, num_folds, run_suffix=''):

    device = get_device()

    for prompt in os.listdir(dataset_path):

        # For each prompt - language pair, train a model
        for language in languages:

            for test_fold in range(1, num_folds+1):

                start = datetime.now()

                torch.cuda.empty_cache()

                print(prompt, language, test_fold)

                val_fold = test_fold+1
                if val_fold > num_folds:
                    val_fold=1

                ref_folds = list(range(1, num_folds+1))
                ref_folds.remove(test_fold)
                ref_folds.remove(val_fold)

                df_ref_list = []

                for ref_fold in ref_folds:

                    df_ref_list.append(read_data(os.path.join(dataset_path, prompt, language, 'fold_' + str(ref_fold) + '.csv'), target_column=target_column, answer_column=answer_column))
                
                df_ref = pd.concat(df_ref_list)
                df_ref.reset_index(inplace=True)
                df_test = read_data(os.path.join(dataset_path, prompt, language, 'fold_' + str(test_fold)+ '.csv'), target_column=target_column, answer_column=answer_column)
                
                run_path_sbert = os.path.join(RESULT_PATH_EXP_1 + run_suffix, dataset_name, prompt, language, 'pretrained', 'fold_' + str(test_fold))

                if not os.path.exists(os.path.join(run_path_sbert)):

                    os.makedirs(run_path_sbert)

                    # Load model
                    model = SentenceTransformer(SBERT_BASE_MODEL)
                    model.max_seq_length=128
                    df_ref['embedding'] = df_ref[answer_column].apply(model.encode)
                    
                    # Zero-shot evaluation of model on all languages
                    for test_lang in languages:

                        run_path_test_sbert = os.path.join(run_path_sbert, test_lang)

                        if not os.path.exists(run_path_test_sbert):

                            os.mkdir(run_path_test_sbert)
                        
                        df_test_sbert_list = []

                        if test_lang != language:

                            for fold in range(1, num_folds+1):

                                df_test_sbert_list.append(read_data(os.path.join(dataset_path, prompt, test_lang, 'fold_' + str(fold) + '.csv'), target_column=target_column, answer_column=answer_column))
                        
                        else:

                            df_test_sbert_list.append(read_data(os.path.join(dataset_path, prompt, test_lang, 'fold_' + str(test_fold) + '.csv'), target_column=target_column, answer_column=answer_column))
                        
                        df_test_sbert=pd.concat(df_test_sbert_list)
                        df_test_sbert.reset_index(inplace=True)
                        df_test_sbert['embedding'] = df_test_sbert[answer_column].apply(model.encode)
                        gold, pred_max, pred_avg, pred_hybrid = eval_sbert(run_path_test_sbert, df_test=df_test_sbert, df_ref=df_ref, id_column=id_column, answer_column=answer_column, target_column=target_column)

                        write_classification_statistics(filepath=run_path_test_sbert, y_true=gold, y_pred=pred_avg, suffix='')
                        write_classification_statistics(filepath=run_path_test_sbert, y_true=gold, y_pred=pred_max, suffix='_max')
                        write_classification_statistics(filepath=run_path_test_sbert, y_true=gold, y_pred=pred_hybrid, suffix='_hybrid')

                        if translate_test and test_lang != language:

                            run_path_test_sbert_translated = os.path.join(run_path_sbert, test_lang + '_translated')

                            if not os.path.exists(run_path_test_sbert_translated):

                                os.mkdir(run_path_test_sbert_translated)
                            
                            df_test_sbert_translated_list = []

                            for fold in (range(1, num_folds+1)):

                                df_test_sbert_translated_list.append(read_data(os.path.join(dataset_path, prompt, test_lang, 'fold_' + str(fold) + '_translated_' + language + '.csv'), target_column=target_column, answer_column=answer_column))
                            
                            df_test_sbert_translated = pd.concat(df_test_sbert_translated_list)
                            df_test_sbert_translated.reset_index(inplace=True)
                            df_test_sbert_translated['embedding'] = df_test_sbert_translated[answer_column].apply(model.encode)
                            gold, pred_max_translated, pred_avg_translated, pred_hybrid_translated = eval_sbert(run_path_test_sbert_translated, df_test=df_test_sbert_translated, df_ref=df_ref, id_column=id_column, answer_column=answer_column, target_column=target_column)

                            write_classification_statistics(filepath=run_path_test_sbert_translated, y_true=gold, y_pred=pred_avg_translated, suffix='')
                            write_classification_statistics(filepath=run_path_test_sbert_translated, y_true=gold, y_pred=pred_max_translated, suffix='_max')
                            write_classification_statistics(filepath=run_path_test_sbert_translated, y_true=gold, y_pred=pred_hybrid_translated, suffix='_hybrid')
                        
                    end = datetime.now()

                    with open(os.path.join(run_path_sbert, datetime.now().strftime('logs_%H_%M_%d_%m_%Y.log')), 'w') as out_file:

                        out_file.write('Total duration:\t' + str(end - start))


for run in ['_RUN1']:

    for dataset in [ASAP_T, EPIRLS]:

        run_pretrained(
            dataset_path=dataset['dataset_path'],
            dataset_name=dataset['dataset_name'], 
            id_column=dataset['id_column'], 
            answer_column=dataset['answer_column'], 
            target_column=dataset['target_column'],
            languages=dataset['languages'],
            translate_test=dataset['translate_test'], 
            run_suffix=run
        )


for run in ['_RUN1']:

    for dataset in [ASAP_M]:

        run_pretrained_cross_validated(
            dataset_path=dataset['dataset_path'],
            dataset_name=dataset['dataset_name'], 
            id_column=dataset['id_column'], 
            answer_column=dataset['answer_column'], 
            target_column=dataset['target_column'],
            languages=dataset['languages'],
            translate_test=dataset['translate_test'], 
            run_suffix=run,
            num_folds=dataset['num_folds']
        )
