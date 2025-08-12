import os
import sys
import torch

import pandas as pd

from config import EPIRLS, ASAP_T, ASAP_M, SBERT_BASE_MODEL, XLMR_BASE_MODEL, RESULT_PATH_EXP_1, ANSWER_LENGTH, NPCR_BATCH_SIZE, NPCR_NUM_EPOCHS, NPCR_BATCH_SIZE_ASAP_M, RANDOM_SEED
from copy import deepcopy
from model_training.train_npcr import train_npcr
from model_training.utils import read_data, get_device, write_classification_statistics
from npcr.evaluator_core import evaluate_finetuned_model


def run_dataset(dataset_path, dataset_name, id_column, prompt_column, answer_column, target_column, languages, translate_test, batch_size, run_suffix='', run_xlmr=False, run_sbert=False):

    device = get_device()

    result_dir = os.path.join(RESULT_PATH_EXP_1 + run_suffix, dataset_name)

    for prompt in os.listdir(dataset_path):

        # For each prompt - language pair, train a model
        for language in languages:

            torch.cuda.empty_cache()

            models = []

            if run_xlmr:

                models.append((XLMR_BASE_MODEL, 'NPCR_XLMR'))
            
            if run_sbert:

                models.append((SBERT_BASE_MODEL, 'NPCR_SBERT'))

            for model in models:
                
                print(prompt, language, model)

                df_train = read_data(os.path.join(dataset_path, prompt, language, 'train.csv'), answer_column=answer_column, target_column=target_column)
                # Must shuffle! Otherwise training pairs are built to form almost exclusively with similarity label 0
                df_train = df_train.sample(frac=1).reset_index(drop=True)
                df_val = read_data(os.path.join(dataset_path, prompt, language, 'val.csv'), answer_column=answer_column, target_column=target_column)
                df_test = read_data(os.path.join(dataset_path, prompt, language, 'test.csv'), answer_column=answer_column, target_column=target_column)

                base_model = model[0]
                model_name = model[1]

                run_path = os.path.join(result_dir, prompt, language, model_name)

                if not os.path.exists(os.path.join(run_path, 'preds.csv')):
                    
                    gold, pred = train_npcr(target_path=run_path, df_train=df_train, df_val=df_val, df_test=df_test, col_id=id_column, col_prompt=prompt_column, col_answer=answer_column, col_score=target_column, base_model=base_model, max_num=ANSWER_LENGTH, batch_size=batch_size, num_epochs=NPCR_NUM_EPOCHS, save_model=True)
                    write_classification_statistics(filepath=run_path, y_true=gold, y_pred=pred)

                    # Zero-shot evaluation of finetuned model on all **other** languages
                    for test_lang in languages:

                        run_path_test = os.path.join(run_path, test_lang)

                        if not os.path.exists(run_path_test):

                            os.mkdir(run_path_test)

                        df_test_other = read_data(os.path.join(dataset_path, prompt, test_lang, 'test.csv'), answer_column=answer_column, target_column=target_column)
                        gold, pred_test = evaluate_finetuned_model(model_path=os.path.join(run_path, 'best_model'), base_model=base_model, df_ref=df_train, df_test=df_test_other, col_id=id_column, col_prompt=prompt_column, col_answer=answer_column, col_score=target_column, target_path=run_path_test, max_num=ANSWER_LENGTH, suffix='_' + str(test_lang))

                        df_test_copy = deepcopy(df_test_other)
                        df_test_copy['pred'] = pred_test
                        df_test_copy.to_csv(os.path.join(run_path_test, 'preds.csv'))

                        write_classification_statistics(filepath=run_path_test, y_true=gold, y_pred=pred_test, suffix='')

                        # Evaluate on test data translated to target language
                        if translate_test and (test_lang != language):

                            run_path_test_translated = os.path.join(run_path, test_lang + '_translated')

                            if not os.path.exists(run_path_test_translated):

                                os.mkdir(run_path_test_translated)

                            df_test_other_translated = read_data(os.path.join(dataset_path, prompt, test_lang, 'test_translated_m2m_100_1.2B_' + language + '.csv'), answer_column=answer_column, target_column=target_column)
                            gold, pred_test_translated = evaluate_finetuned_model(model_path=os.path.join(run_path, 'best_model'), base_model=base_model, df_ref=df_train, df_test=df_test_other_translated, col_id=id_column, col_prompt=prompt_column, col_answer=answer_column, col_score=target_column, target_path=run_path_test_translated, max_num=ANSWER_LENGTH, suffix='_' + str(test_lang) + '_translated')

                            df_test_translated_copy = deepcopy(df_test_other_translated)
                            df_test_translated_copy['pred'] = pred_test_translated
                            df_test_translated_copy.to_csv(os.path.join(run_path_test_translated, 'preds.csv'))

                            write_classification_statistics(filepath=run_path_test_translated, y_true=gold, y_pred=pred_test_translated, suffix='')

                    # Delete model
                    if os.path.exists(os.path.join(run_path, 'best_model')):
                        os.remove(os.path.join(run_path, 'best_model'))

                else:
                    print('Skipping prompt ' + str(prompt) + ' because it already ran!')



def run_dataset_folds(dataset_path, dataset_name, id_column, prompt_column, answer_column, target_column, languages, translate_test, num_folds, batch_size, run_suffix='', run_xlmr=True, run_sbert=True):

    device = get_device()

    result_dir = os.path.join(RESULT_PATH_EXP_1 + run_suffix, dataset_name)

    for prompt in os.listdir(dataset_path):

        # For each prompt - language pair, train a model
        for language in languages:

            for test_fold in range(1, num_folds+1):

                val_fold = test_fold + 1
                if val_fold > num_folds:
                    val_fold = 1

                training_folds = list(range(1, num_folds + 1))
                training_folds.remove(test_fold)
                training_folds.remove(val_fold)

                torch.cuda.empty_cache()

                print(prompt, language, test_fold)

                # Read data for training
                df_train_list = []

                for train_fold in training_folds:

                   df_train_list.append(read_data(os.path.join(dataset_path, prompt, language, 'fold_' + str(train_fold) + '.csv'), answer_column=answer_column, target_column=target_column))
                
                df_train = pd.concat(df_train_list)
                # Must shuffle! Otherwise training pairs are built to form almost exclusively with similarity label 0
                df_train = df_train.sample(frac=1, random_state=RANDOM_SEED).reset_index(drop=True)
                df_val = read_data(os.path.join(dataset_path, prompt, language, 'fold_' + str(val_fold) + '.csv'), answer_column=answer_column, target_column=target_column)
                df_test = read_data(os.path.join(dataset_path, prompt, language, 'fold_' + str(test_fold) + '.csv'), answer_column=answer_column, target_column=target_column)

                models = []

                if run_xlmr:

                    models.append((XLMR_BASE_MODEL, 'NPCR_XLMR'))
                
                if run_sbert:

                    models.append((SBERT_BASE_MODEL, 'NPCR_SBERT'))

                for model in models:

                    base_model = model[0]
                    model_name = model[1]

                    run_path = os.path.join(result_dir, prompt, language, model_name, 'fold_' + str(test_fold))

                    if not os.path.exists(os.path.join(run_path, 'preds.csv')):
                        
                        gold, pred = train_npcr(target_path=run_path, df_train=df_train, df_val=df_val, df_test=df_test, col_id=id_column, col_prompt=prompt_column, col_answer=answer_column, col_score=target_column, base_model=base_model, max_num=ANSWER_LENGTH, num_epochs=NPCR_NUM_EPOCHS, batch_size=batch_size, save_model=True)
                        write_classification_statistics(filepath=run_path, y_true=gold, y_pred=pred)

                        # Zero-shot evaluation of finetuned model on all **other** languages
                        for test_lang in languages:
 
                            run_path_test = os.path.join(run_path, test_lang)

                            if not os.path.exists(run_path_test):

                                os.mkdir(run_path_test)

                            df_test_other_list = []

                            if test_lang != language:

                                for fold in range(1, num_folds + 1):

                                    df_test_other_list.append(read_data(os.path.join(dataset_path, prompt, test_lang, 'fold_' + str(fold) + '.csv'), answer_column=answer_column, target_column=target_column))
                            
                            else:

                                df_test_other_list.append(read_data(os.path.join(dataset_path, prompt, test_lang, 'fold_' + str(test_fold) + '.csv'), answer_column=answer_column, target_column=target_column))

                            df_test_other = pd.concat(df_test_other_list)
                            df_test_other.reset_index(inplace=True)

                            gold, pred_test = evaluate_finetuned_model(model_path=os.path.join(run_path, 'best_model'), base_model=base_model, df_ref=df_train, df_test=df_test_other, col_id=id_column, col_prompt=prompt_column, col_answer=answer_column, col_score=target_column, target_path=run_path_test, max_num=ANSWER_LENGTH, suffix='_' + str(test_lang))

                            df_test_copy = deepcopy(df_test_other)
                            df_test_copy['pred'] = pred_test
                            df_test_copy.to_csv(os.path.join(run_path_test, 'preds.csv'))

                            write_classification_statistics(filepath=run_path_test, y_true=gold, y_pred=pred_test, suffix='')


                            # Evaluate on test data translated to target language
                            if translate_test and (test_lang != language):

                                run_path_test_translated = os.path.join(run_path, test_lang + '_translated')

                                if not os.path.exists(run_path_test_translated):

                                    os.mkdir(run_path_test_translated)

                                df_test_other_translated_list = []

                                for fold in range(1, num_folds + 1):

                                    df_test_other_translated_list.append(read_data(os.path.join(dataset_path, prompt, test_lang, 'fold_' + str(fold) + '_translated_' + language + '.csv'), answer_column=answer_column, target_column=target_column))
                                                                
                                df_test_other_translated = pd.concat(df_test_other_translated_list)
                                df_test_other_translated.reset_index(inplace=True)

                                gold, pred_test_translated = evaluate_finetuned_model(model_path=os.path.join(run_path, 'best_model'), base_model=base_model, df_ref=df_train, df_test=df_test_other_translated, col_id=id_column, col_prompt=prompt_column, col_answer=answer_column, col_score=target_column, target_path=run_path_test_translated, max_num=ANSWER_LENGTH, suffix='_' + str(test_lang) + '_translated')

                                df_test_translated_copy = deepcopy(df_test_other_translated)
                                df_test_translated_copy['pred'] = pred_test_translated
                                df_test_translated_copy.to_csv(os.path.join(run_path_test_translated, 'preds.csv'))

                                write_classification_statistics(filepath=run_path_test_translated, y_true=gold, y_pred=pred_test_translated, suffix='')

                        # Delete model
                        if os.path.exists(os.path.join(run_path, 'best_model')):

                            os.remove(os.path.join(run_path, 'best_model'))

                    else:
                        print('Skipping prompt ' + str(prompt) + ' because it already ran!')


for run in ['_RUN1']:

    for dataset in [ASAP_T, EPIRLS]:

        run_dataset(
            dataset_path=dataset['dataset_path'], 
            dataset_name=dataset['dataset_name'], 
            id_column=dataset['id_column'],
            prompt_column=dataset['prompt_column'], 
            answer_column=dataset['answer_column'], 
            target_column=dataset['target_column'], 
            languages=dataset['languages'], 
            run_suffix=run, 
            run_xlmr=False,
            run_sbert=False,
            translate_test=dataset['translate_test'],
            batch_size=NPCR_BATCH_SIZE
            )


for run in ['_RUN1']:
    
    for dataset in [ASAP_M]:

        run_dataset_folds(
            dataset_path=dataset['dataset_path'],
            dataset_name=dataset['dataset_name'],
            id_column=dataset['id_column'], 
            prompt_column=dataset['prompt_column'],
            answer_column=dataset['answer_column'], 
            target_column=dataset['target_column'],
            languages=dataset['languages'], 
            run_suffix=run, 
            run_xlmr=False,
            run_sbert=False,
            translate_test=dataset['translate_test'], 
            num_folds=dataset['num_folds'],
            batch_size=NPCR_BATCH_SIZE_ASAP_M
            )
