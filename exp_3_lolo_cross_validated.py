import os
import shutil
import sys
import torch

import pandas as pd

from config import NPCR_ANSWER_LENGTH, ASAP_M, SBERT_NUM_EPOCHS, BERT_NUM_EPOCHS, SBERT_BASE_MODEL, XLMR_BASE_MODEL, SBERT_NUM_PAIRS, SBERT_NUM_VAL_PAIRS, RESULT_PATH_EXP_3
from copy import deepcopy
from model_training.train_xlmr import train_xlmr
from model_training.train_xlmr_sbert_core import train_xlmr as train_xlmr_sbert_core
from model_training.train_sbert import train_sbert
from model_training.train_npcr import train_npcr
from model_training.utils import read_data, get_device, eval_sbert, write_classification_statistics
from sentence_transformers import SentenceTransformer


random_state = 3456786544


def run_full(dataset_path, dataset_name, id_column, prompt_column, answer_column, target_column, languages, translate_train, num_folds, run_suffix='', run_xlmr=True, run_sbert=True, run_npcr_xlmr=True, run_npcr_sbert=True, run_xlmr_swap_sbert=True, run_sbert_swap_xlmr=True, run_pretrained=True, bert_batch_size=32, sbert_batch_size=64):

    device = get_device()

    condition = 'combine_all_other'

    if translate_train:

        condition = condition + '_translated'

    for prompt in os.listdir(dataset_path):

        # For each prompt - language pair, train a model
        for test_language in languages:

            all_predictions = {}

            for val_fold in range(1, num_folds + 1):

                print(prompt, test_language, val_fold)

                # Read test, val data
                # Training is combination of data in other languages
                df_val = read_data(os.path.join(dataset_path, prompt, test_language, 'fold_' + str(val_fold) + '.csv'), answer_column=answer_column, target_column=target_column)
                df_test = pd.DataFrame()

                test_folds = list(range(1, num_folds + 1))
                test_folds.remove(val_fold)

                for test_fold in test_folds:
                    df_temp = read_data(os.path.join(dataset_path, prompt, test_language, 'fold_' + str(test_fold) + '.csv'), answer_column=answer_column, target_column=target_column)
                    df_test = pd.concat([df_test, df_temp])

                df_test.reset_index(inplace=True)

                # Combine data of all *other* languages as training data
                df_train = pd.DataFrame()
                other_languages = deepcopy(languages)
                other_languages.remove(test_language)
                
                for other_language in other_languages:

                    train_folds = list(range(1, num_folds + 1))
                    train_folds.remove(val_fold)

                    if val_fold + 1 > num_folds:

                        train_folds.remove(1)

                    else:

                        train_folds.remove(val_fold+1)

                    for train_fold in train_folds:

                        df_other = pd.DataFrame()

                        if translate_train:   

                            df_other = read_data(os.path.join(dataset_path, prompt, other_language, 'fold_' + str(train_fold) + '_translated_' + test_language + '.csv'), answer_column=answer_column, target_column=target_column)
                        
                        else:

                            df_other = read_data(os.path.join(dataset_path, prompt, other_language, 'fold_' + str(train_fold) + '.csv'), answer_column=answer_column, target_column=target_column)

                        df_train = pd.concat([df_train, df_other])
                
                df_train.reset_index(inplace=True)

                if run_xlmr:

                    run_path_bert = os.path.join(RESULT_PATH_EXP_3 + run_suffix, condition, dataset_name, prompt, test_language, 'XLMR', 'fold_' + str(val_fold))

                    if not os.path.exists(os.path.join(run_path_bert, 'preds.csv')):

                        if not os.path.exists(run_path_bert):

                            os.makedirs(run_path_bert)

                        gold, xlmr_pred = train_xlmr(run_path_bert, df_train=df_train, df_val=df_val, df_test=df_test, answer_column=answer_column, target_column=target_column, base_model=XLMR_BASE_MODEL, num_epochs=BERT_NUM_EPOCHS, batch_size=bert_batch_size, save_model=False)
                        write_classification_statistics(filepath=run_path_bert, y_true=gold, y_pred=xlmr_pred)
                        df_train.to_csv(os.path.join(run_path_bert, 'train.csv'))
                        df_val.to_csv(os.path.join(run_path_bert, 'val.csv'))
                        df_test.to_csv(os.path.join(run_path_bert, 'test.csv'))

                        preds_xlmr = all_predictions.get('XLMR', pd.DataFrame())
                        preds_xlmr = pd.concat([preds_xlmr, pd.read_csv(os.path.join(run_path_bert, 'preds.csv'))])
                        all_predictions['XLMR'] = preds_xlmr


                if run_sbert:

                    run_path_sbert = os.path.join(RESULT_PATH_EXP_3 + run_suffix, condition, dataset_name, prompt, test_language, 'SBERT', 'fold_' + str(val_fold))
                    
                    if not os.path.exists(os.path.join(run_path_sbert, 'preds.csv')):

                        if not os.path.exists(run_path_sbert):

                            os.makedirs(run_path_sbert)

                        gold, pred_max, pred_avg = train_sbert(run_path_sbert, df_train=df_train, df_val=df_val, df_test=df_test, id_column=id_column, answer_column=answer_column, target_column=target_column, base_model=SBERT_BASE_MODEL, num_epochs=SBERT_NUM_EPOCHS, batch_size=sbert_batch_size, do_warmup=False, save_model=False, num_pairs_per_example=SBERT_NUM_PAIRS, num_val_pairs=SBERT_NUM_VAL_PAIRS)
                        write_classification_statistics(filepath=run_path_sbert, y_true=gold, y_pred=pred_avg, suffix='')
                        write_classification_statistics(filepath=run_path_sbert, y_true=gold, y_pred=pred_max, suffix='_max')
                        df_train.to_csv(os.path.join(run_path_sbert, 'train.csv'))
                        df_val.to_csv(os.path.join(run_path_sbert, 'val.csv'))
                        df_test.to_csv(os.path.join(run_path_sbert, 'test.csv'))

                        preds_sbert = all_predictions.get('SBERT', pd.DataFrame())
                        preds_sbert = pd.concat([preds_sbert, pd.read_csv(os.path.join(run_path_sbert, 'preds.csv'))])
                        all_predictions['SBERT'] = preds_sbert

                
                if run_xlmr_swap_sbert:

                    run_path_bert_swap_sbert = os.path.join(RESULT_PATH_EXP_3 + run_suffix, condition, dataset_name, prompt, language, 'XLMR_SBERTcore', 'fold_' + str(val_fold))

                    if not os.path.exists(os.path.join(run_path_bert_swap_sbert, 'preds.csv')):

                        gold, xlmr_swap_sbert_pred = train_xlmr_sbert_core(run_path_bert_swap_sbert, df_train=df_train, df_val=df_val, df_test=df_test, answer_column=answer_column, target_column=target_column, num_epochs=BERT_NUM_EPOCHS, batch_size=bert_batch_size, save_model=False, base_model='/models/'+SBERT_BASE_MODEL)

                        write_classification_statistics(filepath=run_path_bert_swap_sbert, y_true=gold, y_pred=xlmr_swap_sbert_pred)
                        df_train.to_csv(os.path.join(run_path_bert_swap_sbert, 'train.csv'))
                        df_val.to_csv(os.path.join(run_path_bert_swap_sbert, 'val.csv'))
                        df_test.to_csv(os.path.join(run_path_bert_swap_sbert, 'test.csv'))

                        preds_xlmr_swap_sbert = all_predictions.get('XLMR_SBERTcore', pd.DataFrame())
                        preds_xlmr_swap_sbert = pd.concat([preds_xlmr_swap_sbert, pd.read_csv(os.path.join(run_path_bert_swap_sbert, 'preds.csv'))])
                        all_predictions['XLMR_SBERTcore'] = preds_xlmr_swap_sbert

                
                if run_sbert_swap_xlmr:

                    run_path_sbert_swap_xlmr = os.path.join(RESULT_PATH_EXP_3 + run_suffix, condition, dataset_name, prompt, test_language, 'SBERT_XLMRcore', 'fold_' + str(val_fold))

                    if not os.path.exists(os.path.join(run_path_sbert_swap_xlmr, 'preds.csv')):

                        gold, pred_max_xlmr_core, pred_avg_xlmr_core = train_sbert(run_path_sbert_swap_xlmr, answer_column=answer_column, id_column=id_column, target_column=target_column, df_train=df_train, df_val=df_val, df_test=df_test, base_model=XLMR_BASE_MODEL, num_epochs=SBERT_NUM_EPOCHS, batch_size=sbert_batch_size, do_warmup=False, save_model=False, num_pairs_per_example=SBERT_NUM_PAIRS, num_val_pairs=SBERT_NUM_VAL_PAIRS)
                        write_classification_statistics(filepath=run_path_sbert_swap_xlmr, y_true=gold, y_pred=pred_avg_xlmr_core, suffix='')
                        write_classification_statistics(filepath=run_path_sbert_swap_xlmr, y_true=gold, y_pred=pred_max_xlmr_core, suffix='_max')
                        df_train.to_csv(os.path.join(run_path_sbert_swap_xlmr, 'train.csv'))
                        df_val.to_csv(os.path.join(run_path_sbert_swap_xlmr, 'val.csv'))
                        df_test.to_csv(os.path.join(run_path_sbert_swap_xlmr, 'test.csv'))

                        preds_sbert_swap_xlmr = all_predictions.get('SBERT_XLMRcore', pd.DataFrame())
                        preds_sbert_swap_xlmr = pd.concat([preds_sbert_swap_xlmr, pd.read_csv(os.path.join(run_path_sbert_swap_xlmr, 'preds.csv'))])
                        all_predictions['SBERT_XLMRcore'] = preds_sbert_swap_xlmr


                if run_npcr_xlmr:

                    run_path_npcr_xlmr = os.path.join(RESULT_PATH_EXP_3 + run_suffix, condition, dataset_name, prompt, test_language, 'NPCR_XLMR', 'fold_' + str(val_fold))
                    
                    if not os.path.exists(os.path.join(run_path_npcr_xlmr, 'preds.csv')):

                        if not os.path.exists(run_path_npcr_xlmr):

                            os.makedirs(run_path_npcr_xlmr)

                        gold, npcr_xlmr_pred = train_npcr(target_path=run_path_npcr_xlmr, df_train=df_train, df_val=df_val, df_test=df_test, col_prompt=prompt_column, col_answer=answer_column, col_score=target_column, base_model=XLMR_BASE_MODEL, max_num=NPCR_ANSWER_LENGTH, training_with_same_score=True, save_model=False)
                        write_classification_statistics(filepath=run_path_npcr_xlmr, y_true=gold, y_pred=npcr_xlmr_pred)
                        df_train.to_csv(os.path.join(run_path_npcr_xlmr, 'train.csv'))
                        df_val.to_csv(os.path.join(run_path_npcr_xlmr, 'val.csv'))
                        df_test.to_csv(os.path.join(run_path_npcr_xlmr, 'test.csv'))

                        preds_npcr_xlmr = all_predictions.get('NPCR_XLMR', pd.DataFrame())
                        preds_npcr_xlmr = pd.concat([preds_npcr_xlmr, pd.read_csv(os.path.join(run_path_npcr_xlmr, 'preds.csv'))])
                        all_predictions['NPCR_XLMR'] = preds_npcr_xlmr


                if run_npcr_sbert:

                    run_path_npcr_sbert = os.path.join(RESULT_PATH_EXP_3 + run_suffix, condition, dataset_name, prompt, test_language, 'NPCR_SBERT', 'fold_' + str(val_fold))
                    
                    if not os.path.exists(os.path.join(run_path_npcr_sbert, 'preds.csv')):

                        if not os.path.exists(run_path_npcr_sbert):

                            os.makedirs(run_path_npcr_sbert)

                        gold, npcr_sbert_pred = train_npcr(target_path=run_path_npcr_sbert, df_train=df_train, df_val=df_val, df_test=df_test, col_prompt=prompt_column, col_answer=answer_column, col_score=target_column, base_model=SBERT_BASE_MODEL, max_num=NPCR_ANSWER_LENGTH, training_with_same_score=True, save_model=False)
                        write_classification_statistics(filepath=run_path_npcr_sbert, y_true=gold, y_pred=npcr_sbert_pred)
                        df_train.to_csv(os.path.join(run_path_npcr_sbert, 'train.csv'))
                        df_val.to_csv(os.path.join(run_path_npcr_sbert, 'val.csv'))
                        df_test.to_csv(os.path.join(run_path_npcr_sbert, 'test.csv'))

                        preds_npcr_sbert = all_predictions.get('NPCR_SBERT', pd.DataFrame())
                        preds_npcr_sbert = pd.concat([preds_npcr_sbert, pd.read_csv(os.path.join(run_path_npcr_sbert, 'preds.csv'))])
                        all_predictions['NPCR_SBERT'] = preds_npcr_sbert


                if run_pretrained:

                    run_path_pretrained = os.path.join(RESULT_PATH_EXP_3 + run_suffix, condition, dataset_name, prompt, test_language, 'pretrained', 'fold_' + str(val_fold))
                    
                    if not os.path.exists(os.path.join(run_path_pretrained, 'preds.csv')):

                        if not os.path.exists(run_path_pretrained):

                            os.makedirs(run_path_pretrained)
                            
                        # Load pretrained model 
                        model = SentenceTransformer(SBERT_BASE_MODEL)
                        df_ref = pd.concat([df_train, df_val])
                        df_ref['embedding'] = df_ref[answer_column].apply(model.encode)
                        df_test['embedding'] = df_test[answer_column].apply(model.encode)

                        # Predict on within-test data
                        gold, pred_max_pretrained, pred_avg_pretrained = eval_sbert(run_path_pretrained, df_test=df_test, df_ref=df_ref, id_column=id_column, answer_column=answer_column, target_column=target_column)
                        write_classification_statistics(filepath=run_path_pretrained, y_true=gold, y_pred=pred_avg_pretrained, suffix='')
                        write_classification_statistics(filepath=run_path_pretrained, y_true=gold, y_pred=pred_max_pretrained, suffix='_max')
                        df_train.to_csv(os.path.join(run_path_pretrained, 'train.csv'))
                        df_val.to_csv(os.path.join(run_path_pretrained, 'val.csv'))
                        df_test.to_csv(os.path.join(run_path_pretrained, 'test.csv'))

                        preds_pretrained = all_predictions.get('pretrained', pd.DataFrame())
                        preds_pretrained = pd.concat([preds_pretrained, pd.read_csv(os.path.join(run_path_pretrained, 'preds.csv'))])
                        all_predictions['pretrained'] = preds_pretrained


            for model, df_preds in all_predictions.items():
                    
                df_preds.to_csv(os.path.join(RESULT_PATH_EXP_3 + run_suffix, condition, dataset_name, prompt, test_language, model, 'preds.csv'))
                df_preds = df_preds.reset_index()
                gold = df_preds[target_column]

                if model == 'SBERT' or model == 'pretrained' or model == 'SBERT_XLMRcore':

                    pred_avg = df_preds['pred_avg']
                    pred_max = df_preds['pred_max']

                    write_classification_statistics(filepath=os.path.join(RESULT_PATH_EXP_3 + run_suffix, condition, dataset_name, prompt, test_language, model), y_true=gold, y_pred=pred_avg, suffix='')
                    write_classification_statistics(filepath=os.path.join(RESULT_PATH_EXP_3 + run_suffix, condition, dataset_name, prompt, test_language, model), y_true=gold, y_pred=pred_max, suffix='_max')
                
                else:

                    pred = df_preds['pred']
                    write_classification_statistics(filepath=os.path.join(RESULT_PATH_EXP_3 + run_suffix, condition, dataset_name, prompt, test_language, model), y_true=gold, y_pred=pred, suffix='')


def run_downsampled(dataset_path, dataset_name, id_column, prompt_column, answer_column, target_column, languages, translate_train, num_folds, run_suffix='', run_xlmr=True, run_sbert=True, run_npcr_xlmr=True, run_npcr_sbert=True, run_xlmr_swap_sbert=True, run_sbert_swap_xlmr=True, run_pretrained=True, bert_batch_size=32, sbert_batch_size=64):

    device = get_device()
    condition = 'combine_downsampled'

    if translate_train:

        condition = condition + '_translated'

    for prompt in os.listdir(dataset_path):

        # For each prompt - language pair, train a model
        for test_language in languages:

            all_predictions = {}

            for val_fold in range(1, num_folds + 1):

                print(prompt, test_language, val_fold)

                # Read test, val data
                # Training is combination of data in other languages
                df_val = read_data(os.path.join(dataset_path, prompt, test_language, 'fold_' + str(val_fold) + '.csv'), answer_column=answer_column, target_column=target_column)
                df_test = pd.DataFrame()

                test_folds = list(range(1, num_folds + 1))
                test_folds.remove(val_fold)

                for test_fold in test_folds:
                    df_temp = read_data(os.path.join(dataset_path, prompt, test_language, 'fold_' + str(test_fold) + '.csv'), answer_column=answer_column, target_column=target_column)
                    df_test = pd.concat([df_test, df_temp])

                df_test.reset_index(inplace=True)
                # Combine data of all *other* languages as training data
                other_languages = deepcopy(languages)
                other_languages.remove(test_language)

                dfs = []
                
                for other_language in other_languages:

                    if (val_fold + 1) > num_folds:
                        train_folds = [1, val_fold]
                    else:
                        train_folds = [val_fold, val_fold + 1]

                    # Take all of the first fold, and a quarter of the second
                    df_other = pd.DataFrame()

                    if translate_train:   

                        df_other = read_data(os.path.join(dataset_path, prompt, other_language, 'fold_' + str(train_folds[0]) + '_translated_' + test_language + '.csv'), answer_column=answer_column, target_column=target_column)
                    
                    else:

                        df_other = read_data(os.path.join(dataset_path, prompt, other_language, 'fold_' + str(train_folds[0]) + '.csv'), answer_column=answer_column, target_column=target_column)

                    df_rest = pd.DataFrame()

                    if translate_train:  

                        df_rest = read_data(os.path.join(dataset_path, prompt, other_language, 'fold_' + str(train_folds[1]) + '_translated_' + test_language + '.csv'), answer_column=answer_column, target_column=target_column)
                    
                    else:

                        df_rest = read_data(os.path.join(dataset_path, prompt, other_language, 'fold_' + str(train_folds[1]) + '.csv'), answer_column=answer_column, target_column=target_column)
                    
                    num_to_sample = len(df_other)/len(other_languages)
                    df_sample = df_rest.sample(int(num_to_sample), random_state=random_state)
                    dfs.append(df_other)
                    dfs.append(df_sample)
                    
                df_train = pd.concat(dfs)
                df_train.reset_index(inplace=True)

                if run_xlmr:

                    run_path_bert = os.path.join(RESULT_PATH_EXP_3 + run_suffix, condition, dataset_name, prompt, test_language, 'XLMR', 'fold_' + str(val_fold))

                    if not os.path.exists(os.path.join(run_path_bert, 'preds.csv')):

                        if not os.path.exists(run_path_bert):

                            os.makedirs(run_path_bert)

                        gold, xlmr_pred = train_xlmr(run_path_bert, df_train=df_train, df_val=df_val, df_test=df_test, answer_column=answer_column, target_column=target_column, base_model=XLMR_BASE_MODEL, num_epochs=BERT_NUM_EPOCHS, batch_size=bert_batch_size, save_model=False)
                        write_classification_statistics(filepath=run_path_bert, y_true=gold, y_pred=xlmr_pred)
                        df_train.to_csv(os.path.join(run_path_bert, 'train.csv'))
                        df_val.to_csv(os.path.join(run_path_bert, 'val.csv'))
                        df_test.to_csv(os.path.join(run_path_bert, 'test.csv'))

                        preds_xlmr = all_predictions.get('XLMR', pd.DataFrame())
                        preds_xlmr = pd.concat([preds_xlmr, pd.read_csv(os.path.join(run_path_bert, 'preds.csv'))])
                        all_predictions['XLMR'] = preds_xlmr


                if run_sbert:

                    run_path_sbert = os.path.join(RESULT_PATH_EXP_3 + run_suffix, condition, dataset_name, prompt, test_language, 'SBERT', 'fold_' + str(val_fold))
                    
                    if not os.path.exists(os.path.join(run_path_sbert, 'preds.csv')):

                        if not os.path.exists(run_path_sbert):

                            os.makedirs(run_path_sbert)

                        gold, pred_max, pred_avg = train_sbert(run_path_sbert, df_train=df_train, df_val=df_val, df_test=df_test, id_column=id_column, answer_column=answer_column, target_column=target_column, base_model=SBERT_BASE_MODEL, num_epochs=SBERT_NUM_EPOCHS, batch_size=sbert_batch_size, do_warmup=False, save_model=False, num_pairs_per_example=SBERT_NUM_PAIRS, num_val_pairs=SBERT_NUM_VAL_PAIRS)
                        write_classification_statistics(filepath=run_path_sbert, y_true=gold, y_pred=pred_avg, suffix='')
                        write_classification_statistics(filepath=run_path_sbert, y_true=gold, y_pred=pred_max, suffix='_max')
                        df_train.to_csv(os.path.join(run_path_sbert, 'train.csv'))
                        df_val.to_csv(os.path.join(run_path_sbert, 'val.csv'))
                        df_test.to_csv(os.path.join(run_path_sbert, 'test.csv'))

                        preds_sbert = all_predictions.get('SBERT', pd.DataFrame())
                        preds_sbert = pd.concat([preds_sbert, pd.read_csv(os.path.join(run_path_sbert, 'preds.csv'))])
                        all_predictions['SBERT'] = preds_sbert

                
                if run_xlmr_swap_sbert:

                    run_path_bert_swap_sbert = os.path.join(RESULT_PATH_EXP_3 + run_suffix, condition, dataset_name, prompt, language, 'XLMR_SBERTcore', 'fold_' + str(val_fold))

                    if not os.path.exists(os.path.join(run_path_bert_swap_sbert, 'preds.csv')):

                        gold, xlmr_swap_sbert_pred = train_xlmr_sbert_core(run_path_bert_swap_sbert, df_train=df_train, df_val=df_val, df_test=df_test, answer_column=answer_column, target_column=target_column, num_epochs=BERT_NUM_EPOCHS, batch_size=bert_batch_size, save_model=False, base_model='/models/'+SBERT_BASE_MODEL)

                        write_classification_statistics(filepath=run_path_bert_swap_sbert, y_true=gold, y_pred=xlmr_swap_sbert_pred)
                        df_train.to_csv(os.path.join(run_path_bert_swap_sbert, 'train.csv'))
                        df_val.to_csv(os.path.join(run_path_bert_swap_sbert, 'val.csv'))
                        df_test.to_csv(os.path.join(run_path_bert_swap_sbert, 'test.csv'))

                        preds_xlmr_swap_sbert = all_predictions.get('XLMR_SBERTcore', pd.DataFrame())
                        preds_xlmr_swap_sbert = pd.concat([preds_xlmr_swap_sbert, pd.read_csv(os.path.join(run_path_bert_swap_sbert, 'preds.csv'))])
                        all_predictions['XLMR_SBERTcore'] = preds_xlmr_swap_sbert

                
                if run_sbert_swap_xlmr:

                    run_path_sbert_swap_xlmr = os.path.join(RESULT_PATH_EXP_3 + run_suffix, condition, dataset_name, prompt, test_language, 'SBERT_XLMRcore', 'fold_' + str(val_fold))

                    if not os.path.exists(os.path.join(run_path_sbert_swap_xlmr, 'preds.csv')):

                        gold, pred_max_xlmr_core, pred_avg_xlmr_core = train_sbert(run_path_sbert_swap_xlmr, answer_column=answer_column, id_column=id_column, target_column=target_column, df_train=df_train, df_val=df_val, df_test=df_test, base_model=XLMR_BASE_MODEL, num_epochs=SBERT_NUM_EPOCHS, batch_size=sbert_batch_size, do_warmup=False, save_model=False, num_pairs_per_example=SBERT_NUM_PAIRS, num_val_pairs=SBERT_NUM_VAL_PAIRS)
                        write_classification_statistics(filepath=run_path_sbert_swap_xlmr, y_true=gold, y_pred=pred_avg_xlmr_core, suffix='')
                        write_classification_statistics(filepath=run_path_sbert_swap_xlmr, y_true=gold, y_pred=pred_max_xlmr_core, suffix='_max')
                        df_train.to_csv(os.path.join(run_path_sbert_swap_xlmr, 'train.csv'))
                        df_val.to_csv(os.path.join(run_path_sbert_swap_xlmr, 'val.csv'))
                        df_test.to_csv(os.path.join(run_path_sbert_swap_xlmr, 'test.csv'))

                        preds_sbert_swap_xlmr = all_predictions.get('SBERT_XLMRcore', pd.DataFrame())
                        preds_sbert_swap_xlmr = pd.concat([preds_sbert_swap_xlmr, pd.read_csv(os.path.join(run_path_sbert_swap_xlmr, 'preds.csv'))])
                        all_predictions['SBERT_XLMRcore'] = preds_sbert_swap_xlmr


                if run_npcr_xlmr:

                    run_path_npcr_xlmr = os.path.join(RESULT_PATH_EXP_3 + run_suffix, condition, dataset_name, prompt, test_language, 'NPCR_XLMR', 'fold_' + str(val_fold))
                    
                    if not os.path.exists(os.path.join(run_path_npcr_xlmr, 'preds.csv')):

                        if not os.path.exists(run_path_npcr_xlmr):

                            os.makedirs(run_path_npcr_xlmr)

                        gold, npcr_xlmr_pred = train_npcr(target_path=run_path_npcr_xlmr, df_train=df_train, df_val=df_val, df_test=df_test, col_prompt=prompt_column, col_answer=answer_column, col_score=target_column, base_model=XLMR_BASE_MODEL, max_num=NPCR_ANSWER_LENGTH, training_with_same_score=True, save_model=False)
                        write_classification_statistics(filepath=run_path_npcr_xlmr, y_true=gold, y_pred=npcr_xlmr_pred)
                        df_train.to_csv(os.path.join(run_path_npcr_xlmr, 'train.csv'))
                        df_val.to_csv(os.path.join(run_path_npcr_xlmr, 'val.csv'))
                        df_test.to_csv(os.path.join(run_path_npcr_xlmr, 'test.csv'))

                        preds_npcr_xlmr = all_predictions.get('NPCR_XLMR', pd.DataFrame())
                        preds_npcr_xlmr = pd.concat([preds_npcr_xlmr, pd.read_csv(os.path.join(run_path_npcr_xlmr, 'preds.csv'))])
                        all_predictions['NPCR_XLMR'] = preds_npcr_xlmr


                if run_npcr_sbert:

                    run_path_npcr_sbert = os.path.join(RESULT_PATH_EXP_3 + run_suffix, condition, dataset_name, prompt, test_language, 'NPCR_SBERT', 'fold_' + str(val_fold))
                    
                    if not os.path.exists(os.path.join(run_path_npcr_sbert, 'preds.csv')):

                        if not os.path.exists(run_path_npcr_sbert):

                            os.makedirs(run_path_npcr_sbert)

                        gold, npcr_sbert_pred = train_npcr(target_path=run_path_npcr_sbert, df_train=df_train, df_val=df_val, df_test=df_test, col_prompt=prompt_column, col_answer=answer_column, col_score=target_column, base_model=SBERT_BASE_MODEL, max_num=NPCR_ANSWER_LENGTH, training_with_same_score=True, save_model=False)
                        write_classification_statistics(filepath=run_path_npcr_sbert, y_true=gold, y_pred=npcr_sbert_pred)
                        df_train.to_csv(os.path.join(run_path_npcr_sbert, 'train.csv'))
                        df_val.to_csv(os.path.join(run_path_npcr_sbert, 'val.csv'))
                        df_test.to_csv(os.path.join(run_path_npcr_sbert, 'test.csv'))

                        preds_npcr_sbert = all_predictions.get('NPCR_SBERT', pd.DataFrame())
                        preds_npcr_sbert = pd.concat([preds_npcr_sbert, pd.read_csv(os.path.join(run_path_npcr_sbert, 'preds.csv'))])
                        all_predictions['NPCR_SBERT'] = preds_npcr_sbert


                if run_pretrained:

                    run_path_pretrained = os.path.join(RESULT_PATH_EXP_3 + run_suffix, condition, dataset_name, prompt, test_language, 'pretrained', 'fold_' + str(val_fold))
                    
                    if not os.path.exists(os.path.join(run_path_pretrained, 'preds.csv')):

                        if not os.path.exists(run_path_pretrained):

                            os.makedirs(run_path_pretrained)
                            
                        # Load pretrained model 
                        model = SentenceTransformer(SBERT_BASE_MODEL)
                        df_ref = pd.concat([df_train, df_val])
                        df_ref['embedding'] = df_ref[answer_column].apply(model.encode)
                        df_test['embedding'] = df_test[answer_column].apply(model.encode)

                        # Predict on within-test data
                        gold, pred_max_pretrained, pred_avg_pretrained = eval_sbert(run_path_pretrained, df_test=df_test, df_ref=df_ref, id_column=id_column, answer_column=answer_column, target_column=target_column)
                        write_classification_statistics(filepath=run_path_pretrained, y_true=gold, y_pred=pred_avg_pretrained, suffix='')
                        write_classification_statistics(filepath=run_path_pretrained, y_true=gold, y_pred=pred_max_pretrained, suffix='_max')
                        df_train.to_csv(os.path.join(run_path_pretrained, 'train.csv'))
                        df_val.to_csv(os.path.join(run_path_pretrained, 'val.csv'))
                        df_test.to_csv(os.path.join(run_path_pretrained, 'test.csv'))

                        preds_pretrained = all_predictions.get('pretrained', pd.DataFrame())
                        preds_pretrained = pd.concat([preds_pretrained, pd.read_csv(os.path.join(run_path_pretrained, 'preds.csv'))])
                        all_predictions['pretrained'] = preds_pretrained


            for model, df_preds in all_predictions.items():
                    
                df_preds.to_csv(os.path.join(RESULT_PATH_EXP_3 + run_suffix, condition, dataset_name, prompt, test_language, model, 'preds.csv'))
                df_preds = df_preds.reset_index()
                gold = df_preds[target_column]

                if model == 'SBERT' or model == 'pretrained' or model == 'SBERT_XLMRcore':

                    pred_avg = df_preds['pred_avg']
                    pred_max = df_preds['pred_max']

                    write_classification_statistics(filepath=os.path.join(RESULT_PATH_EXP_3 + run_suffix, condition, dataset_name, prompt, test_language, model), y_true=gold, y_pred=pred_avg, suffix='')
                    write_classification_statistics(filepath=os.path.join(RESULT_PATH_EXP_3 + run_suffix, condition, dataset_name, prompt, test_language, model), y_true=gold, y_pred=pred_max, suffix='_max')
                
                else:
                    pred = df_preds['pred']
                    write_classification_statistics(filepath=os.path.join(RESULT_PATH_EXP_3 + run_suffix, condition, dataset_name, prompt, test_language, model), y_true=gold, y_pred=pred, suffix='')


## Downsampled
for run in ['_RUN1', '_RUN2', '_RUN3']:
    
    for dataset in [ASAP_M]:

        for translate_train in [True, False]:

            run_downsampled(
                dataset_path=dataset['dataset_path'], 
                dataset_name=dataset['dataset_name'], 
                id_column=dataset['id_column'],
                prompt_column=dataset['prompt_column'], 
                answer_column=dataset['answer_column'], 
                target_column=dataset['target_column'], 
                languages=dataset['languages'], 
                run_suffix=run, 
                num_folds=dataset['num_folds'],
                translate_train=translate_train,
                run_xlmr=True,
                run_sbert=True,
                run_npcr_xlmr=True,
                run_npcr_sbert=True,
                run_xlmr_swap_sbert=True,
                run_sbert_swap_xlmr=True,
                run_pretrained=True,
                )


## Full:
for run in ['_RUN1', '_RUN2', '_RUN3']:

    for dataset in [ASAP_M]:
        
        for translate_train in [True, False]:

            run_full(
                dataset_path=dataset['dataset_path'], 
                dataset_name=dataset['dataset_name'], 
                id_column=dataset['id_column'], 
                prompt_column=dataset['prompt_column'], 
                answer_column=dataset['answer_column'], 
                target_column=dataset['target_column'], 
                languages=dataset['languages'], 
                run_suffix=run, 
                num_folds=dataset['num_folds'],
                translate_train=translate_train,
                run_xlmr=True,
                run_sbert=True,
                run_npcr_xlmr=True,
                run_npcr_sbert=True,
                run_xlmr_swap_sbert=True,
                run_sbert_swap_xlmr=True,
                run_pretrained=True,
                )
