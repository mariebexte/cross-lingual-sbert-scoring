import os
import shutil
import sys
import torch

import pandas as pd

from config import EPIRLS, ASAP_M, ASAP_T, SBERT_BASE_MODEL, XLMR_BASE_MODEL, SBERT_NUM_EPOCHS, BERT_NUM_EPOCHS, SBERT_NUM_PAIRS, SBERT_NUM_VAL_PAIRS, RESULT_PATH_EXP_1
from copy import deepcopy
from model_training.sbert_for_classification import SbertForSequenceClassification
from model_training.train_xlmr_sbert_core import train_xlmr
from model_training.train_sbert import train_sbert
from model_training.utils import read_data, get_device, eval_sbert_classification, eval_sbert, write_classification_statistics
from sentence_transformers import SentenceTransformer
from transformers import AutoConfig


def run_model_swap(dataset_path, dataset_name, languages, id_column, answer_column, target_column, translate_test, run_sbert=True, run_xlmr=True, run_suffix='', bert_batch_size=32, sbert_batch_size=64):

    device = get_device()

    result_dir = os.path.join(RESULT_PATH_EXP_1 + run_suffix, dataset_name)

    for prompt in os.listdir(dataset_path):

        # For each prompt - language pair, train a model
        for language in languages:

            torch.cuda.empty_cache()

            print(prompt, language)

            if run_sbert:

                df_train = read_data(os.path.join(dataset_path, prompt, language, 'train.csv'), answer_column=answer_column, target_column=target_column)
                df_val = read_data(os.path.join(dataset_path, prompt, language, 'val.csv'), answer_column=answer_column, target_column=target_column)
                df_test = read_data(os.path.join(dataset_path, prompt, language, 'test.csv'), answer_column=answer_column, target_column=target_column)

                run_path_sbert = os.path.join(result_dir, prompt, language, 'SBERT_XLMRcore')

                if not os.path.exists(os.path.join(run_path_sbert, 'preds.csv')):

                    gold, pred_max, pred_avg = train_sbert(run_path_sbert, answer_column=answer_column, id_column=id_column, target_column=target_column, df_train=df_train, df_val=df_val, df_test=df_test, base_model=XLMR_BASE_MODEL, num_epochs=SBERT_NUM_EPOCHS, batch_size=sbert_batch_size, do_warmup=False, save_model=True, num_pairs_per_example=SBERT_NUM_PAIRS, num_val_pairs=SBERT_NUM_VAL_PAIRS)
                    
                    # Eval trained model on within-language data
                    write_classification_statistics(filepath=run_path_sbert, y_true=gold, y_pred=pred_avg, suffix='')
                    write_classification_statistics(filepath=run_path_sbert, y_true=gold, y_pred=pred_max, suffix='_max')

                    # Load model that was just trained 
                    model = SentenceTransformer(os.path.join(run_path_sbert, 'finetuned_model'))
                    df_ref = pd.concat([df_train, df_val])
                    df_ref['embedding'] = df_ref[answer_column].apply(model.encode)
                    
                    # Zero-shot evaluation of finetuned model on all **other** languages
                    for test_lang in languages:

                        run_path_test_sbert = os.path.join(run_path_sbert, test_lang)

                        if not os.path.exists(run_path_test_sbert):

                            os.mkdir(run_path_test_sbert)
                        
                        df_test_sbert = read_data(os.path.join(dataset_path, prompt, test_lang, 'test.csv'), answer_column=answer_column, target_column=target_column)
                        df_test_sbert['embedding'] = df_test_sbert[answer_column].apply(model.encode)
                        gold, pred_max_test, pred_avg_test = eval_sbert(run_path_test_sbert, df_test=df_test_sbert, df_ref=df_ref, id_column=id_column, answer_column=answer_column, target_column=target_column)

                        write_classification_statistics(filepath=run_path_test_sbert, y_true=gold, y_pred=pred_avg_test, suffix='')
                        write_classification_statistics(filepath=run_path_test_sbert, y_true=gold, y_pred=pred_max_test, suffix='_max')

                        if translate_test and test_lang != language:

                            run_path_test_sbert_translated = os.path.join(run_path_sbert, test_lang + '_translated')

                            if not os.path.exists(run_path_test_sbert_translated):

                                os.mkdir(run_path_test_sbert_translated)
                            
                            df_test_sbert_translated = read_data(os.path.join(dataset_path, prompt, test_lang, 'test_translated_m2m_100_1.2B_' + language + '.csv'), answer_column=answer_column, target_column=target_column)
                            df_test_sbert_translated['embedding'] = df_test_sbert_translated[answer_column].apply(model.encode)
                            gold, pred_max_translated, pred_avg_translated = eval_sbert(run_path_test_sbert_translated, df_test=df_test_sbert_translated, df_ref=df_ref, id_column=id_column, answer_column=answer_column, target_column=target_column)

                            write_classification_statistics(filepath=run_path_test_sbert_translated, y_true=gold, y_pred=pred_avg_translated, suffix='')
                            write_classification_statistics(filepath=run_path_test_sbert_translated, y_true=gold, y_pred=pred_max_translated, suffix='_max')

                    shutil.rmtree(os.path.join(run_path_sbert, 'finetuned_model'))


            if run_xlmr:

                df_train = read_data(os.path.join(dataset_path, prompt, language, 'train.csv'), answer_column=answer_column, target_column=target_column)
                df_val = read_data(os.path.join(dataset_path, prompt, language, 'val.csv'), answer_column=answer_column, target_column=target_column)
                df_test = read_data(os.path.join(dataset_path, prompt, language, 'test.csv'), answer_column=answer_column, target_column=target_column)

                run_path_bert = os.path.join(result_dir, prompt, language, 'XLMR_SBERTcore')

                if not os.path.exists(os.path.join(run_path_bert, 'preds.csv')):

                    gold, xlmr_pred = train_xlmr(run_path_bert, df_train=df_train, df_val=df_val, df_test=df_test, answer_column=answer_column, target_column=target_column, num_epochs=BERT_NUM_EPOCHS, batch_size=bert_batch_size, save_model=True, base_model='/models/'+SBERT_BASE_MODEL)
                    
                    write_classification_statistics(filepath=run_path_bert, y_true=gold, y_pred=xlmr_pred)
                    
                    config = AutoConfig.from_pretrained('/models/'+SBERT_BASE_MODEL)
                    config.sbert_path = '/models/'+SBERT_BASE_MODEL
                    config.num_labels = len(df_train[target_column].unique())
                    bert_model = SbertForSequenceClassification(config).to(device)
                    bert_model.load_state_dict(torch.load(os.path.join(run_path_bert, 'best_model', 'pytorch_model.bin')))

                    # Zero-shot evaluation of finetuned model on all **other** languages
                    for test_lang in languages:

                        df_test_bert = read_data(os.path.join(dataset_path, prompt, test_lang, 'test.csv'), answer_column=answer_column, target_column=target_column)
                        gold, xlmr_pred_test = eval_sbert_classification(bert_model, df_test_bert, answer_column=answer_column, target_column=target_column)

                        run_path_test_bert = os.path.join(run_path_bert, test_lang)

                        if not os.path.exists(run_path_test_bert):

                            os.mkdir(run_path_test_bert)

                        df_test_copy = deepcopy(df_test_bert)
                        df_test_copy['pred'] = xlmr_pred_test
                        df_test_copy.to_csv(os.path.join(run_path_test_bert, 'preds.csv'))

                        write_classification_statistics(filepath=run_path_test_bert, y_true=gold, y_pred=xlmr_pred_test, suffix='')

                        if translate_test and test_lang != language:

                            df_test_bert_translated = read_data(os.path.join(dataset_path, prompt, test_lang, 'test_translated_m2m_100_1.2B_' + language + '.csv'), answer_column=answer_column, target_column=target_column)
                            gold, xlmr_pred_translated = eval_sbert_classification(bert_model, df_test_bert_translated, answer_column=answer_column, target_column=target_column)

                            run_path_test_bert_translated = os.path.join(run_path_bert, test_lang + '_translated')

                            if not os.path.exists(run_path_test_bert_translated):

                                os.mkdir(run_path_test_bert_translated)

                            df_test_translated_copy = deepcopy(df_test_bert_translated)
                            df_test_translated_copy['pred'] = xlmr_pred_translated
                            df_test_translated_copy.to_csv(os.path.join(run_path_test_bert_translated, 'preds.csv'))

                            write_classification_statistics(filepath=run_path_test_bert_translated, y_true=gold, y_pred=xlmr_pred_translated, suffix='')
                    
                    shutil.rmtree(os.path.join(run_path_bert, 'best_model'))


def run_model_swap_cross_validated(dataset_path, dataset_name, languages, id_column, answer_column, target_column, translate_test, num_folds, run_sbert=True, run_xlmr=True, run_suffix='', bert_batch_size=32, sbert_batch_size=64):

    device = get_device()
    
    result_dir = os.path.join(RESULT_PATH_EXP_1 + run_suffix, dataset_name)

    for prompt in os.listdir(dataset_path):

        # For each prompt - language pair, train a model
        for language in languages:

            torch.cuda.empty_cache()

            print(prompt, language)

            if run_sbert:

                for test_fold in range(1, num_folds+1):

                    val_fold = test_fold+1
                    if val_fold > num_folds:
                        val_fold=1
                    
                    train_folds = list(range(1, num_folds+1))
                    train_folds.remove(test_fold)
                    train_folds.remove(val_fold)

                    # Read data for training
                    df_train_list = []

                    for train_fold in train_folds:

                        df_train_list.append(read_data(os.path.join(dataset_path, prompt, language, 'fold_' + str(train_fold) + '.csv'), answer_column=answer_column, target_column=target_column))
                    
                    df_train = pd.concat(df_train_list)
                    df_train.reset_index(inplace=True)
                    df_val = read_data(os.path.join(dataset_path, prompt, language, 'fold_' + str(val_fold) + '.csv'), answer_column=answer_column, target_column=target_column)
                    df_test = read_data(os.path.join(dataset_path, prompt, language, 'fold_' + str(test_fold) + '.csv'), answer_column=answer_column, target_column=target_column)

                    run_path_sbert = os.path.join(result_dir, prompt, language, 'SBERT_XLMRcore', 'fold_' + str(test_fold))

                    if not os.path.exists(os.path.join(run_path_sbert, 'preds.csv')):

                        gold, pred_max, pred_avg = train_sbert(run_path_sbert, answer_column=answer_column, id_column=id_column, target_column=target_column, df_train=df_train, df_val=df_val, df_test=df_test, base_model=XLMR_BASE_MODEL, num_epochs=SBERT_NUM_EPOCHS, batch_size=sbert_batch_size, do_warmup=False, save_model=True, num_pairs_per_example=SBERT_NUM_PAIRS, num_val_pairs=SBERT_NUM_VAL_PAIRS)
                        
                        # Eval trained model on within-language data
                        write_classification_statistics(filepath=run_path_sbert, y_true=gold, y_pred=pred_avg, suffix='')
                        write_classification_statistics(filepath=run_path_sbert, y_true=gold, y_pred=pred_max, suffix='_max')

                        # Load model that was just trained 
                        model = SentenceTransformer(os.path.join(run_path_sbert, 'finetuned_model'))
                        df_ref = pd.concat([df_train, df_val])
                        df_ref['embedding'] = df_ref[answer_column].apply(model.encode)
                        
                        # Zero-shot evaluation of finetuned model on all **other** languages
                        for test_lang in languages:

                            run_path_test_sbert = os.path.join(run_path_sbert, test_lang)

                            if not os.path.exists(run_path_test_sbert):

                                os.mkdir(run_path_test_sbert)
                            
                            df_test_sbert_list = []

                            if test_lang != language:

                                for fold in range(1, num_folds+1):

                                    df_test_sbert_list.append(read_data(os.path.join(dataset_path, prompt, test_lang, 'fold_' + str(fold) + '.csv'), answer_column=answer_column, target_column=target_column))
                            
                            else:

                                df_test_sbert_list.append(read_data(os.path.join(dataset_path, prompt, test_lang, 'fold_' + str(test_fold) + '.csv'), answer_column=answer_column, target_column=target_column))
                            
                            df_test_sbert = pd.concat(df_test_sbert_list)
                            df_test_sbert.reset_index(inplace=True)
                            df_test_sbert['embedding'] = df_test_sbert[answer_column].apply(model.encode)
                            gold, pred_max, pred_avg = eval_sbert(run_path_test_sbert, df_test=df_test_sbert, df_ref=df_ref, id_column=id_column, answer_column=answer_column, target_column=target_column)

                            write_classification_statistics(filepath=run_path_test_sbert, y_true=gold, y_pred=pred_avg, suffix='')
                            write_classification_statistics(filepath=run_path_test_sbert, y_true=gold, y_pred=pred_max, suffix='_max')

                            if translate_test and test_lang != language:

                                run_path_test_sbert_translated = os.path.join(run_path_sbert, test_lang + '_translated')

                                if not os.path.exists(run_path_test_sbert_translated):

                                    os.mkdir(run_path_test_sbert_translated)
                                
                                df_test_sbert_translated_list = []

                                for fold in range(1, num_folds+1):

                                    df_test_sbert_translated_list.append(read_data(os.path.join(dataset_path, prompt, test_lang, 'fold_' + str(fold) + '_translated_' + language + '.csv'), answer_column=answer_column, target_column=target_column))
                                
                                df_test_sbert_translated = pd.concat(df_test_sbert_translated_list)
                                df_test_sbert_translated.reset_index(inplace=True)
                                df_test_sbert_translated['embedding'] = df_test_sbert_translated[answer_column].apply(model.encode)
                                gold, pred_max_translated, pred_avg_translated = eval_sbert(run_path_test_sbert_translated, df_test=df_test_sbert_translated, df_ref=df_ref, id_column=id_column, answer_column=answer_column, target_column=target_column)

                                write_classification_statistics(filepath=run_path_test_sbert_translated, y_true=gold, y_pred=pred_avg_translated, suffix='')
                                write_classification_statistics(filepath=run_path_test_sbert_translated, y_true=gold, y_pred=pred_max_translated, suffix='_max')

                        shutil.rmtree(os.path.join(run_path_sbert, 'finetuned_model'))


            if run_xlmr:

                for test_fold in range(1, num_folds+1):

                    val_fold = test_fold+1

                    if val_fold > num_folds:

                        val_fold = 1
                    
                    train_folds = list(range(1, num_folds+1))
                    train_folds.remove(test_fold)
                    train_folds.remove(val_fold)

                    # Read data for training
                    df_train_list = []

                    for train_fold in train_folds:

                        df_train_list.append(read_data(os.path.join(dataset_path, prompt, language, 'fold_' + str(train_fold) + '.csv'), answer_column=answer_column, target_column=target_column))
                    
                    df_train = pd.concat(df_train_list)
                    df_train.reset_index(inplace=True)
                    df_val = read_data(os.path.join(dataset_path, prompt, language, 'fold_' + str(val_fold) + '.csv'), answer_column=answer_column, target_column=target_column)
                    df_test = read_data(os.path.join(dataset_path, prompt, language, 'fold_' + str(test_fold) + '.csv'), answer_column=answer_column, target_column=target_column)

                    run_path_bert = os.path.join(result_dir, prompt, language, 'XLMR_SBERTcore', 'fold_' + str(test_fold))

                    if not os.path.exists(os.path.join(run_path_bert, 'preds.csv')):

                        gold, xlmr_pred = train_xlmr(run_path_bert, df_train=df_train, df_val=df_val, df_test=df_test, answer_column=answer_column, target_column=target_column, num_epochs=BERT_NUM_EPOCHS, batch_size=bert_batch_size, save_model=True, base_model='/models/'+SBERT_BASE_MODEL)
                        
                        write_classification_statistics(filepath=run_path_bert, y_true=gold, y_pred=xlmr_pred)
                        
                        config = AutoConfig.from_pretrained('/models/'+SBERT_BASE_MODEL)
                        config.sbert_path = '/models/'+SBERT_BASE_MODEL
                        config.num_labels = len(df_train[target_column].unique())
                        bert_model = SbertForSequenceClassification(config).to(device)
                        bert_model.load_state_dict(torch.load(os.path.join(run_path_bert, 'best_model', 'pytorch_model.bin')))

                        # Zero-shot evaluation of finetuned model on all **other** languages
                        for test_lang in languages:

                            df_test_bert_list = []

                            if test_lang != language:

                                for fold in range(1, num_folds+1):

                                    df_test_bert_list.append(read_data(os.path.join(dataset_path, prompt, test_lang, 'fold_' + str(fold) + '.csv'), answer_column=answer_column, target_column=target_column))
                            else:

                                df_test_bert_list.append(read_data(os.path.join(dataset_path, prompt, test_lang, 'fold_' + str(test_fold) + '.csv'), answer_column=answer_column, target_column=target_column))
                            
                            df_test_bert = pd.concat(df_test_bert_list)
                            df_test_bert.reset_index(inplace=True)
                            gold, xlmr_pred_test = eval_sbert_classification(bert_model, df_test_bert, answer_column=answer_column, target_column=target_column)

                            run_path_test_bert = os.path.join(run_path_bert, test_lang)

                            if not os.path.exists(run_path_test_bert):

                                os.mkdir(run_path_test_bert)

                            df_test_copy = deepcopy(df_test_bert)
                            df_test_copy['pred'] = xlmr_pred_test
                            df_test_copy.to_csv(os.path.join(run_path_test_bert, 'preds.csv'))

                            write_classification_statistics(filepath=run_path_test_bert, y_true=gold, y_pred=xlmr_pred_test, suffix='')

                            if translate_test and test_lang != language:

                                df_test_bert_translated_list = []

                                for fold in range(1, num_folds+1):

                                    df_test_bert_translated_list.append(read_data(os.path.join(dataset_path, prompt, test_lang, 'fold_' + str(fold) + '_translated_' + language + '.csv'), answer_column=answer_column, target_column=target_column))
                                
                                df_test_bert_translated = pd.concat(df_test_bert_translated_list)
                                df_test_bert_translated.reset_index(inplace=True)
                                gold, xlmr_pred_translated = eval_sbert_classification(bert_model, df_test_bert_translated, answer_column=answer_column, target_column=target_column)

                                run_path_test_bert_translated = os.path.join(run_path_bert, test_lang + '_translated')

                                if not os.path.exists(run_path_test_bert_translated):

                                    os.mkdir(run_path_test_bert_translated)

                                df_test_translated_copy = deepcopy(df_test_bert_translated)
                                df_test_translated_copy['pred'] = xlmr_pred_translated
                                df_test_translated_copy.to_csv(os.path.join(run_path_test_bert_translated, 'preds.csv'))

                                write_classification_statistics(filepath=run_path_test_bert_translated, y_true=gold, y_pred=xlmr_pred_translated, suffix='')
                        
                        shutil.rmtree(os.path.join(run_path_bert, 'best_model'))
        

for run in ['_RUN1']:
# for run in ['_RUN1', '_RUN2', '_RUN3']:

    for dataset in [ASAP_T]:
    # for dataset in [EPIRLS, ASAP_T]:

        sbert_batch_size = 64
        bert_batch_size = 32

        if 'ASAP' in dataset['dataset_name']:

            sbert_batch_size = 64
            bert_batch_size = 32

        run_model_swap(
            dataset_path=dataset['dataset_path'], 
            dataset_name=dataset['dataset_name'], 
            id_column=dataset['id_column'], 
            answer_column=dataset['answer_column'], 
            target_column=dataset['target_column'], 
            languages=dataset['languages'], 
            run_sbert=False, 
            run_xlmr=True, 
            run_suffix=run, 
            translate_test=dataset['translate_test'],
            sbert_batch_size=sbert_batch_size,
            bert_batch_size=bert_batch_size
            )


# for run in ['_RUN1', '_RUN2', '_RUN3']:

#     for dataset in [ASAP_M]:

#         run_model_swap_cross_validated(
#             dataset_path=dataset['dataset_path'], 
#             dataset_name=dataset['dataset_name'], 
#             id_column=dataset['id_column'], 
#             answer_column=dataset['answer_column'], 
#             target_column=dataset['target_column'], 
#             languages=dataset['languages'], 
#             run_sbert=True, 
#             run_xlmr=True, 
#             run_suffix=run, 
#             translate_test=dataset['translate_test'],
#             num_folds=dataset['num_folds'],
#             sbert_batch_size=64,
#             bert_batch_size=32
#             )
