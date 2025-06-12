import os
import shutil
import sys
import torch

import pandas as pd

from config import EPIRLS, ASAP_T, ASAP_M, RESULT_PATH_EXP_1, SBERT_BASE_MODEL, XLMR_BASE_MODEL, SBERT_NUM_EPOCHS, BERT_NUM_EPOCHS, SBERT_BATCH_SIZE, BERT_BATCH_SIZE, SBERT_NUM_PAIRS, SBERT_NUM_VAL_PAIRS
from copy import deepcopy
from model_training.train_xlmr import train_xlmr
from model_training.train_sbert import train_sbert
from model_training.utils import read_data, get_device, eval_bert, eval_sbert, write_classification_statistics
from sentence_transformers import SentenceTransformer
from transformers import BertForSequenceClassification, BertTokenizer, XLMRobertaTokenizer, XLMRobertaForSequenceClassification


def run_exp_1(dataset_path, dataset_name, languages, id_column, answer_column, target_column, translate_test, run_sbert=True, run_xlmr=True, run_suffix=''):

    device = get_device()

    for prompt in os.listdir(dataset_path):

        # For each prompt - language pair, train a model
        for language in languages:

            torch.cuda.empty_cache()

            print(prompt, language)

            if run_sbert:

                # Read data for training
                df_train = read_data(os.path.join(dataset_path, prompt, language, 'train.csv'), target_column=target_column, answer_column=answer_column)
                df_val = read_data(os.path.join(dataset_path, prompt, language, 'val.csv'), target_column=target_column, answer_column=answer_column)
                df_test = read_data(os.path.join(dataset_path, prompt, language, 'test.csv'), target_column=target_column, answer_column=answer_column)

                run_path_sbert = os.path.join(RESULT_PATH_EXP_1 + run_suffix, dataset_name, prompt, language, 'SBERT')

                if not os.path.exists(os.path.join(run_path_sbert, 'preds.csv')):

                    gold, pred_max, pred_avg = train_sbert(run_path_sbert, df_train=df_train, df_val=df_val, df_test=df_test, id_column=id_column, answer_column=answer_column, target_column=target_column, base_model=SBERT_BASE_MODEL, num_epochs=SBERT_NUM_EPOCHS, batch_size=SBERT_BATCH_SIZE, save_model=True, num_pairs_per_example=SBERT_NUM_PAIRS, num_val_pairs=SBERT_NUM_VAL_PAIRS)
                    
                    # Eval trained model on within-language data
                    write_classification_statistics(filepath=run_path_sbert, y_true=gold, y_pred=pred_avg, suffix='')
                    write_classification_statistics(filepath=run_path_sbert, y_true=gold, y_pred=pred_max, suffix='_max')

                    # Load model that was just trained 
                    model = SentenceTransformer(os.path.join(run_path_sbert, 'finetuned_model'))
                    model.eval()
                    df_ref = deepcopy(df_train)
                    df_ref['embedding'] = df_ref[answer_column].apply(model.encode)
                    
                    # Zero-shot evaluation of finetuned model on all **other** languages
                    for test_lang in languages:

                        run_path_test_sbert = os.path.join(run_path_sbert, test_lang)

                        if not os.path.exists(run_path_test_sbert):

                            os.mkdir(run_path_test_sbert)
                        
                        df_test_sbert = read_data(os.path.join(dataset_path, prompt, test_lang, 'test.csv'), target_column=target_column, answer_column=answer_column)
                        df_test_sbert['embedding'] = df_test_sbert[answer_column].apply(model.encode)
                        gold, pred_max_test, pred_avg_test = eval_sbert(run_path_test_sbert, df_test=df_test_sbert, df_ref=df_ref, id_column=id_column, answer_column=answer_column, target_column=target_column)

                        write_classification_statistics(filepath=run_path_test_sbert, y_true=gold, y_pred=pred_avg_test, suffix='')
                        write_classification_statistics(filepath=run_path_test_sbert, y_true=gold, y_pred=pred_max_test, suffix='_max')

                        # Evaluate on test data translated into target language
                        if translate_test and test_lang != language:

                            run_path_test_sbert_translated = os.path.join(run_path_sbert, test_lang + '_translated')

                            if not os.path.exists(run_path_test_sbert_translated):

                                os.mkdir(run_path_test_sbert_translated)
                            
                            df_test_sbert_translated = read_data(os.path.join(dataset_path, prompt, test_lang, 'test_translated_m2m_100_1.2B_' + language + '.csv'), target_column=target_column, answer_column=answer_column)
                            df_test_sbert_translated['embedding'] = df_test_sbert_translated[answer_column].apply(model.encode)
                            gold, pred_max_translated, pred_avg_translated = eval_sbert(run_path_test_sbert_translated, df_test=df_test_sbert_translated, df_ref=df_ref, id_column=id_column, answer_column=answer_column, target_column=target_column)

                            write_classification_statistics(filepath=run_path_test_sbert_translated, y_true=gold, y_pred=pred_avg, suffix='')
                            write_classification_statistics(filepath=run_path_test_sbert_translated, y_true=gold, y_pred=pred_max, suffix='_max')

                    shutil.rmtree(os.path.join(run_path_sbert, 'finetuned_model'))


            if run_xlmr:

                df_train = read_data(os.path.join(dataset_path, prompt, language, 'train.csv'), target_column=target_column, answer_column=answer_column)
                df_val = read_data(os.path.join(dataset_path, prompt, language, 'val.csv'), target_column=target_column, answer_column=answer_column)
                df_test = read_data(os.path.join(dataset_path, prompt, language, 'test.csv'), target_column=target_column, answer_column=answer_column)

                run_path_bert = os.path.join(RESULT_PATH_EXP_1 + run_suffix, dataset_name, prompt, language, 'XLMR')

                if not os.path.exists(os.path.join(run_path_bert, 'preds.csv')):

                    gold, xlmr_pred = train_xlmr(run_path_bert, df_train=df_train, df_val=df_val, df_test=df_test, answer_column=answer_column, target_column=target_column, base_model=XLMR_BASE_MODEL, num_epochs=BERT_NUM_EPOCHS, batch_size=BERT_BATCH_SIZE, save_model=True)
                    
                    write_classification_statistics(filepath=run_path_bert, y_true=gold, y_pred=xlmr_pred)
                    
                    bert_model = XLMRobertaForSequenceClassification.from_pretrained(os.path.join(run_path_bert, 'best_model')).to(device)
                    bert_tokenizer = XLMRobertaTokenizer.from_pretrained(os.path.join(run_path_bert, 'best_model'))

                    # Zero-shot evaluation of finetuned model on all **other** languages
                    for test_lang in languages:

                        df_test_bert = read_data(os.path.join(dataset_path, prompt, test_lang, 'test.csv'), target_column=target_column, answer_column=answer_column)
                        gold, xlmr_pred = eval_bert(bert_model, bert_tokenizer, df_test_bert, answer_column=answer_column, target_column=target_column)

                        run_path_test_bert = os.path.join(run_path_bert, test_lang)

                        if not os.path.exists(run_path_test_bert):

                            os.mkdir(run_path_test_bert)

                        df_test_copy = deepcopy(df_test_bert)
                        df_test_copy['pred'] = xlmr_pred
                        df_test_copy.to_csv(os.path.join(run_path_test_bert, 'preds.csv'))

                        write_classification_statistics(filepath=run_path_test_bert, y_true=gold, y_pred=xlmr_pred, suffix='')
                    
                        if translate_test and test_lang != language:

                            df_test_bert_translated = read_data(os.path.join(dataset_path, prompt, test_lang, 'test_translated_m2m_100_1.2B_' + language + '.csv'), target_column=target_column, answer_column=answer_column)
                            gold, xlmr_pred_translated = eval_bert(bert_model, bert_tokenizer, df_test_bert_translated, answer_column=answer_column, target_column=target_column)

                            run_path_test_bert_translated = os.path.join(run_path_bert, test_lang + '_translated')

                            if not os.path.exists(run_path_test_bert_translated):

                                os.mkdir(run_path_test_bert_translated)

                            df_test_translated_copy = deepcopy(df_test_bert_translated)
                            df_test_translated_copy['pred'] = xlmr_pred_translated
                            df_test_translated_copy.to_csv(os.path.join(run_path_test_bert_translated, 'preds.csv'))

                            write_classification_statistics(filepath=run_path_test_bert_translated, y_true=gold, y_pred=xlmr_pred_translated, suffix='')

                    shutil.rmtree(os.path.join(run_path_bert, 'best_model'))



def run_exp_1_cross_validated(dataset_path, dataset_name, languages, id_column, answer_column, target_column, translate_test, num_folds, run_sbert=True, run_xlmr=True, run_suffix=''):

    device = get_device()

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
                    df_train_list=[]

                    for train_fold in train_folds:

                        df_train_list.append(read_data(os.path.join(dataset_path, prompt, language, 'fold_' + str(train_fold) + '.csv'), target_column=target_column, answer_column=answer_column))

                    df_train = pd.concat(df_train_list)
                    df_train.reset_index(inplace=True)
                    df_val = read_data(os.path.join(dataset_path, prompt, language, 'fold_' + str(val_fold) + '.csv'), target_column=target_column, answer_column=answer_column)
                    df_test = read_data(os.path.join(dataset_path, prompt, language, 'fold_' + str(test_fold) + '.csv'), target_column=target_column, answer_column=answer_column)

                    run_path_sbert = os.path.join(RESULT_PATH_EXP_1 + run_suffix, dataset_name, prompt, language, 'SBERT', 'fold_' + str(test_fold))

                    if not os.path.exists(os.path.join(run_path_sbert, 'preds.csv')):

                        gold, pred_max, pred_avg = train_sbert(run_path_sbert, df_train=df_train, df_val=df_val, df_test=df_test, id_column=id_column, answer_column=answer_column, target_column=target_column, base_model=SBERT_BASE_MODEL, num_epochs=SBERT_NUM_EPOCHS, batch_size=SBERT_BATCH_SIZE, save_model=True, num_pairs_per_example=SBERT_NUM_PAIRS, num_val_pairs=SBERT_NUM_VAL_PAIRS)
                        
                        # Eval trained model on within-language data
                        write_classification_statistics(filepath=run_path_sbert, y_true=gold, y_pred=pred_avg, suffix='')
                        write_classification_statistics(filepath=run_path_sbert, y_true=gold, y_pred=pred_max, suffix='_max')

                        # Load model that was just trained 
                        model = SentenceTransformer(os.path.join(run_path_sbert, 'finetuned_model'))

                        # Obtain test predictions
                        model.eval()

                        df_ref = deepcopy(df_train)
                        df_ref['embedding'] = df_ref[answer_column].apply(model.encode)
                        
                        # Zero-shot evaluation of finetuned model on all **other** languages
                        for test_lang in languages:

                            run_path_test_sbert = os.path.join(run_path_sbert, test_lang)

                            if not os.path.exists(run_path_test_sbert):

                                os.mkdir(run_path_test_sbert)
                            
                            df_test_sbert_list = []

                            if test_lang != language:

                                for fold in range(1, num_folds+1):

                                    df_test_sbert_list.append(read_data(os.path.join(dataset_path, prompt, test_lang, 'fold_' + str(fold) + '.csv')), target_column=target_column, answer_column=answer_column)
                                
                            else:

                                df_test_sbert_list.append(read_data(os.path.join(dataset_path, prompt, test_lang, 'fold_' + str(test_fold) + '.csv')), target_column=target_column, answer_column=answer_column)

                            df_test_sbert = pd.concat(df_test_sbert_list)
                            df_test_sbert.reset_index(inplace=True)
                            df_test_sbert['embedding'] = df_test_sbert[answer_column].apply(model.encode)
                            
                            gold, pred_max, pred_avg = eval_sbert(run_path_test_sbert, df_test=df_test_sbert, df_ref=df_ref, id_column=id_column, answer_column=answer_column, target_column=target_column)

                            write_classification_statistics(filepath=run_path_test_sbert, y_true=gold, y_pred=pred_avg, suffix='')
                            write_classification_statistics(filepath=run_path_test_sbert, y_true=gold, y_pred=pred_max, suffix='_max')

                            # Evaluate on test data translated into target language
                            if translate_test and test_lang != language:

                                run_path_test_sbert_translated = os.path.join(run_path_sbert, test_lang + '_translated')

                                if not os.path.exists(run_path_test_sbert_translated):

                                    os.mkdir(run_path_test_sbert_translated)
                                
                                df_test_sbert_translated_list = []
                                
                                for fold in range(1, num_folds+1):

                                    df_test_sbert_translated_list.append(read_data(os.path.join(dataset_path, prompt, test_lang, 'fold_' + str(fold) + '_translated_' + language + '.csv')), target_column=target_column, answer_column=answer_column)
                                
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
                        val_fold=1

                    train_folds = list(range(1, num_folds+1))
                    train_folds.remove(test_fold)
                    train_folds.remove(val_fold)

                    # Read data for training
                    df_train_list=[]

                    for train_fold in train_folds:

                        df_train_list.append(read_data(os.path.join(dataset_path, prompt, language, 'fold_' + str(train_fold) + '.csv'), target_column=target_column, answer_column=answer_column))

                    df_train = pd.concat(df_train_list)
                    df_train.reset_index(inplace=True)
                    df_val = read_data(os.path.join(dataset_path, prompt, language, 'fold_' + str(val_fold) + '.csv'), target_column=target_column, answer_column=answer_column)
                    df_test = read_data(os.path.join(dataset_path, prompt, language, 'fold_' + str(test_fold) + '.csv'), target_column=target_column, answer_column=answer_column)

                    run_path_bert = os.path.join(RESULT_PATH_EXP_1 + run_suffix, dataset_name, prompt, language, 'XLMR', 'fold_' + str(test_fold))

                    if not os.path.exists(os.path.join(run_path_bert, 'preds.csv')):

                        gold, xlmr_pred = train_xlmr(run_path_bert, df_train=df_train, df_val=df_val, df_test=df_test, answer_column=answer_column, target_column=target_column, base_model=XLMR_BASE_MODEL, num_epochs=BERT_NUM_EPOCHS, batch_size=BERT_BATCH_SIZE, save_model=True)
                        
                        write_classification_statistics(filepath=run_path_bert, y_true=gold, y_pred=xlmr_pred)
                        
                        bert_model = XLMRobertaForSequenceClassification.from_pretrained(os.path.join(run_path_bert, 'best_model')).to(device)
                        bert_tokenizer = XLMRobertaTokenizer.from_pretrained(os.path.join(run_path_bert, 'best_model'))

                        # Zero-shot evaluation of finetuned model on all **other** languages
                        for test_lang in languages:

                            df_test_bert_list = []

                            if test_lang != language:

                                for fold in range(1, num_folds):

                                    df_test_bert_list.append(read_data(os.path.join(dataset_path, prompt, test_lang, 'fold_' + str(fold) + '.csv'), target_column=target_column, answer_column=answer_column))
                                
                            else:

                                df_test_bert_list.append(read_data(os.path.join(dataset_path, prompt, test_lang, 'fold_' + str(test_fold) + '.csv'), target_column=target_column, answer_column=answer_column))

                            df_test_bert = pd.concat(df_test_bert_list)
                            df_test_bert.reset_index(inplace=True)
                            gold, xlmr_pred_test = eval_bert(bert_model, bert_tokenizer, df_test_bert, answer_column=answer_column, target_column=target_column)

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

                                    df_test_bert_translated_list.append(read_data(os.path.join(dataset_path, prompt, test_lang, 'fold_' + str(fold) + '_translated_' + language + '.csv'), target_column=target_column, answer_column=answer_column))
                                
                                df_test_bert_translated = pd.concat(df_test_bert_translated_list)
                                df_test_bert_translated.reset_index(inplace=True)
                                gold, xlmr_pred_translated = eval_bert(bert_model, bert_tokenizer, df_test_bert_translated, answer_column=answer_column, target_column=target_column)

                                run_path_test_bert_translated = os.path.join(run_path_bert, test_lang + '_translated')

                                if not os.path.exists(run_path_test_bert_translated):

                                    os.mkdir(run_path_test_bert_translated)

                                df_test_translated_copy = deepcopy(df_test_bert_translated)
                                df_test_translated_copy['pred'] = xlmr_pred_translated
                                df_test_translated_copy.to_csv(os.path.join(run_path_test_bert_translated, 'preds.csv'))

                                write_classification_statistics(filepath=run_path_test_bert_translated, y_true=gold, y_pred=xlmr_pred_translated, suffix='')

                        shutil.rmtree(os.path.join(run_path_bert, 'best_model'))


for run in ['_RUN1']:

    for dataset in [ASAP_T]:
    # for dataset in [EPIRLS, ASAP_T]:

        run_exp_1(
            dataset_path=dataset['dataset_path'], 
            dataset_name=dataset['dataset_name'], 
            id_column=dataset['id_column'], 
            answer_column=dataset['answer_column'], 
            target_column=dataset['target_column'], 
            languages=dataset['languages'], 
            run_sbert=True, 
            run_xlmr=True,
            run_suffix=run, 
            translate_test=dataset['translate_test'],
            )


for run in ['_RUN1']:

    for dataset in [ASAP_M]:

        run_exp_1_cross_validated(
            dataset_path=dataset['dataset_path'], 
            dataset_name=dataset['dataset_name'], 
            id_column=dataset['id_column'], 
            answer_column=dataset['answer_column'], 
            target_column=dataset['target_column'], 
            languages=dataset['languages'], 
            run_sbert=True, 
            run_xlmr=True, 
            run_suffix=run, 
            translate_test=dataset['translate_test'],
            num_folds=dataset['num_folds'],
            )
