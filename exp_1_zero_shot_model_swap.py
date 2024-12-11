import os
import shutil
import sys
import torch

import pandas as pd

from copy import deepcopy
from model_training.sbert_for_classification import SbertForSequenceClassification
from model_training.train_mbert import train_mbert
from model_training.train_xlmr_sbert_core import train_xlmr
from model_training.train_sbert import train_sbert
from sentence_transformers import SentenceTransformer
from transformers import BertForSequenceClassification, BertTokenizer, AutoConfig
from utils import eval_sbert_classification, write_classification_statistics, read_data, eval_sbert

device = 'cpu'
if torch.cuda.is_available():
    device = 'cuda'


sbert_model_name = 'xlm-roberta-base'
sbert_num_epochs = 8
sbert_num_pairs = 25
sbert_num_val_pairs = 1000

# bert_model_name = 'paraphrase-multilingual-MiniLM-L12-v2'
bert_model_name = '/models/paraphrase-multilingual-MiniLM-L12-v2'
bert_num_epochs = 10

results_folder = '/results/exp_1_zero_shot_model_swap'


def run_model_swap(data_path, languages, id_column, answer_column, target_column, dataset, run_sbert=True, run_xlmr=True, run_suffix='', translate_test=False, bert_batch_size=32, sbert_batch_size=64):

    result_dir = os.path.join(results_folder + run_suffix, dataset)

    # prompts = os.listdir(data_path)
    # prompts.remove('E011R02C')
    # prompts.remove('E011R08C')
    # prompts.remove('E011T08C')
    # for prompt in prompts:
    # for prompt in ['E011T08C']:
    for prompt in os.listdir(data_path):

        # For each prompt - language pair, train a model
        # for language in os.listdir(os.path.join(data_path, prompt)):
        for language in languages:

            torch.cuda.empty_cache()

            print(prompt, language)

            if run_sbert:

                # Read data for training
                df_train = read_data(os.path.join(data_path, prompt, language, 'train.csv'), answer_column=answer_column, target_column=target_column)
                df_val = read_data(os.path.join(data_path, prompt, language, 'val.csv'), answer_column=answer_column, target_column=target_column)
                df_test = read_data(os.path.join(data_path, prompt, language, 'test.csv'), answer_column=answer_column, target_column=target_column)

                #  ---------- Train SBERT ------------
                run_path_sbert = os.path.join(result_dir, prompt, language, 'SBERT')
                if not os.path.exists(os.path.join(run_path_sbert, 'preds.csv')):

                    gold, pred_max, pred_avg = train_sbert(run_path_sbert, answer_column=answer_column, id_column=id_column, target_column=target_column, df_train=df_train, df_val=df_val, df_test=df_test, base_model=sbert_model_name, num_epochs=sbert_num_epochs, batch_size=sbert_batch_size, do_warmup=False, save_model=True, num_pairs_per_example=sbert_num_pairs, num_val_pairs=sbert_num_val_pairs)
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
                        
                        df_test_sbert = read_data(os.path.join(data_path, prompt, test_lang, 'test.csv'), answer_column=answer_column, target_column=target_column)
                        df_test_sbert['embedding'] = df_test_sbert[answer_column].apply(model.encode)
                        gold, pred_max, pred_avg = eval_sbert(run_path_test_sbert, df_test_sbert, df_ref, id_column=id_column, answer_column=answer_column, target_column=target_column)

                        write_classification_statistics(filepath=run_path_test_sbert, y_true=gold, y_pred=pred_avg, suffix='')
                        write_classification_statistics(filepath=run_path_test_sbert, y_true=gold, y_pred=pred_max, suffix='_max')

                        if translate_test and test_lang != language:

                            run_path_test_sbert_translated = os.path.join(run_path_sbert, test_lang + '_translated')
                            if not os.path.exists(run_path_test_sbert_translated):
                                os.mkdir(run_path_test_sbert_translated)
                            
                            df_test_sbert_translated = read_data(os.path.join(data_path, prompt, test_lang, 'test_translated_m2m_100_1.2B_' + language + '.csv'), answer_column=answer_column, target_column=target_column)
                            df_test_sbert_translated['embedding'] = df_test_sbert_translated[answer_column].apply(model.encode)
                            gold, pred_max_translated, pred_avg_translated = eval_sbert(run_path_test_sbert_transated, df_test_sbert_translated, df_ref, id_column=id_column, answer_column=answer_column, target_column=target_column)

                            write_classification_statistics(filepath=run_path_test_sbert_translated, y_true=gold, y_pred=pred_avg_translated, suffix='')
                            write_classification_statistics(filepath=run_path_test_sbert_translated, y_true=gold, y_pred=pred_max_translated, suffix='_max')

                    shutil.rmtree(os.path.join(run_path_sbert, 'finetuned_model'))


            if run_xlmr:

                # Read data for training
                df_train = read_data(os.path.join(data_path, prompt, language, 'train.csv'), answer_column=answer_column, target_column=target_column)
                df_val = read_data(os.path.join(data_path, prompt, language, 'val.csv'), answer_column=answer_column, target_column=target_column)
                df_test = read_data(os.path.join(data_path, prompt, language, 'test.csv'), answer_column=answer_column, target_column=target_column)

                # ------------- Train XLMR -------------
                run_path_bert = os.path.join(result_dir, prompt, language, 'XLMR')
                if not os.path.exists(os.path.join(run_path_bert, 'preds.csv')):

                    gold, xlmr_pred = train_xlmr(run_path_bert, df_train=df_train, df_val=df_val, df_test=df_test, answer_column=answer_column, target_column=target_column, num_epochs=bert_num_epochs, batch_size=bert_batch_size, save_model=True, base_model=bert_model_name)
                    
                    write_classification_statistics(filepath=run_path_bert, y_true=gold, y_pred=xlmr_pred)
                    
                    config = AutoConfig.from_pretrained(bert_model_name)
                    config.sbert_path = bert_model_name
                    config.num_labels = len(df_train[target_column].unique())
                    bert_model = SbertForSequenceClassification(config).to(device)
                    # bert_model.load_state_dict(os.path.join(run_path_bert, 'best_model'))
                    # bert_model = torch.load(os.path.join(run_path_bert, 'best_model', 'pytorch_model.bin'))
                    bert_model.load_state_dict(torch.load(os.path.join(run_path_bert, 'best_model', 'pytorch_model.bin')))

                    # Zero-shot evaluation of finetuned model on all **other** languages
                    for test_lang in languages:

                        df_test_bert = read_data(os.path.join(data_path, prompt, test_lang, 'test.csv'), answer_column=answer_column, target_column=target_column)
                        gold, xlmr_pred = eval_sbert_classification(bert_model, df_test_bert, answer_column=answer_column, target_column=target_column)

                        run_path_test_bert = os.path.join(run_path_bert, test_lang)
                        if not os.path.exists(run_path_test_bert):
                            os.mkdir(run_path_test_bert)

                        df_test_copy = deepcopy(df_test_bert)
                        df_test_copy['pred'] = xlmr_pred
                        df_test_copy.to_csv(os.path.join(run_path_test_bert, 'preds.csv'))

                        write_classification_statistics(filepath=run_path_test_bert, y_true=gold, y_pred=xlmr_pred, suffix='')

                        if translate_test and test_lang != language:

                            df_test_bert_translated = read_data(os.path.join(data_path, prompt, test_lang, 'test_translated_m2m_100_1.2B_' + language + '.csv'), answer_column=answer_column, target_column=target_column)
                            gold, xlmr_pred_translated = eval_sbert_classification(bert_model, df_test_bert_translated, answer_column=answer_column, target_column=target_column)

                            run_path_test_bert_translated = os.path.join(run_path_bert, test_lang + '_translated')
                            if not os.path.exists(run_path_test_bert_translated):
                                os.mkdir(run_path_test_bert_translated)

                            df_test_translated_copy = deepcopy(df_test_bert_translated)
                            df_test_translated_copy['pred'] = xlmr_pred_translated
                            df_test_translated_copy.to_csv(os.path.join(run_path_test_bert_translated, 'preds.csv'))

                            write_classification_statistics(filepath=run_path_test_bert_translated, y_true=gold, y_pred=xlmr_pred_translated, suffix='')
                    
                    shutil.rmtree(os.path.join(run_path_bert, 'best_model'))


def run_model_swap_cross_validated(data_path, num_folds, languages, id_column, answer_column, target_column, dataset, run_sbert=True, run_xlmr=True, run_suffix='', translate_test=True, bert_batch_size=32, sbert_batch_size=64):

    result_dir = os.path.join(results_folder + run_suffix, dataset)

    # prompts = os.listdir(data_path)
    # prompts.remove('E011R02C')
    # prompts.remove('E011R08C')
    # prompts.remove('E011T08C')
    # for prompt in prompts:
    # for prompt in ['E011T08C']:
    for prompt in os.listdir(data_path):

        # For each prompt - language pair, train a model
        # for language in os.listdir(os.path.join(data_path, prompt)):
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
                        df_train_list.append(read_data(os.path.join(data_path, prompt, language, 'fold_' + str(train_fold) + '.csv'), answer_column=answer_column, target_column=target_column))
                    df_train = pd.concat(df_train_list)
                    df_train.reset_index(inplace=True)
                    df_val = read_data(os.path.join(data_path, prompt, language, 'fold_' + str(val_fold) + '.csv'), answer_column=answer_column, target_column=target_column)
                    df_test = read_data(os.path.join(data_path, prompt, language, 'fold_' + str(test_fold) + '.csv'), answer_column=answer_column, target_column=target_column)

                    #  ---------- Train SBERT ------------
                    run_path_sbert = os.path.join(result_dir, prompt, language, 'SBERT', 'fold_' + str(test_fold))
                    if not os.path.exists(os.path.join(run_path_sbert, 'preds.csv')):

                        gold, pred_max, pred_avg = train_sbert(run_path_sbert, answer_column=answer_column, id_column=id_column, target_column=target_column, df_train=df_train, df_val=df_val, df_test=df_test, base_model=sbert_model_name, num_epochs=sbert_num_epochs, batch_size=sbert_batch_size, do_warmup=False, save_model=True, num_pairs_per_example=sbert_num_pairs, num_val_pairs=sbert_num_val_pairs)
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
                                    df_test_sbert_list.append(read_data(os.path.join(data_path, prompt, test_lang, 'fold_' + str(fold) + '.csv'), answer_column=answer_column, target_column=target_column))
                            else:
                                df_test_sbert_list.append(read_data(os.path.join(data_path, prompt, test_lang, 'fold_' + str(test_fold) + '.csv'), answer_column=answer_column, target_column=target_column))
                            df_test_sbert = pd.concat(df_test_sbert_list)
                            df_test_sbert.reset_index(inplace=True)
                            df_test_sbert['embedding'] = df_test_sbert[answer_column].apply(model.encode)
                            gold, pred_max, pred_avg = eval_sbert(run_path_test_sbert, df_test_sbert, df_ref, id_column=id_column, answer_column=answer_column, target_column=target_column)

                            write_classification_statistics(filepath=run_path_test_sbert, y_true=gold, y_pred=pred_avg, suffix='')
                            write_classification_statistics(filepath=run_path_test_sbert, y_true=gold, y_pred=pred_max, suffix='_max')

                            if translate_test and test_lang != language:

                                run_path_test_sbert_translated = os.path.join(run_path_sbert, test_lang + '_translated')
                                if not os.path.exists(run_path_test_sbert_translated):
                                    os.mkdir(run_path_test_sbert_translated)
                                
                                df_test_sbert_translated_list = []
                                for fold in range(1, num_folds+1):
                                    df_test_sbert_translated_list.append(read_data(os.path.join(data_path, prompt, test_lang, 'fold_' + str(fold) + '_translated_' + language + '.csv'), answer_column=answer_column, target_column=target_column))
                                df_test_sbert_translated = pd.concat(df_test_sbert_translated_list)
                                df_test_sbert_translated['embedding'] = df_test_sbert_translated[answer_column].apply(model.encode)
                                gold, pred_max_translated, pred_avg_translated = eval_sbert(run_path_test_sbert_translated, df_test_sbert_translated, df_ref, id_column=id_column, answer_column=answer_column, target_column=target_column)

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
                    df_train_list = []
                    for train_fold in train_folds:
                        df_train_list.append(read_data(os.path.join(data_path, prompt, language, 'fold_' + str(train_fold) + '.csv'), answer_column=answer_column, target_column=target_column))
                    df_train = pd.concat(df_train_list)
                    df_train.reset_index(inplace=True)
                    df_val = read_data(os.path.join(data_path, prompt, language, 'fold_' + str(val_fold) + '.csv'), answer_column=answer_column, target_column=target_column)
                    df_test = read_data(os.path.join(data_path, prompt, language, 'fold_' + str(test_fold) + '.csv'), answer_column=answer_column, target_column=target_column)

                    # ------------- Train XLMR -------------
                    run_path_bert = os.path.join(result_dir, prompt, language, 'XLMR', 'fold_' + str(test_fold))
                    if not os.path.exists(os.path.join(run_path_bert, 'preds.csv')):

                        gold, xlmr_pred = train_xlmr(run_path_bert, df_train=df_train, df_val=df_val, df_test=df_test, answer_column=answer_column, target_column=target_column, num_epochs=bert_num_epochs, batch_size=bert_batch_size, save_model=True, base_model=bert_model_name)
                        
                        write_classification_statistics(filepath=run_path_bert, y_true=gold, y_pred=xlmr_pred)
                        
                        config = AutoConfig.from_pretrained(bert_model_name)
                        config.sbert_path = bert_model_name
                        config.num_labels = len(df_train[target_column].unique())
                        bert_model = SbertForSequenceClassification(config).to(device)
                        # bert_model.load_state_dict(os.path.join(run_path_bert, 'best_model'))
                        # bert_model = torch.load(os.path.join(run_path_bert, 'best_model', 'pytorch_model.bin'))
                        bert_model.load_state_dict(torch.load(os.path.join(run_path_bert, 'best_model', 'pytorch_model.bin')))

                        # Zero-shot evaluation of finetuned model on all **other** languages
                        for test_lang in languages:

                            df_test_bert_list = []
                            if test_lang != language:
                                for fold in range(1, num_folds+1):
                                    df_test_bert_list.append(read_data(os.path.join(data_path, prompt, test_lang, 'fold_' + str(fold) + '.csv'), answer_column=answer_column, target_column=target_column))
                            else:
                                df_test_bert_list.append(read_data(os.path.join(data_path, prompt, test_lang, 'fold_' + str(test_fold) + '.csv'), answer_column=answer_column, target_column=target_column))
                            df_test_bert = pd.concat(df_test_bert_list)
                            df_test_bert.reset_index(inplace=True)
                            gold, xlmr_pred = eval_sbert_classification(bert_model, df_test_bert, answer_column=answer_column, target_column=target_column)

                            run_path_test_bert = os.path.join(run_path_bert, test_lang)
                            if not os.path.exists(run_path_test_bert):
                                os.mkdir(run_path_test_bert)

                            df_test_copy = deepcopy(df_test_bert)
                            df_test_copy['pred'] = xlmr_pred
                            df_test_copy.to_csv(os.path.join(run_path_test_bert, 'preds.csv'))

                            write_classification_statistics(filepath=run_path_test_bert, y_true=gold, y_pred=xlmr_pred, suffix='')

                            if translate_test and test_lang != language:

                                df_test_bert_translated_list = []
                                for fold in range(1, num_folds+1):
                                    df_test_bert_translated_list.append(read_data(os.path.join(data_path, prompt, test_lang, 'fold_' + str(fold) + '_translated_' + language + '.csv'), answer_column=answer_column, target_column=target_column))
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
        

## ePIRLS
# languages = ['ar', 'da', 'en', 'he', 'it', 'ka', 'nb', 'pt', 'sl', 'sv', 'zh']
# run_model_swap(data_path='/data/exp', id_column='id', answer_column='Value', target_column='score', languages=languages, run_sbert=True, run_xlmr=True, dataset='ePIRLS', run_suffix='_test', translate_test=True)

## Run ASAP (translated)
languages = ['ar', 'da', 'en', 'he', 'it', 'ka', 'nb', 'pt', 'sl', 'sv', 'zh']
# run_model_swap(data_path='/data/ASAP/split', id_column='ItemId', answer_column='AnswerText', target_column='Score1', languages=languages, dataset='ASAP_translated', run_suffix='_RUN1', run_sbert=False, sbert_batch_size=16)
run_model_swap(data_path='/data/ASAP/split', id_column='ItemId', answer_column='AnswerText', target_column='Score1', languages=languages, dataset='ASAP_translated', run_suffix='_RUN1', run_xlmr=False, sbert_batch_size=16)

## Run ASAP (multilingual)
# languages = ['de', 'en', 'es', 'fr', 'zh']
# run_model_swap_cross_validated(data_path='/data/ASAP_crosslingual/split', id_column='id', num_folds=7, answer_column='text', target_column='score', languages=languages, dataset='ASAP_crosslingual', run_suffix='_RUN1', sbert_batch_size=16)