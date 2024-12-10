import pandas as pd
from train_NPCR import train_model
from transformers import BertForSequenceClassification, BertTokenizer, XLMRobertaTokenizer, XLMRobertaForSequenceClassification
from sentence_transformers import SentenceTransformer
from utils import eval_bert, write_classification_statistics, read_data
from npcr.evaluator_core import evaluate_finetuned_model
from copy import deepcopy
import os
import sys
import torch
import shutil


def run_dataset(data_path, prompt_id_column, answer_column, target_column, languages, dataset, run_suffix='', translate_test=False):

    device = 'cpu'
    if torch.cuda.is_available():
        device = 'cuda'

    sbert_model_name = 'paraphrase-multilingual-MiniLM-L12-v2'
    bert_model_name = 'xlm-roberta-base'

    result_dir = os.path.join('/results/exp_1_zero_shot_NPCR' + run_suffix, dataset)

    for prompt in os.listdir(data_path):

        # For each prompt - language pair, train a model
        for language in languages:

            torch.cuda.empty_cache()

            print(prompt, language)

            # Read data for training
            df_train = read_data(os.path.join(data_path, prompt, language, 'train.csv'), answer_column=answer_column, target_column=target_column)
            # Must shuffle! Otherwise training pairs are built to form almost exclusively with similarity label 0
            df_train = df_train.sample(frac=1, random_state=7542).reset_index(drop=True)
            df_val = read_data(os.path.join(data_path, prompt, language, 'val.csv'), answer_column=answer_column, target_column=target_column)
            df_test = read_data(os.path.join(data_path, prompt, language, 'test.csv'), answer_column=answer_column, target_column=target_column)

            for model in [(sbert_model_name, 'SBERT'), (bert_model_name, 'XLMR')]:

                base_model = model[0]
                model_name = model[1]

                run_path = os.path.join(result_dir, prompt, language, model_name)
                if not os.path.exists(os.path.join(run_path, 'preds.csv')):
                    
                    gold, pred = train_model(target_path=run_path, df_train=df_train, df_val=df_val, df_test=df_test, col_prompt=prompt_id_column, col_answer=answer_column, col_score=target_column, base_model=base_model, max_num=128, training_with_same_score=True)
                    write_classification_statistics(filepath=run_path, y_true=gold, y_pred=pred)

                    # Zero-shot evaluation of finetuned model on all **other** languages
                    for test_lang in languages:

                        run_path_test = os.path.join(run_path, test_lang)
                        if not os.path.exists(run_path_test):
                            os.mkdir(run_path_test)

                        df_test_other = read_data(os.path.join(data_path, prompt, test_lang, 'test.csv'), answer_column=answer_column, target_column=target_column)
                        gold, pred_test = evaluate_finetuned_model(model_path=os.path.join(run_path, 'best_model'), base_model=base_model, df_ref=df_train, df_test=df_test_other, col_prompt=prompt_id_column, col_answer=answer_column, col_score=target_column, target_path=run_path_test, max_num=128, suffix='_' + str(test_lang))

                        df_test_copy = deepcopy(df_test_other)
                        df_test_copy['pred'] = pred_test
                        df_test_copy.to_csv(os.path.join(run_path_test, 'preds.csv'))

                        write_classification_statistics(filepath=run_path_test, y_true=gold, y_pred=pred_test, suffix='')

                        if translate_test and (test_lang != language):

                            run_path_test_translated = os.path.join(run_path, test_lang + 'translated')
                            if not os.path.exists(run_path_test_translated):
                                os.mkdir(run_path_test_translated)

                            df_test_other_translated = read_data(os.path.join(data_path, prompt, test_lang, 'test_translated_m2m_100_1.2B_' + language + '.csv'), answer_column=answer_column, target_column=target_column)
                            gold, pred_test_translated = evaluate_finetuned_model(model_path=os.path.join(run_path, 'best_model'), base_model=base_model, df_ref=df_train, df_test=df_test_other_translated, col_prompt=prompt_id_column, col_answer=answer_column, col_score=target_column, target_path=run_path_test_translated, max_num=128, suffix='_' + str(test_lang) + '_translated')

                            df_test_translated_copy = deepcopy(df_test_other_translated)
                            df_test_translated_copy['pred'] = pred_test_translated
                            df_test_translated_copy.to_csv(os.path.join(run_path_test_translated, 'preds.csv'))

                            write_classification_statistics(filepath=run_path_test_translated, y_true=gold, y_pred=pred_test_translated, suffix='')

                    # Delete model
                    if os.path.exists(os.path.join(run_path, 'best_model')):
                        os.remove(os.path.join(run_path, 'best_model'))

                else:
                    print('Skipping prompt ' + str(prompt) + ' because it already ran!')



def run_dataset_folds(data_path, prompt_id_column, answer_column, target_column, languages, dataset, num_folds, run_suffix='', translate_test=True):

    device = 'cpu'
    if torch.cuda.is_available():
        device = 'cuda'

    sbert_model_name = 'paraphrase-multilingual-MiniLM-L12-v2'
    bert_model_name = 'xlm-roberta-base'

    result_dir = os.path.join('/results/exp_1_zero_shot_NPCR' + run_suffix, dataset)

    for prompt in os.listdir(data_path):

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
                   df_train_list.append(read_data(os.path.join(data_path, prompt, language, 'fold_' + str(train_fold) + '.csv'), answer_column=answer_column, target_column=target_column))
                df_train = pd.concat(df_train_list)
                # Must shuffle! Otherwise training pairs are built to form almost exclusively with similarity label 0
                df_train = df_train.sample(frac=1, random_state=7542).reset_index(drop=True)
                df_val = read_data(os.path.join(data_path, prompt, language, 'fold_' + str(val_fold) + '.csv'), answer_column=answer_column, target_column=target_column)
                df_test = read_data(os.path.join(data_path, prompt, language, 'fold_' + str(test_fold) + '.csv'), answer_column=answer_column, target_column=target_column)

                # Run SBERT
                for model in [(sbert_model_name, 'SBERT'), (bert_model_name, 'XLMR')]:

                    base_model = model[0]
                    model_name = model[1]

                    run_path = os.path.join(result_dir, prompt, language, model_name, str(test_fold))
                    if not os.path.exists(os.path.join(run_path, 'preds.csv')):
                        
                        gold, pred = train_model(target_path=run_path, df_train=df_train, df_val=df_val, df_test=df_test, col_prompt=prompt_id_column, col_answer=answer_column, col_score=target_column, base_model=base_model, max_num=128, training_with_same_score=True)
                        write_classification_statistics(filepath=run_path, y_true=gold, y_pred=pred)

                        # Zero-shot evaluation of finetuned model on all **other** languages
                        for test_lang in languages:

                            run_path_test = os.path.join(run_path, test_lang)
                            if not os.path.exists(run_path_test):
                                os.mkdir(run_path_test)

                            df_test_other_list = []
                            if test_lang != language:
                                for fold in range(1, num_folds + 1):
                                    df_test_other_list.append(read_data(os.path.join(data_path, prompt, test_lang, 'fold_' + str(fold) + '.csv'), answer_column=answer_column, target_column=target_column))
                            else:
                                df_test_other_list.append(read_data(os.path.join(data_path, prompt, test_lang, 'fold_' + str(test_fold) + '.csv'), answer_column=answer_column, target_column=target_column))

                            df_test_other = pd.concat(df_test_other_list)
                            print(list(df_test_other.index))
                            print('++++')
                            print('++++ Evaluatiion', str(prompt), str(language), str(test_fold), str(test_lang))
                            print('++++')
                            gold, pred_test = evaluate_finetuned_model(model_path=os.path.join(run_path, 'best_model'), base_model=base_model, df_ref=df_train, df_test=df_test_other, col_prompt=prompt_id_column, col_answer=answer_column, col_score=target_column, target_path=run_path_test, max_num=128, suffix='_' + str(test_lang))

                            df_test_copy = deepcopy(df_test_other)
                            df_test_copy['pred'] = pred_test
                            df_test_copy.to_csv(os.path.join(run_path_test, 'preds.csv'))

                            write_classification_statistics(filepath=run_path_test, y_true=gold, y_pred=pred_test, suffix='')

                            if translate_test and (test_lang != language):

                                run_path_test_translated = os.path.join(run_path, test_lang + 'translated')
                                if not os.path.exists(run_path_test_translated):
                                    os.mkdir(run_path_test_translated)

                                df_test_other_translated_list = []
                                for fold in range(1, num_folds + 1):
                                    df_test_other_translated_list.append(read_data(os.path.join(data_path, prompt, test_lang, 'fold_' + str(fold) + '_translated_' + language + '.csv'), answer_column=answer_column, target_column=target_column))
                                                                
                                df_test_other_translated = pd.concat(df_test_other_translated_list)
                                gold, pred_test_translated = evaluate_finetuned_model(model_path=os.path.join(run_path, 'best_model'), base_model=base_model, df_ref=df_train, df_test=df_test_other_translated, col_prompt=prompt_id_column, col_answer=answer_column, col_score=target_column, target_path=run_path_test_translated, max_num=128, suffix='_' + str(test_lang) + '_translated')

                                df_test_translated_copy = deepcopy(df_test_other_translated)
                                df_test_translated_copy['pred'] = pred_test_translated
                                df_test_translated_copy.to_csv(os.path.join(run_path_test_translated, 'preds.csv'))

                                write_classification_statistics(filepath=run_path_test_translated, y_true=gold, y_pred=pred_test_translated, suffix='')

                        # Delete model
                        if os.path.exists(os.path.join(run_path, 'best_model')):
                            os.remove(os.path.join(run_path, 'best_model'))

                    else:
                        print('Skipping prompt ' + str(prompt) + ' because it already ran!')



## Run ePIRLS
# languages = ['ar', 'da', 'en', 'he', 'it', 'ka', 'nb', 'pt', 'sl', 'sv', 'zh']
# run_dataset(data_path='/data/exp', prompt_id_column='Variable', answer_column='Value', target_column='score', languages=languages, dataset='ePIRLS', run_suffix='', translate_test=True)

## Run ASAP (translated)
# languages = ['ar', 'da', 'en', 'he', 'it', 'ka', 'nb', 'pt', 'sl', 'sv', 'zh']
# run_dataset(data_path='/data/ASAP/split', prompt_id_column='PromptId', answer_column='AnswerText', target_column='Score1', languages=languages, dataset='ASAP_translated', run_suffix='_test')

## Run ASAP (multilingual)
languages = ['de', 'en', 'es', 'fr', 'zh']
run_dataset_folds(data_path='/data/ASAP_crosslingual/split', prompt_id_column='prompt', answer_column='text', target_column='score', languages=languages, num_folds=7, dataset='ASAP_crosslingual', run_suffix='_test')