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


def run_dataset(data_path, prompt_id_column, answer_column, target_column, languages, run_suffix=''):

    device = 'cpu'
    if torch.cuda.is_available():
        device = 'cuda'

    sbert_model_name = 'paraphrase-multilingual-MiniLM-L12-v2'
    bert_model_name = 'xlm-roberta-base'

    result_dir = '/results/exp_1_zero_shot_NPCR' + run_suffix

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

            # Run SBERT
            # for model in [('bert-base-uncased', 'BERT')]:
            # for model in [(sbert_model_name, 'SBERT')]:
            for model in [(bert_model_name, 'XLMR')]:
            # for model in [(sbert_model_name, 'SBERT'), (bert_model_name, 'XLMR')]:

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


                    # Delete model
                    if os.path.exists(os.path.join(run_path, 'best_model')):
                        os.remove(os.path.join(run_path, 'best_model'))

                else:
                    print('Skipping prompt ' + str(prompt) + ' because it already ran!')



## Run ePIRLS
# languages = ['ar', 'da', 'en', 'he', 'it', 'ka', 'nb', 'pt', 'sl', 'sv', 'zh']
# run_dataset(data_path='/data/exp', prompt_id_column='Variable', answer_column='Value', target_column='score', languages=languages, run_suffix='')

## Run ASAP (translated)
languages = ['ar', 'da', 'en', 'he', 'it', 'ka', 'nb', 'pt', 'sl', 'sv', 'zh']
run_dataset(data_path='/data/ASAP/split', prompt_id_column='PromptId', answer_column='AnswerText', target_column='Score1', languages=languages, run_suffix='_test')

## Run ASAP (multilingual)
# languages = ['de', 'en', 'es', 'fr', 'zh']
# run_dataset(data_path='/data/ASAP_crosslingual/split', prompt_id_column='prompt', answer_column='text', target_column='score', languages=languages, run_suffix='')