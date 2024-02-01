import pandas as pd
from train_mbert import train_mbert
from train_xlmr import train_xlmr
from train_sbert import train_sbert, eval_sbert
from transformers import BertForSequenceClassification, BertTokenizer, XLMRobertaTokenizer, XLMRobertaForSequenceClassification
from sentence_transformers import SentenceTransformer
from utils import eval_bert, write_classification_statistics, read_data
from copy import deepcopy
import os
import sys
import torch
import shutil

device = 'cpu'
if torch.cuda.is_available():
    device = 'cuda'

sbert_model_name = 'paraphrase-multilingual-MiniLM-L12-v2'
sbert_num_epochs = 15
sbert_batch_size = 64
# sbert_batch_size = 128
sbert_num_pairs = 25
sbert_num_val_pairs = 1000

bert_batch_size = 32
# bert_batch_size = 64
bert_num_epochs = 20

random_state = 3456786544

languages = ['ar', 'da', 'en', 'he', 'it', 'ka', 'nb', 'pt', 'sl', 'sv', 'zh']

id_column = 'id'
answer_column = 'Value'
target_column = 'score'

result_dir = '/results/exp_3_lolo'
data_path = '/data/exp'


def full_data(run_xlmr=True, run_sbert=True):

    condition = 'combine_all_other'

    for prompt in os.listdir(data_path):

        # For each prompt - language pair, train a model
        # for language in os.listdir(os.path.join(data_path, prompt)):
        for language in languages:

            torch.cuda.empty_cache()

            print(prompt, language)

            # Read test, val data
            # Training is combination of data in other languages
            df_test = read_data(os.path.join(data_path, prompt, language, 'test.csv'))
            df_val = read_data(os.path.join(data_path, prompt, language, 'val.csv'))

            # Combine data of all *other* languages as training data
            df_train = pd.DataFrame()
            other_languages = deepcopy(languages)
            other_languages.remove(language)
            
            for other_language in other_languages:

                df_other = read_data(os.path.join(data_path, prompt, other_language, 'train.csv'))
                df_train = pd.concat([df_train, df_other])
            

            if run_xlmr:
                # ------------- Train XLMR -------------
                run_path_bert = os.path.join(result_dir, condition, prompt, language, 'XLMR')
                # Only run if this has not succesfully run already
                if not os.path.exists(os.path.join(run_path_bert, 'preds.csv')):

                    gold, xlmr_pred = train_xlmr(run_path_bert, df_train=df_train, df_val=df_val, df_test=df_test, answer_column=answer_column, target_column=target_column, num_epochs=bert_num_epochs, batch_size=bert_batch_size, save_model=False)
                    write_classification_statistics(filepath=run_path_bert, y_true=gold, y_pred=xlmr_pred)
                    df_train.to_csv(os.path.join(result_dir, condition, prompt,language, 'XLMR', 'train.csv'))
                    df_val.to_csv(os.path.join(result_dir, condition, prompt,language, 'XLMR', 'val.csv'))
                    df_test.to_csv(os.path.join(result_dir, condition, prompt,language, 'XLMR', 'test.csv'))

            if run_sbert:
                #  ---------- Train SBERT ------------
                run_path_sbert = os.path.join(result_dir, condition, prompt, language, 'SBERT')
                if not os.path.exists(os.path.join(run_path_sbert, 'preds.csv')):

                    gold, pred_max, pred_avg = train_sbert(run_path_sbert, df_train=df_train, df_val=df_val, df_test=df_test, answer_column=answer_column, target_column=target_column, base_model=sbert_model_name, num_epochs=sbert_num_epochs, batch_size=sbert_batch_size, do_warmup=False, save_model=False, num_pairs_per_example=sbert_num_pairs, num_val_pairs=sbert_num_val_pairs)
                    write_classification_statistics(filepath=run_path_sbert, y_true=gold, y_pred=pred_avg, suffix='')
                    write_classification_statistics(filepath=run_path_sbert, y_true=gold, y_pred=pred_max, suffix='_max')
                    df_train.to_csv(os.path.join(result_dir, condition, prompt,language, 'XLMR', 'train.csv'))
                    df_val.to_csv(os.path.join(result_dir, condition, prompt,language, 'XLMR', 'val.csv'))
                    df_test.to_csv(os.path.join(result_dir, condition, prompt,language, 'XLMR', 'test.csv'))



def downsampled_data(run_xlmr=True, run_sbert=True):

    condition = 'combine_downsampled_other'

    for prompt in os.listdir(data_path):

        # For each prompt - language pair, train a model
        # for language in os.listdir(os.path.join(data_path, prompt)):
        for language in languages:

            torch.cuda.empty_cache()

            print(prompt, language)

            # Read test, val data
            # Training is combination of data in other languages
            df_test = read_data(os.path.join(data_path, prompt, language, 'test.csv'))
            df_val = read_data(os.path.join(data_path, prompt, language, 'val.csv'))

            # Combine data of all *other* languages as training data
            df_train = pd.DataFrame()
            other_languages = deepcopy(languages)
            other_languages.remove(language)

            # Just to grab label distribution and number of answers
            df_target_dist = read_data(os.path.join(data_path, prompt, language, 'train.csv'))
            label_dist = dict(df_target_dist[target_column].value_counts())
            
            for other_language in other_languages:
                
                df_other = read_data(os.path.join(data_path, prompt, other_language, 'train.csv'))
                
                # Sample to arrive at same number of answers as before (600 for epirls)
                num_train = len(df_target_dist)
                num_to_sample = int(num_train/len(other_languages))
                proportion_to_sample = num_to_sample/num_train
                
                for label, amount in label_dist.items():

                    amount = int(round(amount*proportion_to_sample, 0))

                    df_label = df_other[df_other[target_column] == label]
                    df_sample = df_label.sample(amount, random_state=random_state)
                    df_train = pd.concat([df_train, df_sample])

 
            if run_xlmr:
                # ------------- Train XLMR -------------
                run_path_bert = os.path.join(result_dir, condition, prompt, language, 'XLMR')
                # Only run if this has not succesfully run already
                if not os.path.exists(os.path.join(run_path_bert, 'preds.csv')):

                    gold, xlmr_pred = train_xlmr(run_path_bert, df_train=df_train, df_val=df_val, df_test=df_test, answer_column=answer_column, target_column=target_column, num_epochs=bert_num_epochs, batch_size=bert_batch_size, save_model=False)
                    write_classification_statistics(filepath=run_path_bert, y_true=gold, y_pred=xlmr_pred)
                    df_train.to_csv(os.path.join(result_dir, condition, prompt,language, 'XLMR', 'train.csv'))
                    df_val.to_csv(os.path.join(result_dir, condition, prompt,language, 'XLMR', 'vsl.csv'))
                    df_test.to_csv(os.path.join(result_dir, condition, prompt,language, 'XLMR', 'test.csv'))

            if run_sbert:
                #  ---------- Train SBERT ------------
                run_path_sbert = os.path.join(result_dir, condition, prompt, language, 'SBERT')
                if not os.path.exists(os.path.join(run_path_sbert, 'preds.csv')):

                    gold, pred_max, pred_avg = train_sbert(run_path_sbert, df_train=df_train, df_val=df_val, df_test=df_test, answer_column=answer_column, target_column=target_column, base_model=sbert_model_name, num_epochs=sbert_num_epochs, batch_size=sbert_batch_size, do_warmup=False, save_model=False, num_pairs_per_example=sbert_num_pairs, num_val_pairs=sbert_num_val_pairs)
                    write_classification_statistics(filepath=run_path_sbert, y_true=gold, y_pred=pred_avg, suffix='')
                    write_classification_statistics(filepath=run_path_sbert, y_true=gold, y_pred=pred_max, suffix='_max')
                    df_train.to_csv(os.path.join(result_dir, condition, prompt,language, 'XLMR', 'train.csv'))
                    df_val.to_csv(os.path.join(result_dir, condition, prompt,language, 'XLMR', 'val.csv'))
                    df_test.to_csv(os.path.join(result_dir, condition, prompt,language, 'XLMR', 'test.csv'))


# downsampled_data(run_sbert=False)
full_data(run_xlmr=False)