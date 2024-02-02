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

languages = ['ar', 'da', 'en', 'he', 'it', 'ka', 'nb', 'pt', 'sl', 'sv', 'zh']

id_column = 'id'
answer_column = 'Value'
target_column = 'score'

result_dir = '/results/exp_2_tradeoff'
data_path = '/data/exp'

random_state = 56398

amounts = [50, 100, 200, 300]


def run_exp(run_xlmr=True, run_sbert=True):

    for prompt in os.listdir(data_path):

        # For each prompt - language pair, train a model
        # for language in os.listdir(os.path.join(data_path, prompt)):
        for language in languages:

            torch.cuda.empty_cache()

            print(prompt, language)

            # Read data for training
            df_val = read_data(os.path.join(data_path, prompt, language, 'val.csv'))
            df_test = read_data(os.path.join(data_path, prompt, language, 'test.csv'))

            # This will be copied and downsampled multiple times
            df_train_target = read_data(os.path.join(data_path, prompt, language, 'train.csv'))
            
            label_dist = dict(df_train_target[target_column].value_counts())

            other_languages = deepcopy(languages)
            other_languages.remove(language)


            for other_language in other_languages:

                df_train_other = read_data(os.path.join(data_path, prompt, other_language, 'train.csv'))

                for num_target in amounts:

                    df_train = deepcopy(df_train_other)

                    # Determine how many answers of each label must be swapped for answers in target language
                    sample_ratio = num_target/len(df_train)

                    for label, amount in label_dist.items():

                        amount = int(round(sample_ratio*amount, 0))

                        # Remove this amount of foreign language
                        df_train_remove = df_train[df_train[target_column] == label].sample(amount, random_state=random_state)
                        df_train = df_train.drop(df_train_remove.index)

                        # Add this amount of target language
                        df_train_target_sample = df_train_target[df_train_target[target_column] == label].sample(amount, random_state=random_state)
                        df_train = pd.concat([df_train, df_train_target_sample])


                    if run_sbert:
                        #  ---------- Train SBERT ------------
                        run_path_sbert = os.path.join(result_dir, prompt, language, 'SBERT', other_language, str(num_target))
                        if not os.path.exists(os.path.join(run_path_sbert, 'preds.csv')):

                            gold, pred_max, pred_avg = train_sbert(run_path_sbert, df_train=df_train, df_val=df_val, df_test=df_test, answer_column=answer_column, target_column=target_column, base_model=sbert_model_name, num_epochs=sbert_num_epochs, batch_size=sbert_batch_size, do_warmup=False, save_model=False, num_pairs_per_example=sbert_num_pairs, num_val_pairs=sbert_num_val_pairs)
                            # Eval trained model on within-language data
                            write_classification_statistics(filepath=run_path_sbert, y_true=gold, y_pred=pred_avg, suffix='')
                            write_classification_statistics(filepath=run_path_sbert, y_true=gold, y_pred=pred_max, suffix='_max')

                            df_train.to_csv(os.path.join(run_path_sbert, 'train.csv'))
                            df_val.to_csv(os.path.join(run_path_sbert, 'val.csv'))
                            df_test.to_csv(os.path.join(run_path_sbert, 'test.csv'))
                            

                    if run_xlmr:
                        # ------------- Train XLMR -------------
                        run_path_bert = os.path.join(result_dir, prompt, language, 'XLMR', other_language, str(num_target))
                        if not os.path.exists(os.path.join(run_path_bert, 'preds.csv')):

                            gold, xlmr_pred = train_xlmr(run_path_bert, df_train=df_train, df_val=df_val, df_test=df_test, answer_column=answer_column, target_column=target_column, num_epochs=bert_num_epochs, batch_size=bert_batch_size, save_model=False)
                            
                            write_classification_statistics(filepath=run_path_bert, y_true=gold, y_pred=xlmr_pred)
                            
                            df_train.to_csv(os.path.join(run_path_bert, 'train.csv'))
                            df_val.to_csv(os.path.join(run_path_bert, 'val.csv'))
                            df_test.to_csv(os.path.join(run_path_bert, 'test.csv'))

# run_exp(run_xlmr=False)
run_exp(run_sbert=False)