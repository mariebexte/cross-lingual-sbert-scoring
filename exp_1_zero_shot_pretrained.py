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
languages = ['ar', 'da', 'en', 'he', 'it', 'ka', 'nb', 'pt', 'sl', 'sv', 'zh']

id_column = 'id'
answer_column = 'Value'
target_column = 'score'

result_dir = '/results/exp_1_zero_shot_pretrained_TEST'
data_path = '/data/exp'


def read_data(path):

    df = pd.read_csv(path)
    df = df.fillna('')
    df[answer_column] = df[answer_column].astype(str)
    return df

# Limba 0
# for prompt in ['E011B03C',  'E011B08C',  'E011B12C',  'E011B14C',  'E011M03C',  'E011B04C',  'E011B09C',  'E011B13C',  'E011M02C', 'E011T17C', 'E011Z09C',  'E011Z14C']:
# Limba 1
# for prompt in ['E011M08C',  'E011M11C',  'E011M15C',  'E011R05C',  'E011R09C',  'E011R14C',  'E011R16C',  'E011T05C',  'E011T09C',  'E011Z04C',  'E011Z12C']:
# Limba 3
for prompt in ['E011M04C',  'E011M09C',  'E011M13C',  'E011R02C',  'E011R08C',  'E011R11C',  'E011R15C',  'E011T02C',  'E011T08C',  'E011T10C',  'E011Z02C']:
# for prompt in os.listdir(data_path):


    # For each prompt - language pair, train a model
    # for language in os.listdir(os.path.join(data_path, prompt)):
    for language in languages:

        torch.cuda.empty_cache()

        print(prompt, language)

        # Read data for training
        df_train = read_data(os.path.join(data_path, prompt, language, 'train.csv'))
        df_val = read_data(os.path.join(data_path, prompt, language, 'val.csv'))
        df_test = read_data(os.path.join(data_path, prompt, language, 'test.csv'))

        
        #  ---------- Train SBERT ------------
        run_path_sbert = os.path.join(result_dir, prompt, language, 'SBERT')
        if not os.path.exists(os.path.join(run_path_sbert)):

            if not os.path.exists(run_path_sbert):
                os.makedirs(run_path_sbert)

            # Load model
            model = SentenceTransformer(sbert_model_name)
            df_ref = pd.concat([df_train, df_val])
            df_ref['embedding'] = df_ref[answer_column].apply(model.encode)

            print(len(df_ref))
            
            # Zero-shot evaluation of model on all languages
            for test_lang in languages:

                run_path_test_sbert = os.path.join(run_path_sbert, test_lang)
                if not os.path.exists(run_path_test_sbert):
                    os.mkdir(run_path_test_sbert)
                
                df_test_sbert = read_data(os.path.join(data_path, prompt, test_lang, 'test.csv'))
                df_test_sbert['embedding'] = df_test_sbert[answer_column].apply(model.encode)
                gold, pred_max, pred_avg = eval_sbert(run_path_test_sbert, df_test_sbert, df_ref, id_column=id_column, answer_column=answer_column, target_column=target_column)

                write_classification_statistics(filepath=run_path_test_sbert, y_true=gold, y_pred=pred_avg, suffix='')
                write_classification_statistics(filepath=run_path_test_sbert, y_true=gold, y_pred=pred_max, suffix='_max')