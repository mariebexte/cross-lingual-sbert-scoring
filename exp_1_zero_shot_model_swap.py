import pandas as pd
from train_mbert import train_mbert
from train_xlmr_sbert_core import train_xlmr
from train_sbert import train_sbert, eval_sbert
from transformers import BertForSequenceClassification, BertTokenizer, AutoConfig
from sentence_transformers import SentenceTransformer
from utils import eval_sbert_classification, write_classification_statistics, read_data
from copy import deepcopy
import os
import sys
import torch
import shutil
from sbert_for_classification import SbertForSequenceClassification

device = 'cpu'
if torch.cuda.is_available():
    device = 'cuda'

sbert_model_name = 'xlm-roberta-base'
sbert_num_epochs = 8
# sbert_batch_size = 16
sbert_batch_size = 64
sbert_num_pairs = 25
sbert_num_val_pairs = 1000

# bert_model_name = 'paraphrase-multilingual-MiniLM-L12-v2'
bert_model_name = '/models/paraphrase-multilingual-MiniLM-L12-v2'
bert_batch_size = 32
bert_num_epochs = 10

languages = ['ar', 'da', 'en', 'he', 'it', 'ka', 'nb', 'pt', 'sl', 'sv', 'zh']
all_languages = ['ar', 'da', 'en', 'he', 'it', 'ka', 'nb', 'pt', 'sl', 'sv', 'zh']

id_column = 'id'
answer_column = 'Value'
target_column = 'score'

result_dir = '/results/exp_1_zero_shot_model_swap'
data_path = '/data/exp'


def read_data(path):

    df = pd.read_csv(path)
    df = df.fillna('')
    df[answer_column] = df[answer_column].astype(str)
    return df

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

        # Read data for training
        df_train = read_data(os.path.join(data_path, prompt, language, 'train.csv'))
        df_val = read_data(os.path.join(data_path, prompt, language, 'val.csv'))
        df_test = read_data(os.path.join(data_path, prompt, language, 'test.csv'))

        
        #  ---------- Train SBERT ------------
        run_path_sbert = os.path.join(result_dir, prompt, language, 'SBERT')
        if not os.path.exists(os.path.join(run_path_sbert, 'preds.csv')):

            gold, pred_max, pred_avg = train_sbert(run_path_sbert, df_train=df_train, df_val=df_val, df_test=df_test, answer_column=answer_column, target_column=target_column, base_model=sbert_model_name, num_epochs=sbert_num_epochs, batch_size=sbert_batch_size, do_warmup=False, save_model=True, num_pairs_per_example=sbert_num_pairs, num_val_pairs=sbert_num_val_pairs)
            # Eval trained model on within-language data
            write_classification_statistics(filepath=run_path_sbert, y_true=gold, y_pred=pred_avg, suffix='')
            write_classification_statistics(filepath=run_path_sbert, y_true=gold, y_pred=pred_max, suffix='_max')

            # Load model that was just trained 
            model = SentenceTransformer(os.path.join(run_path_sbert, 'finetuned_model'))
            df_ref = pd.concat([df_train, df_val])
            df_ref['embedding'] = df_ref[answer_column].apply(model.encode)
            
            # Zero-shot evaluation of finetuned model on all **other** languages
            for test_lang in all_languages:

                run_path_test_sbert = os.path.join(run_path_sbert, test_lang)
                if not os.path.exists(run_path_test_sbert):
                    os.mkdir(run_path_test_sbert)
                
                df_test_sbert = read_data(os.path.join(data_path, prompt, test_lang, 'test.csv'))
                df_test_sbert['embedding'] = df_test_sbert[answer_column].apply(model.encode)
                gold, pred_max, pred_avg = eval_sbert(run_path_test_sbert, df_test_sbert, df_ref, id_column=id_column, answer_column=answer_column, target_column=target_column)

                write_classification_statistics(filepath=run_path_test_sbert, y_true=gold, y_pred=pred_avg, suffix='')
                write_classification_statistics(filepath=run_path_test_sbert, y_true=gold, y_pred=pred_max, suffix='_max')


                # Eperimental: Use val of test language to build pairs
                run_path_test_sbert = os.path.join(run_path_sbert, test_lang + '_target_val')
                if not os.path.exists(run_path_test_sbert):
                    os.mkdir(run_path_test_sbert)
                    
                df_ref = read_data(os.path.join(data_path, prompt, test_lang, 'val.csv'))
                df_ref['embedding'] = df_ref[answer_column].apply(model.encode)
                gold, pred_max, pred_avg = eval_sbert(run_path_test_sbert, df_test_sbert, df_ref, id_column=id_column, answer_column=answer_column, target_column=target_column)
                write_classification_statistics(filepath=run_path_test_sbert, y_true=gold, y_pred=pred_avg, suffix='')
                write_classification_statistics(filepath=run_path_test_sbert, y_true=gold, y_pred=pred_max, suffix='_max')
            
            shutil.rmtree(os.path.join(run_path_sbert, 'finetuned_model'))


        # ------------- Train XLMR -------------
        # run_path_bert = os.path.join(result_dir, prompt, language, 'XLMR')
        # if not os.path.exists(os.path.join(run_path_bert, 'preds.csv')):

        #     gold, xlmr_pred = train_xlmr(run_path_bert, df_train=df_train, df_val=df_val, df_test=df_test, answer_column=answer_column, target_column=target_column, num_epochs=bert_num_epochs, batch_size=bert_batch_size, save_model=True, base_model=bert_model_name)
            
        #     write_classification_statistics(filepath=run_path_bert, y_true=gold, y_pred=xlmr_pred)
            
        #     config = AutoConfig.from_pretrained(bert_model_name)
        #     config.sbert_path = bert_model_name
        #     config.num_labels = len(df_train[target_column].unique())
        #     bert_model = SbertForSequenceClassification(config).to(device)
        #     # bert_model.load_state_dict(os.path.join(run_path_bert, 'best_model'))
        #     # bert_model = torch.load(os.path.join(run_path_bert, 'best_model', 'pytorch_model.bin'))
        #     bert_model.load_state_dict(torch.load(os.path.join(run_path_bert, 'best_model', 'pytorch_model.bin')))

        #     # Zero-shot evaluation of finetuned model on all **other** languages
        #     for test_lang in all_languages:

        #         df_test_bert = read_data(os.path.join(data_path, prompt, test_lang, 'test.csv'))
        #         gold, xlmr_pred = eval_sbert_classification(bert_model, df_test_bert)

        #         run_path_test_bert = os.path.join(run_path_bert, test_lang)
        #         if not os.path.exists(run_path_test_bert):
        #             os.mkdir(run_path_test_bert)

        #         df_test_copy = deepcopy(df_test_bert)
        #         df_test_copy['pred'] = xlmr_pred
        #         df_test_copy.to_csv(os.path.join(run_path_test_bert, 'preds.csv'))

        #         write_classification_statistics(filepath=run_path_test_bert, y_true=gold, y_pred=xlmr_pred, suffix='')
            
        #     shutil.rmtree(os.path.join(run_path_bert, 'best_model'))