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
sbert_num_epochs = 8
sbert_batch_size = 64
# sbert_batch_size = 128
sbert_num_pairs = 25
sbert_num_val_pairs = 1000

# bert_batch_size = 8
bert_batch_size = 32
# bert_batch_size = 64
bert_num_epochs = 10

languages = ['ar', 'da', 'en', 'he', 'it', 'ka', 'nb', 'pt', 'sl', 'sv', 'zh']

id_column = 'id'
answer_column = 'Value'
target_column = 'score'

result_dir = '/results/exp_1_translation_pretrained'
data_path = '/data/exp'
translation_model = 'm2m_100_1.2B'


def read_data(path):

    df = pd.read_csv(path)
    df = df.fillna('')
    df[answer_column] = df[answer_column].astype(str)
    return df

# Limba 0
# for prompt in ['E011B03C',  'E011B08C',  'E011B12C',  'E011B14C',  'E011M03C',  'E011M08C',  'E011M11C',  'E011M15C',  'E011R05C',  'E011R09C',  'E011R14C',  'E011R16C',  'E011T05C',  'E011T09C',  'E011T17C',  'E011Z04C',  'E011Z12C']:
# Limba 1
# for prompt in ['E011B04C',  'E011B09C',  'E011B13C',  'E011M02C',  'E011M04C',  'E011M09C',  'E011M13C',  'E011R02C',  'E011R08C',  'E011R11C',  'E011R15C',  'E011T02C',  'E011T08C',  'E011T10C',  'E011Z02C',  'E011Z09C',  'E011Z14C']:
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
        # run_path_sbert = os.path.join(result_dir, prompt, language, 'SBERT')
        # if not os.path.exists(os.path.join(run_path_sbert, 'preds.csv')):

        #     gold, pred_max, pred_avg = train_sbert(run_path_sbert, df_train=df_train, df_val=df_val, df_test=df_test, answer_column=answer_column, target_column=target_column, base_model=sbert_model_name, num_epochs=sbert_num_epochs, batch_size=sbert_batch_size, do_warmup=False, save_model=True, num_pairs_per_example=sbert_num_pairs, num_val_pairs=sbert_num_val_pairs)
        #     # Eval trained model on within-language data
        #     write_classification_statistics(filepath=run_path_sbert, y_true=gold, y_pred=pred_avg, suffix='')
        #     write_classification_statistics(filepath=run_path_sbert, y_true=gold, y_pred=pred_max, suffix='_max')

        #     # Load model that was just trained 
        #     model = SentenceTransformer(os.path.join(run_path_sbert, 'finetuned_model'))
        #     df_ref = pd.concat([df_train, df_val])
        #     df_ref['embedding'] = df_ref[answer_column].apply(model.encode)
            
        #    # Evaluation of finetuned model on translated data from all **other** languages
        #    # Translated language from other to target language
        #     for test_lang in languages:

        #         run_path_test_sbert = os.path.join(run_path_sbert, test_lang)
        #         if not os.path.exists(run_path_test_sbert):
        #             os.mkdir(run_path_test_sbert)
                
        #         if language == test_lang:
        #             df_test_sbert = read_data(os.path.join(data_path, prompt, language, 'test.csv'))
        #         else:
        #             df_test_sbert = read_data(os.path.join(data_path, prompt, test_lang, 'test_translated_' + translation_model + '_' + language + '.csv'))


        #         df_test_sbert['embedding'] = df_test_sbert[answer_column].apply(model.encode)
        #         gold, pred_max, pred_avg = eval_sbert(run_path_test_sbert, df_test_sbert, df_ref, id_column=id_column, answer_column=answer_column, target_column=target_column)

        #         write_classification_statistics(filepath=run_path_test_sbert, y_true=gold, y_pred=pred_avg, suffix='')
        #         write_classification_statistics(filepath=run_path_test_sbert, y_true=gold, y_pred=pred_max, suffix='_max')

            
        #     shutil.rmtree(os.path.join(run_path_sbert, 'finetuned_model'))



        #  ---------- Run pretrained SBERT ------------
        run_path_sbert = os.path.join(result_dir, prompt, language, 'SBERT')
        if not os.path.exists(os.path.join(run_path_sbert, 'preds.csv')):

            if not os.path.exists(run_path_sbert):
                os.makedirs(run_path_sbert)

            # Load model
            model = SentenceTransformer(sbert_model_name)
            df_ref = pd.concat([df_train, df_val])
            df_ref['embedding'] = df_ref[answer_column].apply(model.encode)
            
           # Evaluation of finetuned model on translated data from all **other** languages
           # Translated language from other to target language
            for test_lang in languages:

                run_path_test_sbert = os.path.join(run_path_sbert, test_lang)
                if not os.path.exists(run_path_test_sbert):
                    os.mkdir(run_path_test_sbert)
                
                if language == test_lang:
                    df_test_sbert = read_data(os.path.join(data_path, prompt, language, 'test.csv'))
                else:
                    df_test_sbert = read_data(os.path.join(data_path, prompt, test_lang, 'test_translated_' + translation_model + '_' + language + '.csv'))


                df_test_sbert['embedding'] = df_test_sbert[answer_column].apply(model.encode)
                gold, pred_max, pred_avg = eval_sbert(run_path_test_sbert, df_test_sbert, df_ref, id_column=id_column, answer_column=answer_column, target_column=target_column)

                write_classification_statistics(filepath=run_path_test_sbert, y_true=gold, y_pred=pred_avg, suffix='')
                write_classification_statistics(filepath=run_path_test_sbert, y_true=gold, y_pred=pred_max, suffix='_max')


        # ------------- Train XLMR -------------
        # run_path_bert = os.path.join(result_dir, prompt, language, 'XLMR')
        # if not os.path.exists(os.path.join(run_path_bert, 'preds.csv')):

        #     gold, xlmr_pred = train_xlmr(run_path_bert, df_train=df_train, df_val=df_val, df_test=df_test, answer_column=answer_column, target_column=target_column, num_epochs=bert_num_epochs, batch_size=bert_batch_size, save_model=True)
            
        #     write_classification_statistics(filepath=run_path_bert, y_true=gold, y_pred=xlmr_pred)
            
        #     bert_model = XLMRobertaForSequenceClassification.from_pretrained(os.path.join(run_path_bert, 'best_model')).to(device)
        #     bert_tokenizer = XLMRobertaTokenizer.from_pretrained(os.path.join(run_path_bert, 'best_model'))

        #     # Evaluation of finetuned model on translated data from all **other** languages
        #     # Translated language from other to target language
        #     for test_lang in languages:

        #         if language == test_lang:
        #             df_test_bert = read_data(os.path.join(data_path, prompt, language, 'test.csv'))
        #         else:
        #             df_test_bert = read_data(os.path.join(data_path, prompt, test_lang, 'test_translated_' + translation_model + '_' + language + '.csv'))
        #         gold, xlmr_pred = eval_bert(bert_model, bert_tokenizer, df_test_bert)

        #         run_path_test_bert = os.path.join(run_path_bert, test_lang)
        #         if not os.path.exists(run_path_test_bert):
        #             os.mkdir(run_path_test_bert)

        #         df_test_copy = deepcopy(df_test_bert)
        #         df_test_copy['pred'] = xlmr_pred
        #         df_test_copy.to_csv(os.path.join(run_path_test_bert, 'preds.csv'))

        #         write_classification_statistics(filepath=run_path_test_bert, y_true=gold, y_pred=xlmr_pred, suffix='')
            
        #     shutil.rmtree(os.path.join(run_path_bert, 'best_model'))


