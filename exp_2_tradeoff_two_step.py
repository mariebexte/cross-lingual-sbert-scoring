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
sbert_num_epochs = 7
# sbert_num_epochs = 15
sbert_batch_size = 64
# sbert_batch_size = 128
sbert_num_pairs = 25
sbert_num_val_pairs = 1000

bert_batch_size = 32
# bert_batch_size = 64
bert_num_epochs = 10
# bert_num_epochs = 20

languages = ['ar', 'da', 'en', 'he', 'it', 'ka', 'nb', 'pt', 'sl', 'sv', 'zh']

id_column = 'id'
answer_column = 'Value'
target_column = 'score'

result_dir = '/results/exp_2_tradeoff_two_step_epochs-halved'
data_path = '/data/exp'

random_state = 56398

amounts = [35, 75, 150, 300]


def run_exp(run_xlmr=True, run_sbert=True):

    # Wika 1: LXMERT
    # Wika 2: SBERT
    for prompt in ['E011B14C', 'E011R02C', 'E011B03C', 'E011B12C', 'E011M03C', 'E011M11C', 'E011R05C', 'E011B04C']:

    # Limba: 1
    # for prompt in ['E011B09C','E011T05C', 'E011T17C', 'E011Z12C']:
    # Limba: 2
    # for prompt in ['E011M13C', 'E011R08C', 'E011R15C', 'E011T08C']: 
    # Limba: 3
    # for prompt in ['E011B08C', 'E011M08C', 'E011M15C', 'E011R09C']:

    # Lovelace: 0
    # for prompt in ['E011Z04C', 'E011M02C', 'E011M09C', 'E011R11C']:
    # Lovelace: 1
    # for prompt in ['E011Z09C', 'E011R14C', 'E011Z14C', 'E011T10C']:

    # Turing: 0
    # for prompt in ['E011M04C', 'E011B13C', 'E011T09C']:
    # Turing: 1
    # for prompt in ['E011R16C', 'E011T02C', 'E011T02C']:


    ### OLD
    # for prompt in ['E011B03C', 'E011B12C', 'E011M03C', 'E011M11C', 'E011R05C']: # Wika:1
    # for prompt in ['E011B09C','E011T05C', 'E011T17C', 'E011Z12C', 'E011B04C']: # Wika:2
    # for prompt in ['E011M13C', 'E011R08C', 'E011R15C', 'E011T08C']: # Limba:1
    # for prompt in ['E011B08C', 'E011M08C', 'E011M15C', 'E011R09C', 'E011R16C']: # Limba:2
    # for prompt in ['E011Z04C', 'E011M02C', 'E011M09C', 'E011R11C', 'E011T02C']: # Limba:3
    # for prompt in ['E011Z09C', 'E011R14C', 'E011Z14C', 'E011T10C']: # Lovelace:0
    # for prompt in ['E011M04C', 'E011B13C', 'E011T09C', 'E011Z02C']: # Lovelace:1

    # for prompt in os.listdir(data_path):

        # This is the base language
        for base_language in languages:

            torch.cuda.empty_cache()

            print(prompt, base_language)

            # Read data for training
            df_val_base = read_data(os.path.join(data_path, prompt, base_language, 'val.csv'))
            df_test_base = read_data(os.path.join(data_path, prompt, base_language, 'test.csv'))

            # This will be copied and downsampled multiple times
            df_train_base = read_data(os.path.join(data_path, prompt, base_language, 'train.csv'))
            
            # The distribution based on which answer counts will be calculated
            label_dist = dict(df_train_base[target_column].value_counts())

            for num_target in amounts:

                # Train base model for this base langauge and amount
                # Remove required number of training answers
                df_train_base_reduced = deepcopy(df_train_base)

                # Determine how many answers of each label must be swapped for answers in target language
                sample_ratio = num_target/len(df_train_base)

                for label, amount in label_dist.items():

                    amount = int(round(sample_ratio*amount, 0))

                    # Remove this amount of foreign language
                    df_train_remove = df_train_base[df_train_base[target_column] == label].sample(amount, random_state=random_state)
                    df_train_base_reduced = df_train_base_reduced.drop(df_train_remove.index)


                # Train the base model for this amount
                if run_xlmr:
                    # ------------- Train XLMR -------------
                    run_path_bert = os.path.join(result_dir, 'base_models', prompt, base_language, str(num_target), 'XLMR')
                    if not os.path.exists(os.path.join(run_path_bert, 'preds.csv')):

                        gold, xlmr_pred = train_xlmr(run_path_bert, df_train=df_train_base_reduced, df_val=df_val_base, df_test=df_test_base, answer_column=answer_column, target_column=target_column, num_epochs=bert_num_epochs, batch_size=bert_batch_size, save_model=True)
                        
                        write_classification_statistics(filepath=run_path_bert, y_true=gold, y_pred=xlmr_pred)
                        
                        df_train_base_reduced.to_csv(os.path.join(run_path_bert, 'train.csv'))
                        df_val_base.to_csv(os.path.join(run_path_bert, 'val.csv'))
                        df_test_base.to_csv(os.path.join(run_path_bert, 'test.csv'))
                
                if run_sbert:
                    #  ---------- Train SBERT ------------
                    run_path_sbert = os.path.join(result_dir, 'base_models', prompt, base_language, str(num_target), 'SBERT')
                    if not os.path.exists(os.path.join(run_path_sbert, 'preds.csv')):

                        gold, pred_max, pred_avg = train_sbert(run_path_sbert, df_train=df_train_base_reduced, df_val=df_val_base, df_test=df_test_base, answer_column=answer_column, target_column=target_column, base_model=sbert_model_name, num_epochs=sbert_num_epochs, batch_size=sbert_batch_size, do_warmup=False, save_model=True, num_pairs_per_example=sbert_num_pairs, num_val_pairs=sbert_num_val_pairs)
                        # Eval trained model on within-language data
                        write_classification_statistics(filepath=run_path_sbert, y_true=gold, y_pred=pred_avg, suffix='')
                        write_classification_statistics(filepath=run_path_sbert, y_true=gold, y_pred=pred_max, suffix='_max')

                        df_train_base_reduced.to_csv(os.path.join(run_path_sbert, 'train.csv'))
                        df_val_base.to_csv(os.path.join(run_path_sbert, 'val.csv'))
                        df_test_base.to_csv(os.path.join(run_path_sbert, 'test.csv'))


                # for each target language, finetune the base model with the amount specified
                target_languages = deepcopy(languages)
                target_languages.remove(base_language)

                for target_language in target_languages:

                    # Sample required amount of training data
                    df_train_target = read_data(os.path.join(data_path, prompt, target_language, 'train.csv'))
                    df_val_target = read_data(os.path.join(data_path, prompt, target_language, 'val.csv'))
                    df_test_target = read_data(os.path.join(data_path, prompt, target_language, 'test.csv'))

                    df_train_target_sample = pd.DataFrame()

                    for label, amount in label_dist.items():

                        amount = int(round(sample_ratio*amount, 0))

                        # Collect this amount of target language
                        df_train_target_sample_label = df_train_target[df_train_target[target_column] == label].sample(amount, random_state=random_state)
                        df_train_target_sample = pd.concat([df_train_target_sample, df_train_target_sample_label])


                    if run_xlmr:
                        # ------------- Train XLMR -------------
                        run_path_bert_finetune = os.path.join(result_dir, prompt, target_language, 'XLMR', base_language, str(num_target))

                        if not os.path.exists(os.path.join(run_path_bert_finetune, 'preds.csv')):
                        
                            base_model_xlmr = os.path.join(run_path_bert, 'best_model')

                            gold, xlmr_pred = train_xlmr(run_path_bert_finetune, df_train=df_train_target_sample, df_val=df_val_target, df_test=df_test_target, answer_column=answer_column, target_column=target_column, num_epochs=bert_num_epochs, batch_size=bert_batch_size, base_model=base_model_xlmr, save_model=False)
                            
                            write_classification_statistics(filepath=run_path_bert_finetune, y_true=gold, y_pred=xlmr_pred)
                            
                            df_train_target_sample.to_csv(os.path.join(run_path_bert_finetune, 'train.csv'))
                            df_val_target.to_csv(os.path.join(run_path_bert_finetune, 'val.csv'))
                            df_test_target.to_csv(os.path.join(run_path_bert_finetune, 'test.csv'))
                
                    if run_sbert:
                        #  ---------- Train SBERT ------------
                        run_path_sbert_finetune = os.path.join(result_dir, prompt, target_language, 'SBERT', base_language, str(num_target))
                        
                        if not os.path.exists(os.path.join(run_path_sbert_finetune, 'preds.csv')):

                            base_model_sbert = os.path.join(run_path_sbert, 'finetuned_model')

                            gold, pred_max, pred_avg = train_sbert(run_path_sbert_finetune, df_train=df_train_target_sample, df_val=df_val_target, df_test=df_test_target, answer_column=answer_column, target_column=target_column, base_model=base_model_sbert, num_epochs=sbert_num_epochs, batch_size=sbert_batch_size, do_warmup=False, save_model=False, num_pairs_per_example=sbert_num_pairs, num_val_pairs=sbert_num_val_pairs)
                            # Eval trained model on within-language data
                            write_classification_statistics(filepath=run_path_sbert_finetune, y_true=gold, y_pred=pred_avg, suffix='')
                            write_classification_statistics(filepath=run_path_sbert_finetune, y_true=gold, y_pred=pred_max, suffix='_max')

                            df_train_target_sample.to_csv(os.path.join(run_path_sbert_finetune, 'train.csv'))
                            df_val_target.to_csv(os.path.join(run_path_sbert_finetune, 'val.csv'))
                            df_test_target.to_csv(os.path.join(run_path_sbert_finetune, 'test.csv'))
                            
                if run_xlmr:
                    if os.path.exists(os.path.join(run_path_bert, 'best_model')):
                        shutil.rmtree(os.path.join(run_path_bert, 'best_model'))

                if run_sbert:
                    if os.path.exists(os.path.join(run_path_sbert, 'finetuned_model')):
                        shutil.rmtree(os.path.join(run_path_sbert, 'finetuned_model'))


# run_exp()
run_exp(run_xlmr=False)
# run_exp(run_sbert=False)