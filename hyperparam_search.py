import os
import sys
import logging
import pandas as pd
import torch

from train_xlmr import train_xlmr
from train_mbert import train_mbert
from train_sbert import train_sbert

from utils import eval_bert, write_classification_statistics


data_folder_root = '/data' 
dataset_name = 'dev'
results_folder = '/results/results_dev_SBERT'


for prompt in ['E011B16C', 'E011M18C']:
# for prompt in ['E011B16C']:
# for prompt in ['E011B06C']:
# for prompt in ['E011B06C', 'E011B10C', 'E011B15C']:
# for prompt in os.listdir(os.path.join(data_folder, dataset_name)):

    print(prompt)

    for language in os.listdir(os.path.join(data_folder_root, dataset_name, prompt)):

        print(language)

        data_folder = os.path.join(data_folder_root, dataset_name, prompt, language)

        df_train = pd.read_csv(os.path.join(data_folder, 'train.csv'))
        df_train = df_train.fillna('')
        df_val = pd.read_csv(os.path.join(data_folder, 'val.csv'))
        df_val = df_val.fillna('')
        df_test = pd.read_csv(os.path.join(data_folder, 'test.csv'))
        df_test = df_test.fillna('')

        print('Num Train:', len(df_train))
        print('Num Val:', len(df_val))
        print('Num Test:', len(df_test))

        print(df_test) 

        # for batch_size in [64, 32]:
        
        #     # run_path = os.path.join(results_folder, dataset_name, prompt, language, 'MBERT_bs' + str(batch_size))
        #     # gold, mbert_pred = train_mbert(run_path, df_train, df_val, df_test, answer_column="Value", target_column="score", num_epochs=20, batch_size=batch_size)
        #     # write_classification_statistics(filepath=run_path, y_true=gold, y_pred=mbert_pred)
            
        #     # run_path = os.path.join(results_folder, dataset_name, prompt, language, 'XLMR_bs' + str(batch_size))
        #     # gold, xlmr_pred = train_xlmr(run_path, df_train, df_val, df_test, answer_column="Value", target_column="score", num_epochs=20, batch_size=batch_size)
        #     # write_classification_statistics(filepath=run_path, y_true=gold, y_pred=xlmr_pred)

        #     run_path = os.path.join(results_folder, dataset_name, prompt, language, 'MBERT_bs' + str(batch_size) + '_with_warmup')
        #     gold, mbert_pred_warm = train_mbert(run_path, df_train, df_val, df_test, answer_column="Value", target_column="score", num_epochs=20, batch_size=batch_size, do_warmup=True)
        #     write_classification_statistics(filepath=run_path, y_true=gold, y_pred=mbert_pred_warm)
            
        #     run_path = os.path.join(results_folder, dataset_name, prompt, language, 'XLMR_bs' + str(batch_size) + '_with_warmup')
        #     gold, xlmr_pred_warm = train_xlmr(run_path, df_train, df_val, df_test, answer_column="Value", target_column="score", num_epochs=20, batch_size=batch_size, do_warmup=True)
        #     write_classification_statistics(filepath=run_path, y_true=gold, y_pred=xlmr_pred_warm)

        # # TODO: SBERT
        model_name = 'paraphrase-multilingual-MiniLM-L12-v2'
        num_epochs = 15

        # for batch_size in [256]:
        for batch_size, num_pairs, num_val_pairs in [(128, 25, 1000), (128, 50, 1000)]:
        # for batch_size, num_pairs, num_val_pairs in [(128, 50, 1000), (128, 50, 2000), (128, 25, 2000)]:
        # for batch_size, num_pairs in [(128, 15), (64, 15), (32, 15)]:
        # for batch_size, num_pairs in [(128, 25), (64, 25), (32, 25)]:
        # for batch_size, num_pairs in [(128, 50), (128, 100), (64, 50), (64, 100), (32, 50)]:
            # for num_pairs in [100]:
            # for num_pairs in [100, 200, 300]:

            run_path = os.path.join(results_folder, dataset_name, prompt, language, 'SBERT'+ '_' + model_name + '_numPairs' + str(num_pairs) + '_' + str(batch_size) + '_valPairs' + str(num_val_pairs) + '_warm')
            gold, pred_max, pred_avg = train_sbert(run_path, df_train, df_val, df_test, answer_column="Value", target_column="score", base_model=model_name, num_epochs=num_epochs, batch_size=batch_size, do_warmup=True, save_model=True, num_pairs_per_example=num_pairs, num_val_pairs=num_val_pairs)
            # train_sbert(run_path, df_train, df_val, df_test, answer_column="Value", target_column="score", base_model="xlm-roberta-base", num_epochs=20, batch_size=16, do_warmup=False, save_model=True)
            # train_sbert(run_path, df_train, df_val, df_test, answer_column="Value", target_column="score", base_model="xlm-roberta-base", num_epochs=20, batch_size=16, do_warmup=False, save_model=True)
            # df = pd.read_csv(os.path.join(run_path, 'predictions_sim.csv'))
            # gold = list(df['score'])
            # pred_avg = list(df['pred_avg'])
            # pred_max = list(df['pred_max'])
            write_classification_statistics(filepath=run_path, y_true=gold, y_pred=pred_avg, suffix='_avg')
            write_classification_statistics(filepath=run_path, y_true=gold, y_pred=pred_max, suffix='_max')

        # sys.exit(0)



## Snippet for evaluation

# device = 'cpu'
# if torch.cuda.is_available():
#     device = 'cuda'

# base_model = '/results/dev/E011B15C/da/XLMR/best_model'
# df_test = pd.read_csv('/data/dev/E011B15C/en/test.csv')
# model = XLMRobertaForSequenceClassification.from_pretrained(base_model).to(device)
# tokenizer = XLMRobertaTokenizer.from_pretrained(base_model)
# gold, pred = eval_bert(model, tokenizer, df_test)
# write_classification_statistics(os.path.join('/results/dev/E011B15C/da/XLMR/'), y_true=gold, y_pred=pred, suffix='_en')
        