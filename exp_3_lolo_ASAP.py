import os
import shutil
import sys
import torch

import pandas as pd

from copy import deepcopy
from model_training.train_mbert import train_mbert
from model_training.train_xlmr import train_xlmr
from model_training.train_sbert import train_sbert
from sentence_transformers import SentenceTransformer
from transformers import BertForSequenceClassification, BertTokenizer, XLMRobertaTokenizer, XLMRobertaForSequenceClassification
from utils import eval_bert, write_classification_statistics, read_data, eval_sbert


device = 'cpu'
if torch.cuda.is_available():
    device = 'cuda'

sbert_model_name = 'paraphrase-multilingual-MiniLM-L12-v2'
sbert_num_epochs = 8
# sbert_num_epochs = 15
sbert_batch_size = 64
# sbert_batch_size = 128
sbert_num_pairs = 25
sbert_num_val_pairs = 1000

bert_batch_size = 8
# bert_batch_size = 16
# bert_batch_size = 32
# bert_batch_size = 64
bert_num_epochs = 10
# bert_num_epochs = 20

random_state = 3456786544

languages = ['fr', 'zh']
# languages = ['de', 'en', 'es', 'fr', 'zh']
all_languages = ['de', 'en', 'es', 'fr', 'zh']

id_column = 'id'
answer_column = 'text'
target_column = 'score'

result_dir = '/results/exp_3_lolo_ASAPcross'
data_path = '/data/ASAP_crosslingual/split'

num_folds = 7


def full_data(run_xlmr=True, run_sbert=True, run_pretrained=False, translated=False):

    condition = 'combine_all_other'

    if translated:
        condition = condition + '_translated'

    for prompt in ['1', '10']:
    # for prompt in ['1', '2', '10']:
    # for prompt in os.listdir(data_path):

        # For each prompt - language pair, train a model
        # for language in os.listdir(os.path.join(data_path, prompt)):
        for test_language in languages:

            all_predictions = {}

            for val_fold in range(1, num_folds + 1):

                print(prompt, test_language, val_fold)

                # Read test, val data
                # Training is combination of data in other languages
                df_val = read_data(os.path.join(data_path, prompt, test_language, 'fold_' + str(val_fold) + '.csv'), answer_column=answer_column)
                df_test = pd.DataFrame()

                test_folds = list(range(1, num_folds + 1))
                test_folds.remove(val_fold)

                for test_fold in test_folds:
                    df_temp = read_data(os.path.join(data_path, prompt, test_language, 'fold_' + str(test_fold) + '.csv'), answer_column=answer_column)
                    df_test = pd.concat([df_test, df_temp])

                # Combine data of all *other* languages as training data
                df_train = pd.DataFrame()
                other_languages = deepcopy(all_languages)
                other_languages.remove(test_language)
                
                for other_language in other_languages:

                    train_folds = list(range(1, num_folds + 1))
                    train_folds.remove(val_fold)
                    if val_fold + 1 > num_folds:
                        train_folds.remove(1)
                    else:
                        train_folds.remove(val_fold+1)

                    for train_fold in train_folds:

                        df_other = pd.DataFrame()
                        if translated:   
                            df_other = read_data(os.path.join(data_path, prompt, other_language, 'fold_' + str(train_fold) + '_translated_' + test_language + '.csv'), answer_column=answer_column)
                        else:
                            df_other = read_data(os.path.join(data_path, prompt, other_language, 'fold_' + str(train_fold) + '.csv'), answer_column=answer_column)

                        df_train = pd.concat([df_train, df_other])
                
                print('train', len(df_train))
                print('val', len(df_val))
                print('test', len(df_test))

                if run_xlmr:
                    # ------------- Train XLMR -------------
                    run_path_bert = os.path.join(result_dir, condition, prompt, test_language, 'XLMR', 'fold_' + str(val_fold))
                    # Only run if this has not succesfully run already
                    if not os.path.exists(os.path.join(run_path_bert, 'preds.csv')):

                        gold, xlmr_pred = train_xlmr(run_path_bert, df_train=df_train, df_val=df_val, df_test=df_test, answer_column=answer_column, target_column=target_column, num_epochs=bert_num_epochs, batch_size=bert_batch_size, save_model=False)
                        write_classification_statistics(filepath=run_path_bert, y_true=gold, y_pred=xlmr_pred)
                        df_train.to_csv(os.path.join(run_path_bert, 'train.csv'))
                        df_val.to_csv(os.path.join(run_path_bert, 'val.csv'))
                        df_test.to_csv(os.path.join(run_path_bert, 'test.csv'))

                        preds_xlmr = all_predictions.get('XLMR', pd.DataFrame())
                        preds_xlmr = pd.concat([preds_xlmr, pd.read_csv(os.path.join(run_path_bert, 'preds.csv'))])
                        all_predictions['XLMR'] = preds_xlmr

                if run_sbert:
                    #  ---------- Train SBERT ------------
                    run_path_sbert = os.path.join(result_dir, condition, prompt, test_language, 'SBERT', 'fold_' + str(val_fold))
                    if not os.path.exists(os.path.join(run_path_sbert, 'preds.csv')):

                        gold, pred_max, pred_avg = train_sbert(run_path_sbert, df_train=df_train, df_val=df_val, df_test=df_test, id_column=id_column, answer_column=answer_column, target_column=target_column, base_model=sbert_model_name, num_epochs=sbert_num_epochs, batch_size=sbert_batch_size, do_warmup=False, save_model=False, num_pairs_per_example=sbert_num_pairs, num_val_pairs=sbert_num_val_pairs)
                        write_classification_statistics(filepath=run_path_sbert, y_true=gold, y_pred=pred_avg, suffix='')
                        write_classification_statistics(filepath=run_path_sbert, y_true=gold, y_pred=pred_max, suffix='_max')
                        df_train.to_csv(os.path.join(run_path_sbert, 'train.csv'))
                        df_val.to_csv(os.path.join(run_path_sbert, 'val.csv'))
                        df_test.to_csv(os.path.join(run_path_sbert, 'test.csv'))

                        preds_sbert = all_predictions.get('SBERT', pd.DataFrame())
                        preds_sbert = pd.concat([preds_sbert, pd.read_csv(os.path.join(run_path_sbert, 'preds.csv'))])
                        all_predictions['SBERT'] = preds_sbert

                if run_pretrained:
                    #  ---------- Eval pretrained SBERT ------------
                    run_path_pretrained = os.path.join(result_dir, condition, prompt, test_language, 'pretrained', 'fold_' + str(val_fold))
                    if not os.path.exists(os.path.join(run_path_pretrained, 'preds.csv')):

                        if not os.path.exists(run_path_pretrained):
                            os.makedirs(run_path_pretrained)
                            
                        # Load pretrained model 
                        model = SentenceTransformer(sbert_model_name)
                        df_ref = pd.concat([df_train, df_val])
                        df_ref['embedding'] = df_ref[answer_column].apply(model.encode)
                        df_test['embedding'] = df_test[answer_column].apply(model.encode)

                        # Predict on within-test data
                        gold, pred_max, pred_avg = eval_sbert(run_path_pretrained, df_test, df_ref, id_column=id_column, answer_column=answer_column, target_column=target_column)
                        write_classification_statistics(filepath=run_path_pretrained, y_true=gold, y_pred=pred_avg, suffix='')
                        write_classification_statistics(filepath=run_path_pretrained, y_true=gold, y_pred=pred_max, suffix='_max')
                        df_train.to_csv(os.path.join(run_path_pretrained, 'train.csv'))
                        df_val.to_csv(os.path.join(run_path_pretrained, 'val.csv'))
                        df_test.to_csv(os.path.join(run_path_pretrained, 'test.csv'))

                        preds_pretrained = all_predictions.get('pretrained', pd.DataFrame())
                        preds_pretrained = pd.concat([preds_pretrained, pd.read_csv(os.path.join(run_path_pretrained, 'preds.csv'))])
                        all_predictions['pretrained'] = preds_pretrained


            for model, df_preds in all_predictions.items():
                    
                df_preds.to_csv(os.path.join(result_dir, condition, prompt, test_language, model, 'preds.csv'))
                df_preds = df_preds.reset_index()
                gold = df_preds[target_column]
                if model == 'SBERT' or model == 'pretrained':
                    pred_avg = df_preds['pred_avg']
                    pred_max = df_preds['pred_max']
                    write_classification_statistics(filepath=os.path.join(result_dir, condition, prompt, test_language, model), y_true=gold, y_pred=pred_avg, suffix='')
                    write_classification_statistics(filepath=os.path.join(result_dir, condition, prompt, test_language, model), y_true=gold, y_pred=pred_max, suffix='_max')
                else:
                    pred = df_preds['pred']
                    write_classification_statistics(filepath=os.path.join(result_dir, condition, prompt, test_language, model), y_true=gold, y_pred=pred, suffix='')


def downsampled_data(run_xlmr=True, run_sbert=True, run_pretrained=False, translated=False):

    condition = 'combine_downsampled'

    if translated:
        condition = condition + '_translated'

    for prompt in ['1', '2', '10']:
    # for prompt in os.listdir(data_path):

        # For each prompt - language pair, train a model
        # for language in os.listdir(os.path.join(data_path, prompt)):
        for test_language in languages:

            all_predictions = {}

            for val_fold in range(1, num_folds + 1):

                print(prompt, test_language, val_fold)

                # Read test, val data
                # Training is combination of data in other languages
                df_val = read_data(os.path.join(data_path, prompt, test_language, 'fold_' + str(val_fold) + '.csv'), answer_column=answer_column)
                df_test = pd.DataFrame()

                test_folds = list(range(1, num_folds + 1))
                test_folds.remove(val_fold)

                for test_fold in test_folds:
                    df_temp = read_data(os.path.join(data_path, prompt, test_language, 'fold_' + str(test_fold) + '.csv'), answer_column=answer_column)
                    df_test = pd.concat([df_test, df_temp])

                # Combine data of all *other* languages as training data
                df_train = pd.DataFrame()
                other_languages = deepcopy(languages)
                other_languages.remove(test_language)
                
                for other_language in other_languages:

                    if (val_fold + 1) > num_folds:
                        train_folds = [1, val_fold]
                    else:
                        train_folds = [val_fold, val_fold + 1]

                    # take all of the first fold, and a quarter of the second

                    df_other = pd.DataFrame()
                    if translated:   
                        df_other = read_data(os.path.join(data_path, prompt, other_language, 'fold_' + str(train_folds[0]) + '_translated_' + test_language + '.csv'), answer_column=answer_column)
                    else:
                        df_other = read_data(os.path.join(data_path, prompt, other_language, 'fold_' + str(train_folds[0]) + '.csv'), answer_column=answer_column)

                    df_rest = pd.DataFrame()
                    if translated:   
                        df_rest = read_data(os.path.join(data_path, prompt, other_language, 'fold_' + str(train_folds[1]) + '_translated_' + test_language + '.csv'), answer_column=answer_column)
                    else:
                        df_rest = read_data(os.path.join(data_path, prompt, other_language, 'fold_' + str(train_folds[1]) + '.csv'), answer_column=answer_column)
                    
                    num_to_sample = len(df_other)/len(other_languages)
                    # print(num_to_sample)
                    df_sample = df_rest.sample(int(num_to_sample))

                    df_train = pd.concat([df_train, df_other, df_sample])

                
                print('train', len(df_train))
                print('val', len(df_val))
                print('test', len(df_test))

                if run_xlmr:
                    # ------------- Train XLMR -------------
                    run_path_bert = os.path.join(result_dir, condition, prompt, test_language, 'XLMR', 'fold_' + str(val_fold))
                    # Only run if this has not succesfully run already
                    if not os.path.exists(os.path.join(run_path_bert, 'preds.csv')):

                        gold, xlmr_pred = train_xlmr(run_path_bert, df_train=df_train, df_val=df_val, df_test=df_test, answer_column=answer_column, target_column=target_column, num_epochs=bert_num_epochs, batch_size=bert_batch_size, save_model=False)
                        write_classification_statistics(filepath=run_path_bert, y_true=gold, y_pred=xlmr_pred)
                        df_train.to_csv(os.path.join(run_path_bert, 'train.csv'))
                        df_val.to_csv(os.path.join(run_path_bert, 'val.csv'))
                        df_test.to_csv(os.path.join(run_path_bert, 'test.csv'))

                        preds_xlmr = all_predictions.get('XLMR', pd.DataFrame())
                        preds_xlmr = pd.concat([preds_xlmr, pd.read_csv(os.path.join(run_path_bert, 'preds.csv'))])
                        all_predictions['XLMR'] = preds_xlmr

                if run_sbert:
                    #  ---------- Train SBERT ------------
                    run_path_sbert = os.path.join(result_dir, condition, prompt, test_language, 'SBERT', 'fold_' + str(val_fold))
                    if not os.path.exists(os.path.join(run_path_sbert, 'preds.csv')):

                        gold, pred_max, pred_avg = train_sbert(run_path_sbert, df_train=df_train, df_val=df_val, df_test=df_test, id_column=id_column, answer_column=answer_column, target_column=target_column, base_model=sbert_model_name, num_epochs=sbert_num_epochs, batch_size=sbert_batch_size, do_warmup=False, save_model=False, num_pairs_per_example=sbert_num_pairs, num_val_pairs=sbert_num_val_pairs)
                        write_classification_statistics(filepath=run_path_sbert, y_true=gold, y_pred=pred_avg, suffix='')
                        write_classification_statistics(filepath=run_path_sbert, y_true=gold, y_pred=pred_max, suffix='_max')
                        df_train.to_csv(os.path.join(run_path_sbert, 'train.csv'))
                        df_val.to_csv(os.path.join(run_path_sbert, 'val.csv'))
                        df_test.to_csv(os.path.join(run_path_sbert, 'test.csv'))

                        preds_sbert = all_predictions.get('SBERT', pd.DataFrame())
                        preds_sbert = pd.concat([preds_sbert, pd.read_csv(os.path.join(run_path_sbert, 'preds.csv'))])
                        all_predictions['SBERT'] = preds_sbert

                if run_pretrained:
                    #  ---------- Eval pretrained SBERT ------------
                    run_path_pretrained = os.path.join(result_dir, condition, prompt, test_language, 'pretrained', 'fold_' + str(val_fold))
                    if not os.path.exists(os.path.join(run_path_pretrained, 'preds.csv')):

                        if not os.path.exists(run_path_pretrained):
                            os.makedirs(run_path_pretrained)
                            
                        # Load pretrained model 
                        model = SentenceTransformer(sbert_model_name)
                        df_ref = pd.concat([df_train, df_val])
                        df_ref['embedding'] = df_ref[answer_column].apply(model.encode)
                        df_test['embedding'] = df_test[answer_column].apply(model.encode)

                        # Predict on within-test data
                        gold, pred_max, pred_avg = eval_sbert(run_path_pretrained, df_test, df_ref, id_column=id_column, answer_column=answer_column, target_column=target_column)
                        write_classification_statistics(filepath=run_path_pretrained, y_true=gold, y_pred=pred_avg, suffix='')
                        write_classification_statistics(filepath=run_path_pretrained, y_true=gold, y_pred=pred_max, suffix='_max')
                        df_train.to_csv(os.path.join(run_path_pretrained, 'train.csv'))
                        df_val.to_csv(os.path.join(run_path_pretrained, 'val.csv'))
                        df_test.to_csv(os.path.join(run_path_pretrained, 'test.csv'))

                        preds_pretrained = all_predictions.get('pretrained', pd.DataFrame())
                        preds_pretrained = pd.concat([preds_pretrained, pd.read_csv(os.path.join(run_path_pretrained, 'preds.csv'))])
                        all_predictions['pretrained'] = preds_pretrained


            for model, df_preds in all_predictions.items():
                    
                df_preds.to_csv(os.path.join(result_dir, condition, prompt, test_language, model, 'preds.csv'))
                df_preds = df_preds.reset_index()
                gold = df_preds[target_column]
                if model == 'SBERT' or model == 'pretrained':
                    pred_avg = df_preds['pred_avg']
                    pred_max = df_preds['pred_max']
                    write_classification_statistics(filepath=os.path.join(result_dir, condition, prompt, test_language, model), y_true=gold, y_pred=pred_avg, suffix='')
                    write_classification_statistics(filepath=os.path.join(result_dir, condition, prompt, test_language, model), y_true=gold, y_pred=pred_max, suffix='_max')
                else:
                    pred = df_preds['pred']
                    write_classification_statistics(filepath=os.path.join(result_dir, condition, prompt, test_language, model), y_true=gold, y_pred=pred, suffix='')


# downsampled_data(run_sbert=False, run_xlmr=False, run_pretrained=True) #done
# downsampled_data(run_sbert=False, run_xlmr=False, run_pretrained=True, translated=True) #done

# downsampled_data(run_sbert=False, run_xlmr=True, run_pretrained=False) #done
# downsampled_data(run_sbert=False, run_xlmr=True, run_pretrained=False, translated=True) #done

# downsampled_data(run_sbert=True, run_xlmr=False, run_pretrained=False)
# downsampled_data(run_sbert=True, run_xlmr=False, run_pretrained=False, translated=True)

# full_data(run_sbert=False, run_xlmr=False, run_pretrained=True) # done
# full_data(run_sbert=False, run_xlmr=False, run_pretrained=True, translated=True)

# GPU 2
# full_data(run_sbert=False, run_xlmr=True, run_pretrained=False)
# full_data(run_sbert=False, run_xlmr=True, run_pretrained=False, translated=True)

# GPU 3
# full_data(run_sbert=True, run_xlmr=False, run_pretrained=False)
full_data(run_sbert=True, run_xlmr=False, run_pretrained=False, translated=True)

