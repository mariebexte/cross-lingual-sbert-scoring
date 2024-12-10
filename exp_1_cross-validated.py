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

bert_batch_size = 8
# bert_batch_size = 16
# bert_batch_size = 32
bert_num_epochs = 10

languages = ['en', 'es', 'fr', 'de', 'zh']
folds = [1,2,3,4,5,6,7]

id_column = 'id'
answer_column = 'text'
target_column = 'score'

result_dir = '/results/exp_1_cross-validated_m2m_100_1.2B'
data_path = '/data/ASAP_crosslingual/split'
translation_model = 'm2m_100_1.2B'


def read_data(path):

    df = pd.read_csv(path)
    df = df.fillna('')
    df[answer_column] = df[answer_column].astype(str)
    return df


def run_exp(run_sbert=True, run_xlmr=True, run_pretrained=False):

    for prompt in ['1','2','10']:
    # for prompt in os.listdir(data_path):

        # For each prompt - language pair, train 7 models for the 7 folds
        for language in languages:

            model_preds = {}

            for test_fold in folds:

                torch.cuda.empty_cache()

                print(prompt, language, test_fold)

                # Read data
                # Test: The target fold
                df_test = read_data(os.path.join(data_path, prompt, language, 'fold_' + str(test_fold) + '.csv'))

                # Val: target fold + 1, except if last
                val_fold = test_fold + 1
                if val_fold > folds[-1]:
                    val_fold = 1
                df_val = read_data(os.path.join(data_path, prompt, language, 'fold_' + str(val_fold) + '.csv'))

                # Train: All folds except train and val folds
                train_folds = deepcopy(folds)
                train_folds.remove(test_fold)
                train_folds.remove(val_fold)
                df_train = pd.DataFrame()
                for train_fold in train_folds:
                    df_fold = read_data(os.path.join(data_path, prompt, language, 'fold_' + str(train_fold) + '.csv'))
                    df_train = pd.concat([df_train, df_fold])

                # print(train_folds, val_fold, test_fold)
                print(len(df_train), len(df_train['id'].unique()), len(df_val), len(df_val['id'].unique()), len(df_test), len(df_test['id'].unique()))
                print(dict(df_train['score'].value_counts()))
                # df_total = pd.concat([df_train, df_val, df_test])
                # print(len(df_total), len(df_total['id'].unique()))

                
                if run_sbert:

                    #  ---------- Train SBERT ------------
                    run_path_sbert = os.path.join(result_dir, prompt, language, 'SBERT', 'fold_' + str(test_fold))
                    if not os.path.exists(os.path.join(run_path_sbert, 'preds.csv')):

                        gold, pred_max, pred_avg = train_sbert(run_path_sbert, df_train=df_train, df_val=df_val, df_test=df_test, answer_column=answer_column, target_column=target_column, base_model=sbert_model_name, num_epochs=sbert_num_epochs, batch_size=sbert_batch_size, do_warmup=False, save_model=True, num_pairs_per_example=sbert_num_pairs, num_val_pairs=sbert_num_val_pairs)
                        # Eval trained model on within-language data
                        write_classification_statistics(filepath=run_path_sbert, y_true=gold, y_pred=pred_avg, suffix='')
                        write_classification_statistics(filepath=run_path_sbert, y_true=gold, y_pred=pred_max, suffix='_max')

                        df_model_preds = model_preds.get('SBERT', pd.DataFrame())
                        df_model_preds = pd.concat([df_model_preds, pd.read_csv(os.path.join(run_path_sbert, 'preds.csv'))])
                        model_preds['SBERT'] = df_model_preds

                        # Load model that was just trained 
                        model = SentenceTransformer(os.path.join(run_path_sbert, 'finetuned_model'))
                        df_ref = pd.concat([df_train, df_val])
                        df_ref['embedding'] = df_ref[answer_column].apply(model.encode)
                        
                        # Evaluation of finetuned model on data from all **other** languages, both translated and cross-lingual transfer
                        other_languages = deepcopy(languages)
                        other_languages.remove(language)
                        for other_lang in other_languages:

                            run_path_test_sbert_crosslingual = os.path.join(run_path_sbert, other_lang)
                            if not os.path.exists(run_path_test_sbert_crosslingual):
                                os.mkdir(run_path_test_sbert_crosslingual)
                        
                            df_test_sbert = pd.DataFrame()
                            for fold in folds:
                                df_fold_lang = read_data(os.path.join(data_path, prompt, other_lang, 'fold_' + str(fold) + '.csv'))
                                df_test_sbert = pd.concat([df_test_sbert, df_fold_lang])
                            
                            df_test_sbert['embedding'] = df_test_sbert[answer_column].apply(model.encode)
                            gold, pred_max_zeroshot, pred_avg_zeroshot = eval_sbert(run_path_test_sbert_crosslingual, df_test_sbert, df_ref, id_column=id_column, answer_column=answer_column, target_column=target_column)

                            write_classification_statistics(filepath=run_path_test_sbert_crosslingual, y_true=gold, y_pred=pred_avg_zeroshot, suffix='')
                            write_classification_statistics(filepath=run_path_test_sbert_crosslingual, y_true=gold, y_pred=pred_max_zeroshot, suffix='_max')

                            # Evaluation on translated data
                            run_path_test_sbert_translated = os.path.join(run_path_sbert, other_lang + '_translated')
                            if not os.path.exists(run_path_test_sbert_translated):
                                os.mkdir(run_path_test_sbert_translated)
                        
                            df_test_sbert_translated = pd.DataFrame()
                            for fold in folds:
                                df_fold_lang = read_data(os.path.join(data_path, prompt, other_lang, 'fold_' + str(fold) + '_translated_' + language + '_' + translation_model + '.csv'))
                                df_test_sbert_translated = pd.concat([df_test_sbert_translated, df_fold_lang])
                            
                            df_test_sbert_translated['embedding'] = df_test_sbert_translated[answer_column].apply(model.encode)
                            gold, pred_max_translated, pred_avg_translated = eval_sbert(run_path_test_sbert_translated, df_test_sbert_translated, df_ref, id_column=id_column, answer_column=answer_column, target_column=target_column)

                            write_classification_statistics(filepath=run_path_test_sbert_translated, y_true=gold, y_pred=pred_avg_translated, suffix='')
                            write_classification_statistics(filepath=run_path_test_sbert_translated, y_true=gold, y_pred=pred_max_translated, suffix='_max')

                        
                        shutil.rmtree(os.path.join(run_path_sbert, 'finetuned_model'))


                if run_pretrained:

                    #  ---------- Load pretrained SBERT ------------
                    run_path_pretrained = os.path.join(result_dir, prompt, language, 'pretrained', 'fold_' + str(test_fold))
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
                                                
                        # Eval trained model on within-language data
                        write_classification_statistics(filepath=run_path_pretrained, y_true=gold, y_pred=pred_avg, suffix='')
                        write_classification_statistics(filepath=run_path_pretrained, y_true=gold, y_pred=pred_max, suffix='_max')

                        df_model_preds = model_preds.get('pretrained', pd.DataFrame())
                        df_model_preds = pd.concat([df_model_preds, pd.read_csv(os.path.join(run_path_pretrained, 'preds.csv'))])
                        model_preds['pretrained'] = df_model_preds

                        
                        # Evaluation on data from all **other** languages, both translated and cross-lingual transfer
                        other_languages = deepcopy(languages)
                        other_languages.remove(language)
                        for other_lang in other_languages:

                            run_path_test_pretrained_crosslingual = os.path.join(run_path_pretrained, other_lang)
                            if not os.path.exists(run_path_test_pretrained_crosslingual):
                                os.mkdir(run_path_test_pretrained_crosslingual)
                        
                            df_test_sbert = pd.DataFrame()
                            for fold in folds:
                                df_fold_lang = read_data(os.path.join(data_path, prompt, other_lang, 'fold_' + str(fold) + '.csv'))
                                df_test_sbert = pd.concat([df_test_sbert, df_fold_lang])
                            
                            df_test_sbert['embedding'] = df_test_sbert[answer_column].apply(model.encode)
                            gold, pred_max_zeroshot, pred_avg_zeroshot = eval_sbert(run_path_test_pretrained_crosslingual, df_test_sbert, df_ref, id_column=id_column, answer_column=answer_column, target_column=target_column)

                            write_classification_statistics(filepath=run_path_test_pretrained_crosslingual, y_true=gold, y_pred=pred_avg_zeroshot, suffix='')
                            write_classification_statistics(filepath=run_path_test_pretrained_crosslingual, y_true=gold, y_pred=pred_max_zeroshot, suffix='_max')

                            # Evaluation on translated data
                            run_path_test_pretrained_translated = os.path.join(run_path_pretrained, other_lang + '_translated')
                            if not os.path.exists(run_path_test_pretrained_translated):
                                os.mkdir(run_path_test_pretrained_translated)
                        
                            df_test_sbert_translated = pd.DataFrame()
                            for fold in folds:
                                df_fold_lang = read_data(os.path.join(data_path, prompt, other_lang, 'fold_' + str(fold) + '_translated_' + language + '_' + translation_model + '.csv'))
                                df_test_sbert_translated = pd.concat([df_test_sbert_translated, df_fold_lang])
                            
                            df_test_sbert_translated['embedding'] = df_test_sbert_translated[answer_column].apply(model.encode)
                            gold, pred_max_translated, pred_avg_translated = eval_sbert(run_path_test_pretrained_translated, df_test_sbert_translated, df_ref, id_column=id_column, answer_column=answer_column, target_column=target_column)

                            write_classification_statistics(filepath=run_path_test_pretrained_translated, y_true=gold, y_pred=pred_avg_translated, suffix='')
                            write_classification_statistics(filepath=run_path_test_pretrained_translated, y_true=gold, y_pred=pred_max_translated, suffix='_max')


                if run_xlmr:

                    # ------------- Train XLMR -------------
                    run_path_bert = os.path.join(result_dir, prompt, language, 'XLMR', 'fold_' + str(test_fold))
                    if not os.path.exists(os.path.join(run_path_bert, 'preds.csv')):

                        gold, xlmr_pred = train_xlmr(run_path_bert, df_train=df_train, df_val=df_val, df_test=df_test, answer_column=answer_column, target_column=target_column, num_epochs=bert_num_epochs, batch_size=bert_batch_size, save_model=True)
                        
                        write_classification_statistics(filepath=run_path_bert, y_true=gold, y_pred=xlmr_pred)

                        df_model_preds = model_preds.get('XLMR', pd.DataFrame())
                        df_model_preds = pd.concat([df_model_preds, pd.read_csv(os.path.join(run_path_bert, 'preds.csv'))])
                        model_preds['XLMR'] = df_model_preds
                        
                        bert_model = XLMRobertaForSequenceClassification.from_pretrained(os.path.join(run_path_bert, 'best_model')).to(device)
                        bert_tokenizer = XLMRobertaTokenizer.from_pretrained(os.path.join(run_path_bert, 'best_model'))

                        # Evaluation of finetuned model on data from all **other** languages, both translated and cross-lingual transfer
                        other_languages = deepcopy(languages)
                        other_languages.remove(language)
                        for other_lang in other_languages:

                            run_path_test_bert_crosslingual = os.path.join(run_path_bert, other_lang)
                            if not os.path.exists(run_path_test_bert_crosslingual):
                                os.mkdir(run_path_test_bert_crosslingual)

                            df_test_bert = pd.DataFrame()
                            for fold in folds:
                                df_fold_lang = read_data(os.path.join(data_path, prompt, other_lang, 'fold_' + str(fold) + '.csv'))
                                df_test_bert = pd.concat([df_test_bert, df_fold_lang])

                            gold, xlmr_pred_zeroshot = eval_bert(bert_model, bert_tokenizer, df_test_bert, answer_column=answer_column)

                            df_test_copy = deepcopy(df_test_bert)
                            df_test_copy['pred'] = xlmr_pred_zeroshot
                            df_test_copy.to_csv(os.path.join(run_path_test_bert_crosslingual, 'preds.csv'))

                            write_classification_statistics(filepath=run_path_test_bert_crosslingual, y_true=gold, y_pred=xlmr_pred_zeroshot, suffix='')

                            # Evaluation on translated data
                            run_path_test_bert_translated = os.path.join(run_path_bert, other_lang + '_translated')
                            if not os.path.exists(run_path_test_bert_translated):
                                os.mkdir(run_path_test_bert_translated)

                            df_test_bert_translated = pd.DataFrame()
                            for fold in folds:
                                df_fold_lang = read_data(os.path.join(data_path, prompt, other_lang, 'fold_' + str(fold) + '_translated_' + language + '_' + translation_model + '.csv'))
                                df_test_bert_translated = pd.concat([df_test_bert_translated, df_fold_lang])

                            gold, xlmr_pred_translated = eval_bert(bert_model, bert_tokenizer, df_test_bert_translated, answer_column=answer_column)

                            df_test_copy = deepcopy(df_test_bert_translated)
                            df_test_copy['pred'] = xlmr_pred_translated
                            df_test_copy.to_csv(os.path.join(run_path_test_bert_translated, 'preds.csv'))

                            write_classification_statistics(filepath=run_path_test_bert_translated, y_true=gold, y_pred=xlmr_pred_translated, suffix='')
                        
                        shutil.rmtree(os.path.join(run_path_bert, 'best_model'))


            for model, df_preds in model_preds.items():
                df_preds.to_csv(os.path.join(result_dir, prompt, language, model, 'preds.csv'))
                df_preds = df_preds.reset_index()
                gold = df_preds[target_column]
                if model == 'SBERT' or model == 'pretrained':
                    pred_avg = df_preds['pred_avg']
                    pred_max = df_preds['pred_max']
                    write_classification_statistics(filepath=os.path.join(result_dir, prompt, language, model), y_true=gold, y_pred=pred_avg, suffix='')
                    write_classification_statistics(filepath=os.path.join(result_dir, prompt, language, model), y_true=gold, y_pred=pred_max, suffix='_max')
                else:
                    pred = df_preds['pred']
                    write_classification_statistics(filepath=os.path.join(result_dir, prompt, language, model), y_true=gold, y_pred=pred, suffix='')


run_exp(run_sbert=False, run_xlmr=False, run_pretrained=True)
run_exp(run_sbert=False, run_xlmr=True)
run_exp(run_sbert=True, run_xlmr=False)