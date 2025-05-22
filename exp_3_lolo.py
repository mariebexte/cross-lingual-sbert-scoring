import os
import shutil
import sys
import torch

import pandas as pd

from config import EPIRLS, ASAP_T, SBERT_BASE_MODEL, XLMR_BASE_MODEL, SBERT_NUM_EPOCHS, BERT_NUM_EPOCHS, SBERT_NUM_PAIRS, SBERT_NUM_VAL_PAIRS, RESULT_PATH_EXP_3, NPCR_ANSWER_LENGTH
from copy import deepcopy
from model_training.train_xlmr import train_xlmr
from model_training.train_xlmr_sbert_core import train_xlmr as train_xlmr_sbert_core
from model_training.train_sbert import train_sbert
from model_training.train_npcr import train_npcr
from model_training.utils import read_data, get_device, eval_sbert, write_classification_statistics
from sentence_transformers import SentenceTransformer

#import nltk
#nltk.download('punkt')
#nltl.download('punkt_tab')

random_state = 3456786544


def run_full(dataset_path, dataset_name, id_column, prompt_column, answer_column, target_column, languages, run_suffix='', run_xlmr=True, run_sbert=True, run_pretrained=True, run_npcr_sbert=True, run_npcr_xlmr=True, run_xlmr_swap_sbert=True, run_sbert_swap_xlmr=True, bert_batch_size=32, sbert_batch_size=64):

    device = get_device()

    condition = 'combine_all_other'

    for prompt in os.listdir(dataset_path):

        # For each prompt - language pair, train a model
        for language in languages:

            torch.cuda.empty_cache()

            print(prompt, language)

            # Training is combination of data in other languages
            df_test = read_data(os.path.join(dataset_path, prompt, language, 'test.csv'), answer_column=answer_column, target_column=target_column)
            df_val = read_data(os.path.join(dataset_path, prompt, language, 'val.csv'), answer_column=answer_column, target_column=target_column)

            # Combine data of all *other* languages as training data
            df_train = pd.DataFrame()
            other_languages = deepcopy(languages)
            other_languages.remove(language)
            
            for other_language in other_languages:

                df_other = read_data(os.path.join(dataset_path, prompt, other_language, 'train.csv'), answer_column=answer_column, target_column=target_column)
                df_train = pd.concat([df_train, df_other])
            
            df_train.reset_index(inplace=True)


            if run_xlmr:

                run_path_bert = os.path.join(RESULT_PATH_EXP_3 + run_suffix, condition, dataset_name, prompt, language, 'XLMR')

                if not os.path.exists(os.path.join(run_path_bert, 'preds.csv')):

                    gold, xlmr_pred = train_xlmr(run_path_bert, df_train=df_train, df_val=df_val, df_test=df_test, answer_column=answer_column, target_column=target_column, num_epochs=BERT_NUM_EPOCHS, base_model=XLMR_BASE_MODEL, batch_size=bert_batch_size, save_model=False)
                    write_classification_statistics(filepath=run_path_bert, y_true=gold, y_pred=xlmr_pred)
                    df_train.to_csv(os.path.join(run_path_bert, 'train.csv'))
                    df_val.to_csv(os.path.join(run_path_bert, 'val.csv'))
                    df_test.to_csv(os.path.join(run_path_bert, 'test.csv'))


            if run_sbert:

                run_path_sbert = os.path.join(RESULT_PATH_EXP_3 + run_suffix, condition, dataset_name, prompt, language, 'SBERT')

                if not os.path.exists(os.path.join(run_path_sbert, 'preds.csv')):

                    gold, pred_max, pred_avg = train_sbert(run_path_sbert, df_train=df_train, df_val=df_val, df_test=df_test, id_column=id_column, answer_column=answer_column, target_column=target_column, base_model=SBERT_BASE_MODEL, num_epochs=SBERT_NUM_EPOCHS, batch_size=sbert_batch_size, do_warmup=False, save_model=False, num_pairs_per_example=SBERT_NUM_PAIRS, num_val_pairs=SBERT_NUM_VAL_PAIRS)
                    write_classification_statistics(filepath=run_path_sbert, y_true=gold, y_pred=pred_avg, suffix='')
                    write_classification_statistics(filepath=run_path_sbert, y_true=gold, y_pred=pred_max, suffix='_max')
                    df_train.to_csv(os.path.join(run_path_sbert, 'train.csv'))
                    df_val.to_csv(os.path.join(run_path_sbert, 'val.csv'))
                    df_test.to_csv(os.path.join(run_path_sbert, 'test.csv'))

            
            if run_xlmr_swap_sbert:

                run_path_bert_swap_sbert = os.path.join(RESULT_PATH_EXP_3 + run_suffix, condition, dataset_name, prompt, language, 'XLMR_SBERTcore')

                if not os.path.exists(os.path.join(run_path_bert_swap_sbert, 'preds.csv')):

                    gold, xlmr_swap_sbert_pred = train_xlmr_sbert_core(run_path_bert_swap_sbert, df_train=df_train, df_val=df_val, df_test=df_test, answer_column=answer_column, target_column=target_column, num_epochs=BERT_NUM_EPOCHS, batch_size=bert_batch_size, save_model=False, base_model='/models/'+SBERT_BASE_MODEL)

                    write_classification_statistics(filepath=run_path_bert_swap_sbert, y_true=gold, y_pred=xlmr_swap_sbert_pred)
                    df_train.to_csv(os.path.join(run_path_bert_swap_sbert, 'train.csv'))
                    df_val.to_csv(os.path.join(run_path_bert_swap_sbert, 'val.csv'))
                    df_test.to_csv(os.path.join(run_path_bert_swap_sbert, 'test.csv'))

            
            if run_sbert_swap_xlmr:

                run_path_sbert_swap_xlmr = os.path.join(RESULT_PATH_EXP_3 + run_suffix, condition, dataset_name, prompt, language, 'SBERT_XLMRcore')

                if not os.path.exists(os.path.join(run_path_sbert_swap_xlmr, 'preds.csv')):

                    gold, pred_max_xlmr_core, pred_avg_xlmr_core = train_sbert(run_path_sbert_swap_xlmr, answer_column=answer_column, id_column=id_column, target_column=target_column, df_train=df_train, df_val=df_val, df_test=df_test, base_model=XLMR_BASE_MODEL, num_epochs=SBERT_NUM_EPOCHS, batch_size=sbert_batch_size, do_warmup=False, save_model=False, num_pairs_per_example=SBERT_NUM_PAIRS, num_val_pairs=SBERT_NUM_VAL_PAIRS)
                    write_classification_statistics(filepath=run_path_sbert_swap_xlmr, y_true=gold, y_pred=pred_avg_xlmr_core, suffix='')
                    write_classification_statistics(filepath=run_path_sbert_swap_xlmr, y_true=gold, y_pred=pred_max_xlmr_core, suffix='_max')
                    df_train.to_csv(os.path.join(run_path_sbert_swap_xlmr, 'train.csv'))
                    df_val.to_csv(os.path.join(run_path_sbert_swap_xlmr, 'val.csv'))
                    df_test.to_csv(os.path.join(run_path_sbert_swap_xlmr, 'test.csv'))


            if run_npcr_xlmr:

                run_path_npcr_xlmr = os.path.join(RESULT_PATH_EXP_3 + run_suffix, condition, dataset_name, prompt, language, 'NPCR_XLMR')

                if not os.path.exists(os.path.join(run_path_npcr_xlmr, 'preds.csv')):
  
                    gold, npcr_xlmr_pred = train_npcr(target_path=run_path_npcr_xlmr, df_train=df_train, df_val=df_val, df_test=df_test, col_prompt=prompt_column, col_answer=answer_column, col_score=target_column, base_model=XLMR_BASE_MODEL, max_num=NPCR_ANSWER_LENGTH, training_with_same_score=True, save_model=False)

                    write_classification_statistics(filepath=run_path_npcr_xlmr, y_true=gold, y_pred=npcr_xlmr_pred)
                    df_train.to_csv(os.path.join(run_path_npcr_xlmr, 'train.csv'))
                    df_val.to_csv(os.path.join(run_path_npcr_xlmr, 'val.csv'))
                    df_test.to_csv(os.path.join(run_path_npcr_xlmr, 'test.csv'))


            if run_npcr_sbert:

                run_path_npcr_sbert = os.path.join(RESULT_PATH_EXP_3 + run_suffix, condition, dataset_name, prompt, language, 'NPCR_SBERT')

                if not os.path.exists(os.path.join(run_path_npcr_sbert, 'preds.csv')):
  
                    gold, npcr_sbert_pred = train_npcr(target_path=run_path_npcr_sbert, df_train=df_train, df_val=df_val, df_test=df_test, col_prompt=prompt_column, col_answer=answer_column, col_score=target_column, base_model=SBERT_BASE_MODEL, max_num=NPCR_ANSWER_LENGTH, training_with_same_score=True, save_model=False)

                    write_classification_statistics(filepath=run_path_npcr_sbert, y_true=gold, y_pred=npcr_sbert_pred)
                    df_train.to_csv(os.path.join(run_path_npcr_sbert, 'train.csv'))
                    df_val.to_csv(os.path.join(run_path_npcr_sbert, 'val.csv'))
                    df_test.to_csv(os.path.join(run_path_npcr_sbert, 'test.csv'))


            if run_pretrained:

                run_path_pretrained = os.path.join(RESULT_PATH_EXP_3 + run_suffix, condition, dataset_name, prompt, language, 'pretrained')

                if not os.path.exists(os.path.join(run_path_pretrained, 'preds.csv')):

                    if not os.path.exists(run_path_pretrained):

                        os.makedirs(run_path_pretrained)
                        
                    # Load pretrained model 
                    model = SentenceTransformer(SBERT_BASE_MODEL)
                    df_ref = pd.concat([df_train, df_val])
                    df_ref['embedding'] = df_ref[answer_column].apply(model.encode)
                    df_test['embedding'] = df_test[answer_column].apply(model.encode)

                    # Predict on within-test data
                    gold, pred_max_pretrained, pred_avg_pretrained = eval_sbert(run_path_pretrained, df_test, df_ref, id_column=id_column, answer_column=answer_column, target_column=target_column)
                    write_classification_statistics(filepath=run_path_pretrained, y_true=gold, y_pred=pred_avg_pretrained, suffix='')
                    write_classification_statistics(filepath=run_path_pretrained, y_true=gold, y_pred=pred_max_pretrained, suffix='_max')
                    df_train.to_csv(os.path.join(run_path_pretrained, 'train.csv'))
                    df_val.to_csv(os.path.join(run_path_pretrained, 'val.csv'))
                    df_test.to_csv(os.path.join(run_path_pretrained, 'test.csv'))


def run_downsampled(dataset_path, dataset_name, id_column, prompt_column, answer_column, target_column, languages, run_suffix='', run_xlmr=True, run_sbert=True, run_pretrained=True, run_xlmr_swap_sbert=True, run_sbert_swap_xlmr=True, run_npcr_xlmr=True, run_npcr_sbert=True, bert_batch_size=32, sbert_batch_size=64):

    condition = 'combine_downsampled'

    for prompt in os.listdir(dataset_path):

        # For each prompt - language pair, train a model
        for language in languages:
        #for language in ['en', 'he', 'it', 'ka', 'nb', 'pt', 'sl', 'sv', 'zh']:

            torch.cuda.empty_cache()

            print(prompt, language)

            # Training is combination of data in other languages
            df_test = read_data(os.path.join(dataset_path, prompt, language, 'test.csv'), answer_column=answer_column, target_column=target_column)
            df_val = read_data(os.path.join(dataset_path, prompt, language, 'val.csv'), answer_column=answer_column, target_column=target_column)

            # Combine data of all *other* languages as training data
            df_train = pd.DataFrame()
            other_languages = deepcopy(languages)
            other_languages.remove(language)

            # Just to grab label distribution and number of answers
            df_target_dist = read_data(os.path.join(dataset_path, prompt, language, 'train.csv'), answer_column=answer_column, target_column=target_column)
            label_dist = dict(df_target_dist[target_column].value_counts())
            
            for other_language in other_languages:
                
                df_other = read_data(os.path.join(dataset_path, prompt, other_language, 'train.csv'), answer_column=answer_column, target_column=target_column)
                
                # Sample to arrive at same number of answers as before (600 for epirls)
                num_train = len(df_target_dist)
                num_to_sample = int(num_train/len(other_languages))
                proportion_to_sample = num_to_sample/num_train
                
                for label, amount in label_dist.items():

                    amount = int(round(amount*proportion_to_sample, 0))

                    df_label = df_other[df_other[target_column] == label]
                    df_sample = df_label.sample(amount, random_state=random_state)
                    df_train = pd.concat([df_train, df_sample])

            df_train.reset_index(inplace=True)
            

            if run_xlmr:

                run_path_bert = os.path.join(RESULT_PATH_EXP_3 + run_suffix, condition, dataset_name, prompt, language, 'XLMR')

                if not os.path.exists(os.path.join(run_path_bert, 'preds.csv')):

                    gold, xlmr_pred = train_xlmr(run_path_bert, df_train=df_train, df_val=df_val, df_test=df_test, answer_column=answer_column, target_column=target_column, num_epochs=BERT_NUM_EPOCHS, base_model=XLMR_BASE_MODEL, batch_size=bert_batch_size, save_model=False)
                    write_classification_statistics(filepath=run_path_bert, y_true=gold, y_pred=xlmr_pred)
                    df_train.to_csv(os.path.join(run_path_bert, 'train.csv'))
                    df_val.to_csv(os.path.join(run_path_bert, 'vsl.csv'))
                    df_test.to_csv(os.path.join(run_path_bert, 'test.csv'))


            if run_sbert:

                run_path_sbert = os.path.join(RESULT_PATH_EXP_3 + run_suffix, condition, dataset_name, prompt, language, 'SBERT')

                if not os.path.exists(os.path.join(run_path_sbert, 'preds.csv')):

                    gold, pred_max, pred_avg = train_sbert(run_path_sbert, df_train=df_train, df_val=df_val, df_test=df_test, id_column=id_column, answer_column=answer_column, target_column=target_column, base_model=SBERT_BASE_MODEL, num_epochs=SBERT_NUM_EPOCHS, batch_size=sbert_batch_size, do_warmup=False, save_model=False, num_pairs_per_example=SBERT_NUM_PAIRS, num_val_pairs=SBERT_NUM_VAL_PAIRS)
                    write_classification_statistics(filepath=run_path_sbert, y_true=gold, y_pred=pred_avg, suffix='')
                    write_classification_statistics(filepath=run_path_sbert, y_true=gold, y_pred=pred_max, suffix='_max')
                    df_train.to_csv(os.path.join(run_path_sbert, 'train.csv'))
                    df_val.to_csv(os.path.join(run_path_sbert, 'val.csv'))
                    df_test.to_csv(os.path.join(run_path_sbert, 'test.csv'))

            
            if run_xlmr_swap_sbert:

                run_path_bert_swap_sbert = os.path.join(RESULT_PATH_EXP_3 + run_suffix, condition, dataset_name, prompt, language, 'XLMR_SBERTcore')

                if not os.path.exists(os.path.join(run_path_bert_swap_sbert, 'preds.csv')):

                    gold, xlmr_swap_sbert_pred = train_xlmr_sbert_core(run_path_bert_swap_sbert, df_train=df_train, df_val=df_val, df_test=df_test, answer_column=answer_column, target_column=target_column, num_epochs=BERT_NUM_EPOCHS, batch_size=bert_batch_size, save_model=False, base_model='/models/'+SBERT_BASE_MODEL)

                    write_classification_statistics(filepath=run_path_bert_swap_sbert, y_true=gold, y_pred=xlmr_swap_sbert_pred)
                    df_train.to_csv(os.path.join(run_path_bert_swap_sbert, 'train.csv'))
                    df_val.to_csv(os.path.join(run_path_bert_swap_sbert, 'val.csv'))
                    df_test.to_csv(os.path.join(run_path_bert_swap_sbert, 'test.csv'))

            
            if run_sbert_swap_xlmr:

                run_path_sbert_swap_xlmr = os.path.join(RESULT_PATH_EXP_3 + run_suffix, condition, dataset_name, prompt, language, 'SBERT_XLMRcore')

                if not os.path.exists(os.path.join(run_path_sbert_swap_xlmr, 'preds.csv')):

                    gold, pred_max_xlmr_core, pred_avg_xlmr_core = train_sbert(run_path_sbert_swap_xlmr, answer_column=answer_column, id_column=id_column, target_column=target_column, df_train=df_train, df_val=df_val, df_test=df_test, base_model=XLMR_BASE_MODEL, num_epochs=SBERT_NUM_EPOCHS, batch_size=sbert_batch_size, do_warmup=False, save_model=False, num_pairs_per_example=SBERT_NUM_PAIRS, num_val_pairs=SBERT_NUM_VAL_PAIRS)
                    write_classification_statistics(filepath=run_path_sbert_swap_xlmr, y_true=gold, y_pred=pred_avg_xlmr_core, suffix='')
                    write_classification_statistics(filepath=run_path_sbert_swap_xlmr, y_true=gold, y_pred=pred_max_xlmr_core, suffix='_max')
                    df_train.to_csv(os.path.join(run_path_sbert_swap_xlmr, 'train.csv'))
                    df_val.to_csv(os.path.join(run_path_sbert_swap_xlmr, 'val.csv'))
                    df_test.to_csv(os.path.join(run_path_sbert_swap_xlmr, 'test.csv'))


            if run_npcr_xlmr:

                run_path_npcr_xlmr = os.path.join(RESULT_PATH_EXP_3 + run_suffix, condition, dataset_name, prompt, language, 'NPCR_XLMR')

                if not os.path.exists(os.path.join(run_path_npcr_xlmr, 'preds.csv')):
  
                    gold, npcr_xlmr_pred = train_npcr(target_path=run_path_npcr_xlmr, df_train=df_train, df_val=df_val, df_test=df_test, col_prompt=prompt_column, col_answer=answer_column, col_score=target_column, base_model=XLMR_BASE_MODEL, max_num=NPCR_ANSWER_LENGTH, training_with_same_score=True, save_model=False)

                    write_classification_statistics(filepath=run_path_npcr_xlmr, y_true=gold, y_pred=npcr_xlmr_pred)
                    df_train.to_csv(os.path.join(run_path_npcr_xlmr, 'train.csv'))
                    df_val.to_csv(os.path.join(run_path_npcr_xlmr, 'val.csv'))
                    df_test.to_csv(os.path.join(run_path_npcr_xlmr, 'test.csv'))


            if run_npcr_sbert:

                run_path_npcr_sbert = os.path.join(RESULT_PATH_EXP_3 + run_suffix, condition, dataset_name, prompt, language, 'NPCR_SBERT')

                if not os.path.exists(os.path.join(run_path_npcr_sbert, 'preds.csv')):
  
                    gold, npcr_sbert_pred = train_npcr(target_path=run_path_npcr_sbert, df_train=df_train, df_val=df_val, df_test=df_test, col_prompt=prompt_column, col_answer=answer_column, col_score=target_column, base_model=SBERT_BASE_MODEL, max_num=NPCR_ANSWER_LENGTH, training_with_same_score=True, save_model=False)

                    write_classification_statistics(filepath=run_path_npcr_sbert, y_true=gold, y_pred=npcr_sbert_pred)
                    df_train.to_csv(os.path.join(run_path_npcr_sbert, 'train.csv'))
                    df_val.to_csv(os.path.join(run_path_npcr_sbert, 'val.csv'))
                    df_test.to_csv(os.path.join(run_path_npcr_sbert, 'test.csv'))


            if run_pretrained:

                run_path_pretrained = os.path.join(RESULT_PATH_EXP_3 + run_suffix, condition, dataset_name, prompt, language, 'pretrained')

                if not os.path.exists(os.path.join(run_path_pretrained, 'preds.csv')):

                    if not os.path.exists(run_path_pretrained):
                        os.makedirs(run_path_pretrained)
                        
                    # Load pretrained model 
                    model = SentenceTransformer(SBERT_BASE_MODEL)
                    df_ref = pd.concat([df_train, df_val])
                    df_ref['embedding'] = df_ref[answer_column].apply(model.encode)
                    df_test['embedding'] = df_test[answer_column].apply(model.encode)

                    # Predict on within-test data
                    gold, pred_max_pretrained, pred_avg_pretrained = eval_sbert(run_path_pretrained, df_test=df_test, df_ref=df_ref, id_column=id_column, answer_column=answer_column, target_column=target_column)

                    write_classification_statistics(filepath=run_path_pretrained, y_true=gold, y_pred=pred_avg_pretrained, suffix='')
                    write_classification_statistics(filepath=run_path_pretrained, y_true=gold, y_pred=pred_max_pretrained, suffix='_max')
                    df_train.to_csv(os.path.join(run_path_pretrained, 'train.csv'))
                    df_val.to_csv(os.path.join(run_path_pretrained, 'val.csv'))
                    df_test.to_csv(os.path.join(run_path_pretrained, 'test.csv'))


# Downsampled
for run in ['_RUN1', '_RUN2', '_RUN3']:

    for dataset in [EPIRLS, ASAP_T]:

        run_downsampled(
            dataset_path=dataset['dataset_path'], 
            dataset_name=dataset['dataset_name'], 
            id_column=dataset['id_column'], 
            prompt_column=dataset['prompt_column'],
            answer_column=dataset['answer_column'], 
            target_column=dataset['target_column'], 
            languages=dataset['languages'], 
            run_suffix=run,
            run_xlmr=True, 
            run_sbert=True, 
            run_pretrained=True, 
            run_npcr_sbert=True, 
            run_npcr_xlmr=True, 
            run_xlmr_swap_sbert=True, 
            run_sbert_swap_xlmr=True,
            )


# Full
for run in ['_RUN1', '_RUN2', '_RUN3']:
    
    for dataset in [EPIRLS, ASAP_T]:

        run_full(
            dataset_path=dataset['dataset_path'], 
            dataset_name=dataset['dataset_name'], 
            id_column=dataset['id_column'], 
            prompt_column=dataset['prompt_column'],
            answer_column=dataset['answer_column'], 
            target_column=dataset['target_column'], 
            languages=dataset['languages'], 
            run_suffix=run, 
            run_xlmr=True,
            run_sbert=True,
            run_pretrained=True,
            run_npcr_xlmr=True,
            run_npcr_sbert=True,
            run_xlmr_swap_sbert=True,
            run_sbert_swap_xlmr=True,
            )
