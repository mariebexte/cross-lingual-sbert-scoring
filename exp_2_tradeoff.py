import os
import shutil
import sys
import torch

import pandas as pd

from config import EPIRLS, SBERT_BASE_MODEL, XLMR_BASE_MODEL, SBERT_NUM_EPOCHS, BERT_NUM_EPOCHS, NPCR_NUM_EPOCHS, SBERT_BATCH_SIZE, BERT_BATCH_SIZE, NPCR_BATCH_SIZE, ANSWER_LENGTH, SBERT_NUM_PAIRS, SBERT_NUM_VAL_PAIRS, RESULT_PATH_EXP_2
from copy import deepcopy
from model_training.train_xlmr import train_xlmr
from model_training.train_xlmr_sbert_core import train_xlmr as train_xlmr_sbert_core
from model_training.train_sbert import train_sbert
from model_training.train_npcr import train_npcr
from model_training.utils import read_data, get_device, write_classification_statistics


random_state = 56398
amounts = [15, 35, 75, 150, 300]


def run_exp(dataset_path, dataset_name, id_column, prompt_column, answer_column, target_column, languages, run_xlmr=True, run_sbert=True, run_xlmr_swap_sbert=True, run_sbert_swap_xlmr=True, run_npcr_xlmr=True, run_npcr_sbert=True):

    device = get_device()

    for prompt in os.listdir(dataset_path):

        # This is the base language: Train base models that will later be finetuned further with target data
        for base_language in languages:

            torch.cuda.empty_cache()

            print(prompt, base_language)

            # Read data for training
            df_val_base = read_data(os.path.join(dataset_path, prompt, base_language, 'val.csv'), answer_column=answer_column, target_column=target_column)
            df_test_base = read_data(os.path.join(dataset_path, prompt, base_language, 'test.csv'), answer_column=answer_column, target_column=target_column)

            # This will be copied and downsampled multiple times
            df_train_base = read_data(os.path.join(dataset_path, prompt, base_language, 'train.csv'), answer_column=answer_column, target_column=target_column)
            
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
                
                df_train_base_reduced.reset_index(inplace=True)

                ## Train the base model for this amount

                if run_xlmr:

                    run_path_bert = os.path.join(RESULT_PATH_EXP_2, dataset_name, 'base_models', prompt, base_language, str(num_target), 'XLMR')

                    if not os.path.exists(os.path.join(run_path_bert, 'preds.csv')):

                        gold, xlmr_pred = train_xlmr(run_path_bert, df_train=df_train_base_reduced, df_val=df_val_base, df_test=df_test_base, answer_column=answer_column, target_column=target_column, base_model=XLMR_BASE_MODEL, num_epochs=BERT_NUM_EPOCHS, batch_size=BERT_BATCH_SIZE, save_model=True)
                        
                        write_classification_statistics(filepath=run_path_bert, y_true=gold, y_pred=xlmr_pred)
                        
                        df_train_base_reduced.to_csv(os.path.join(run_path_bert, 'train.csv'))
                        df_val_base.to_csv(os.path.join(run_path_bert, 'val.csv'))
                        df_test_base.to_csv(os.path.join(run_path_bert, 'test.csv'))

                if run_sbert:

                    run_path_sbert = os.path.join(RESULT_PATH_EXP_2, dataset_name, 'base_models', prompt, base_language, str(num_target), 'SBERT')
                    
                    if not os.path.exists(os.path.join(run_path_sbert, 'preds.csv')):

                        gold, pred_max, pred_avg = train_sbert(run_path_sbert, df_train=df_train_base_reduced, df_val=df_val_base, df_test=df_test_base, answer_column=answer_column, target_column=target_column, base_model=SBERT_BASE_MODEL, num_epochs=SBERT_NUM_EPOCHS, batch_size=SBERT_BATCH_SIZE, do_warmup=False, save_model=True, num_pairs_per_example=SBERT_NUM_PAIRS, num_val_pairs=SBERT_NUM_VAL_PAIRS)

                        write_classification_statistics(filepath=run_path_sbert, y_true=gold, y_pred=pred_avg, suffix='')
                        write_classification_statistics(filepath=run_path_sbert, y_true=gold, y_pred=pred_max, suffix='_max')

                        df_train_base_reduced.to_csv(os.path.join(run_path_sbert, 'train.csv'))
                        df_val_base.to_csv(os.path.join(run_path_sbert, 'val.csv'))
                        df_test_base.to_csv(os.path.join(run_path_sbert, 'test.csv'))
                

                if run_xlmr_swap_sbert:

                    run_path_bert_swap_sbert = os.path.join(RESULT_PATH_EXP_2, dataset_name, 'base_models', prompt, base_language, str(num_target), 'XLMR_SBERTcore')

                    if not os.path.exists(os.path.join(run_path_bert_swap_sbert, 'preds.csv')):

                        gold, xlmr_swap_sbert_pred = train_xlmr_sbert_core(run_path_bert_swap_sbert, df_train=df_train_base_reduced, df_val=df_val_base, df_test=df_test_base, answer_column=answer_column, target_column=target_column, num_epochs=BERT_NUM_EPOCHS, batch_size=BERT_BATCH_SIZE, save_model=True, base_model='/models/'+SBERT_BASE_MODEL)
                                                
                        write_classification_statistics(filepath=run_path_bert_swap_sbert, y_true=gold, y_pred=xlmr_swap_sbert_pred)
                        
                        df_train_base_reduced.to_csv(os.path.join(run_path_bert_swap_sbert, 'train.csv'))
                        df_val_base.to_csv(os.path.join(run_path_bert_swap_sbert, 'val.csv'))
                        df_test_base.to_csv(os.path.join(run_path_bert_swap_sbert, 'test.csv'))

                
                if run_sbert_swap_xlmr:

                    run_path_sbert_swap_xlmr = os.path.join(RESULT_PATH_EXP_2, dataset_name, 'base_models', prompt, base_language, str(num_target), 'SBERT_XLMRcore')
                    
                    if not os.path.exists(os.path.join(run_path_sbert_swap_xlmr, 'preds.csv')):

                        gold, pred_max_xlmr_core, pred_avg_xlmr_core = train_sbert(run_path_sbert_swap_xlmr, answer_column=answer_column, id_column=id_column, target_column=target_column, df_train=df_train_base_reduced, df_val=df_val_base, df_test=df_test_base, base_model=XLMR_BASE_MODEL, num_epochs=SBERT_NUM_EPOCHS, batch_size=SBERT_BATCH_SIZE, do_warmup=False, save_model=True, num_pairs_per_example=SBERT_NUM_PAIRS, num_val_pairs=SBERT_NUM_VAL_PAIRS)
                        
                        write_classification_statistics(filepath=run_path_sbert_swap_xlmr, y_true=gold, y_pred=pred_avg_xlmr_core, suffix='')
                        write_classification_statistics(filepath=run_path_sbert_swap_xlmr, y_true=gold, y_pred=pred_max_xlmr_core, suffix='_max')

                        df_train_base_reduced.to_csv(os.path.join(run_path_sbert_swap_xlmr, 'train.csv'))
                        df_val_base.to_csv(os.path.join(run_path_sbert_swap_xlmr, 'val.csv'))
                        df_test_base.to_csv(os.path.join(run_path_sbert_swap_xlmr, 'test.csv'))


                if run_npcr_xlmr:

                    run_path_npcr_xlmr = os.path.join(RESULT_PATH_EXP_2, dataset_name, 'base_models', prompt, base_language, str(num_target), 'NPCR_XLMR')

                    if not os.path.exists(os.path.join(run_path_npcr_xlmr, 'preds.csv')):

                        gold, npcr_xlmr_pred = train_npcr(target_path=run_path_npcr_xlmr, df_train=df_train_base_reduced, df_val=df_val_base, df_test=df_test_base, col_prompt=prompt_column, col_answer=answer_column, col_score=target_column, base_model=XLMR_BASE_MODEL, max_num=ANSWER_LENGTH, num_epochs=NPCR_NUM_EPOCHS, batch_size=NPCR_BATCH_SIZE, training_with_same_score=True, save_model=True)
                                                
                        write_classification_statistics(filepath=run_path_npcr_xlmr, y_true=gold, y_pred=npcr_xlmr_pred)
                        
                        df_train_base_reduced.to_csv(os.path.join(run_path_npcr_xlmr, 'train.csv'))
                        df_val_base.to_csv(os.path.join(run_path_npcr_xlmr, 'val.csv'))
                        df_test_base.to_csv(os.path.join(run_path_npcr_xlmr, 'test.csv'))


                if run_npcr_sbert:

                    run_path_npcr_sbert = os.path.join(RESULT_PATH_EXP_2, dataset_name, 'base_models', prompt, base_language, str(num_target), 'NPCR_SBERT')

                    if not os.path.exists(os.path.join(run_path_npcr_sbert, 'preds.csv')):

                        gold, npcr_sbert_pred = train_npcr(target_path=run_path_npcr_sbert, df_train=df_train_base_reduced, df_val=df_val_base, df_test=df_test_base, col_prompt=prompt_column, col_answer=answer_column, col_score=target_column, base_model=SBERT_BASE_MODEL, max_num=ANSWER_LENGTH, num_epochs=NPCR_NUM_EPOCHS, batch_size=NPCR_BATCH_SIZE, training_with_same_score=True, save_model=True)
                                                
                        write_classification_statistics(filepath=run_path_npcr_sbert, y_true=gold, y_pred=npcr_sbert_pred)
                        
                        df_train_base_reduced.to_csv(os.path.join(run_path_npcr_sbert, 'train.csv'))
                        df_val_base.to_csv(os.path.join(run_path_npcr_sbert, 'val.csv'))
                        df_test_base.to_csv(os.path.join(run_path_npcr_sbert, 'test.csv'))


                ## For each target language, finetune the base model with the amount specified

                target_languages = deepcopy(languages)
                target_languages.remove(base_language)

                for target_language in target_languages:

                    # Sample required amount of training data
                    df_train_target = read_data(os.path.join(dataset_path, prompt, target_language, 'train.csv'), target_column=target_column, answer_column=answer_column)
                    df_val_target = read_data(os.path.join(dataset_path, prompt, target_language, 'val.csv'), target_column=target_column, answer_column=answer_column)
                    df_test_target = read_data(os.path.join(dataset_path, prompt, target_language, 'test.csv'), target_column=target_column, answer_column=answer_column)

                    df_train_target_sample = pd.DataFrame()

                    for label, amount in label_dist.items():

                        amount = int(round(sample_ratio*amount, 0))

                        # Collect this amount of target language
                        df_train_target_sample_label = df_train_target[df_train_target[target_column] == label].sample(amount, random_state=random_state)
                        df_train_target_sample = pd.concat([df_train_target_sample, df_train_target_sample_label])
                        df_train_target_sample.reset_index(inplace=True)


                    if run_xlmr:

                        run_path_bert_finetune = os.path.join(RESULT_PATH_EXP_2, dataset_name, prompt, target_language, 'XLMR', base_language, str(num_target))

                        if not os.path.exists(os.path.join(run_path_bert_finetune, 'preds.csv')):
                        
                            finetuned_model_xlmr = os.path.join(run_path_bert, 'best_model')

                            gold, xlmr_pred_finetune = train_xlmr(run_path_bert_finetune, df_train=df_train_target_sample, df_val=df_val_target, df_test=df_test_target, answer_column=answer_column, target_column=target_column, num_epochs=BERT_NUM_EPOCHS, batch_size=BERT_BATCH_SIZE, base_model=finetuned_model_xlmr, save_model=False)
                            
                            write_classification_statistics(filepath=run_path_bert_finetune, y_true=gold, y_pred=xlmr_pred_fineteune)
                            
                            df_train_target_sample.to_csv(os.path.join(run_path_bert_finetune, 'train.csv'))
                            df_val_target.to_csv(os.path.join(run_path_bert_finetune, 'val.csv'))
                            df_test_target.to_csv(os.path.join(run_path_bert_finetune, 'test.csv'))
                

                    if run_sbert:

                        run_path_sbert_finetune = os.path.join(RESULT_PATH_EXP_2, dataset_name, prompt, target_language, 'SBERT', base_language, str(num_target))
                        
                        if not os.path.exists(os.path.join(run_path_sbert_finetune, 'preds.csv')):

                            finetuned_model_sbert = os.path.join(run_path_sbert, 'finetuned_model')

                            gold, pred_max_finetune, pred_avg_finetune = train_sbert(run_path_sbert_finetune, df_train=df_train_target_sample, df_val=df_val_target, df_test=df_test_target, answer_column=answer_column, target_column=target_column, base_model=finetuned_model_sbert, num_epochs=SBERT_NUM_EPOCHS, batch_size=SBERT_BATCH_SIZE, do_warmup=False, save_model=False, num_pairs_per_example=SBERT_NUM_PAIRS, num_val_pairs=SBERT_NUM_VAL_PAIRS)

                            write_classification_statistics(filepath=run_path_sbert_finetune, y_true=gold, y_pred=pred_avg_finetune, suffix='')
                            write_classification_statistics(filepath=run_path_sbert_finetune, y_true=gold, y_pred=pred_max_finetune, suffix='_max')

                            df_train_target_sample.to_csv(os.path.join(run_path_sbert_finetune, 'train.csv'))
                            df_val_target.to_csv(os.path.join(run_path_sbert_finetune, 'val.csv'))
                            df_test_target.to_csv(os.path.join(run_path_sbert_finetune, 'test.csv'))

                    
                    if run_xlmr_swap_sbert:

                        run_path_bert_swap_sbert_finetune = os.path.join(RESULT_PATH_EXP_2, dataset_name, prompt, target_language, 'XLMR_SBERTcore', base_language, str(num_target))

                        if not os.path.exists(os.path.join(run_path_bert_swap_sbert_finetune, 'preds.csv')):
                        
                            finetuned_model_xlmr_swap_sbert = os.path.join(run_path_bert_swap_sbert, 'best_model', 'pytorch_model.bin')

                            gold, xlmr_swap_sbert_pred = train_xlmr_sbert_core(run_path_bert_swap_sbert_finetune, df_train=df_train_target_sample, df_val=df_val_target, df_test=df_test_target, answer_column=answer_column, target_column=target_column, num_epochs=BERT_NUM_EPOCHS, batch_size=BERT_BATCH_SIZE, base_model=finetuned_model_xlmr_swap_sbert, save_model=False, from_pretrained=True)
                            
                            write_classification_statistics(filepath=run_path_bert_swap_sbert_finetune, y_true=gold, y_pred=xlmr_swap_sbert_pred)
                            
                            df_train_target_sample.to_csv(os.path.join(run_path_bert_swap_sbert_finetune, 'train.csv'))
                            df_val_target.to_csv(os.path.join(run_path_bert_swap_sbert_finetune, 'val.csv'))
                            df_test_target.to_csv(os.path.join(run_path_bert_swap_sbert_finetune, 'test.csv'))
                    

                    if run_sbert_swap_xlmr:

                        run_path_sbert_swap_xlmr_finetune = os.path.join(RESULT_PATH_EXP_2, dataset_name, prompt, target_language, 'SBERT_XLMRcore', base_language, str(num_target))
                        
                        if not os.path.exists(os.path.join(run_path_sbert_swap_xlmr_finetune, 'preds.csv')):

                            finetuned_model_sbert_swap_xlmr = os.path.join(run_path_sbert_swap_xlmr, 'finetuned_model')

                            gold, pred_max_xlmr_core, pred_avg_xlmr_core = train_sbert(run_path_sbert_swap_xlmr_finetune, df_train=df_train_target_sample, df_val=df_val_target, df_test=df_test_target, answer_column=answer_column, target_column=target_column, base_model=finetuned_model_sbert_swap_xlmr, num_epochs=SBERT_NUM_EPOCHS, batch_size=SBERT_BATCH_SIZE, do_warmup=False, save_model=False, num_pairs_per_example=SBERT_NUM_PAIRS, num_val_pairs=SBERT_NUM_VAL_PAIRS)

                            write_classification_statistics(filepath=run_path_sbert_swap_xlmr_finetune, y_true=gold, y_pred=pred_avg_xlmr_core, suffix='')
                            write_classification_statistics(filepath=run_path_sbert_swap_xlmr_finetune, y_true=gold, y_pred=pred_max_xlmr_core, suffix='_max')

                            df_train_target_sample.to_csv(os.path.join(run_path_sbert_swap_xlmr_finetune, 'train.csv'))
                            df_val_target.to_csv(os.path.join(run_path_sbert_swap_xlmr_finetune, 'val.csv'))
                            df_test_target.to_csv(os.path.join(run_path_sbert_swap_xlmr_finetune, 'test.csv'))

                    
                    if run_npcr_xlmr:

                        run_path_npcr_xlmr_finetune = os.path.join(RESULT_PATH_EXP_2, dataset_name, prompt, target_language, 'NPCR_XLMR', base_language, str(num_target))

                        if not os.path.exists(os.path.join(run_path_npcr_xlmr_finetune, 'preds.csv')):
                        
                            finetuned_model_npcr_xlmr = os.path.join(run_path_npcr_xlmr, 'best_model')

                            gold, npcr_xlmr_pred_finetune = train_npcr(target_path=run_path_npcr_xlmr_finetune, df_train=df_train_target_sample, df_val=df_val_target, df_test=df_test_target, col_prompt=prompt_column, col_answer=answer_column, col_score=target_column, base_model=XLMR_BASE_MODEL, max_num=ANSWER_LENGTH, num_epochs=NPCR_NUM_EPOCHS, batch_size=NPCR_BATCH_SIZE, training_with_same_score=True, finetuned_model=finetuned_model_npcr_xlmr)                            
                            write_classification_statistics(filepath=run_path_npcr_xlmr_finetune, y_true=gold, y_pred=npcr_xlmr_pred_finetune)
                            
                            df_train_target_sample.to_csv(os.path.join(run_path_npcr_xlmr_finetune, 'train.csv'))
                            df_val_target.to_csv(os.path.join(run_path_npcr_xlmr_finetune, 'val.csv'))
                            df_test_target.to_csv(os.path.join(run_path_npcr_xlmr_finetune, 'test.csv'))


                    if run_npcr_sbert:

                        run_path_npcr_sbert_finetune = os.path.join(RESULT_PATH_EXP_2, dataset_name, prompt, target_language, 'NPCR_SBERT', base_language, str(num_target))

                        if not os.path.exists(os.path.join(run_path_npcr_sbert_finetune, 'preds.csv')):
                        
                            finetuned_model_npcr_sbert = os.path.join(run_path_npcr_sbert, 'best_model')

                            gold, npcr_sbert_pred_finetune = train_npcr(target_path=run_path_npcr_sbert_finetune, df_train=df_train_target_sample, df_val=df_val_target, df_test=df_test_target, col_prompt=prompt_column, col_answer=answer_column, col_score=target_column, base_model=SBERT_BASE_MODEL, max_num=ANSWER_LENGTH, num_epochs=NPCR_NUM_EPOCHS, batch_size=NPCR_BATCH_SIZE, training_with_same_score=True, finetuned_model=finetuned_model_npcr_sbert)                            
                            write_classification_statistics(filepath=run_path_npcr_sbert_finetune, y_true=gold, y_pred=npcr_sbert_pred_finetune)
                            
                            df_train_target_sample.to_csv(os.path.join(run_path_npcr_sbert_finetune, 'train.csv'))
                            df_val_target.to_csv(os.path.join(run_path_npcr_sbert_finetune, 'val.csv'))
                            df_test_target.to_csv(os.path.join(run_path_npcr_sbert_finetune, 'test.csv'))
                     
                            
                if os.path.exists(os.path.join(run_path_bert, 'best_model')):
                    shutil.rmtree(os.path.join(run_path_bert, 'best_model'))

                if os.path.exists(os.path.join(run_path_sbert, 'finetuned_model')):
                    shutil.rmtree(os.path.join(run_path_sbert, 'finetuned_model'))

                if os.path.exists(os.path.join(run_path_bert_swap_sbert, 'best_model')):
                    shutil.rmtree(os.path.join(run_path_bert_swap_sbert, 'best_model'))

                if os.path.exists(os.path.join(run_path_sbert_swap_xlmr, 'finetuned_model')):
                    shutil.rmtree(os.path.join(run_path_sbert_swap_xlmr, 'finetuned_model'))

                if os.path.exists(os.path.join(run_path_npcr_xlmr, 'best_model')):
                    shutil.rmtree(os.path.join(run_path_npcr_xlmr, 'best_model'))

                if os.path.exists(os.path.join(run_path_npcr_sbert, 'best_model')):
                    shutil.rmtree(os.path.join(run_path_npcr_sbert, 'best_model'))


run_exp(
    dataset_path=EPIRLS['dataset_path'],
    dataset_name=EPIRLS['dataset_name'],
    id_column=EPIRLS['id_column'],
    prompt_column=EPIRLS['prompt_column']
    answer_column=EPIRLS['answer_column'],
    target_column=EPIRLS['target_column'],
    languages=EPIRLS['languages'],
    run_xlmr=True,
    run_sbert=True,
    run_xlmr_swap_sbert=True,
    run_sbert_swap_xlmr=True,
    run_npcr_xlmr=True,
    run_npcr_sbert=True,
    )
