import logging
import os
import random
import shutil
import sys
import torch
import glob

import pandas as pd

from datetime import datetime
from model_training.utils import encode_labels, eval_sbert, get_device, cross_dataframes, get_preds_from_pairs
from sentence_transformers.evaluation import SimilarityFunction
from torch.utils.data import DataLoader
from transformers import EarlyStoppingCallback
from tqdm import tqdm

random_state = 3456478
from copy import deepcopy
import datasets
from sentence_transformers import SentenceTransformer, losses, evaluation, SentenceTransformerTrainer, SentenceTransformerTrainingArguments
from config import ANSWER_LENGTH, SBERT_NUM_VAL_PAIRS



def get_paired_data_from_dataframes(df_train, df_val, df_test, target_column, id_column='submission_id', answer_column='text'):
    
    df_train_pairs = cross_dataframes(df=df_train, df_ref=df_train)
    df_val_pairs = cross_dataframes(df=df_val, df_ref=df_train)
    df_test_pairs = cross_dataframes(df=df_test, df_ref=df_train)

    for df_split in [df_train_pairs, df_val_pairs, df_test_pairs]:
    
        df_split[target_column] = (df_split[target_column+'_1'] == df_split[target_column+'_2']).astype(int)

    return df_train_pairs, df_val_pairs, df_test_pairs



def train_sbert(run_path, df_train, df_val, df_test, answer_column, target_column, id_column, base_model, batch_size, num_epochs, num_training_pairs_per_example=None, save_model=False, num_val_pairs_per_example=SBERT_NUM_VAL_PAIRS):
    
    # Clear logger from previous runs
    log = logging.getLogger()
    handlers = log.handlers[:]

    for handler in handlers:
    
        log.removeHandler(handler)
        handler.close()

    device = 'cuda'

    print('**** Running SBERT on:', device)

    if not os.path.exists(run_path):

        os.makedirs(run_path)
    
    logging.basicConfig(filename=os.path.join(run_path, datetime.now().strftime('logs_%H_%M_%d_%m_%Y.log')), filemode='w', level=logging.DEBUG)

    logging.info('base_model:\t' + base_model)
    logging.info('number of epochs:\t' + str(num_epochs))
    logging.info('batch size:\t' + str(batch_size))

    logging.info('num train:\t' + str(len(df_train)))
    logging.info('num val:\t' + str(len(df_val)))
    logging.info('num test:\t' + str(len(df_test)))
    logging.info('num training pairs per example:\t' + str(num_training_pairs_per_example))
    logging.info('num validation pairs per example:\t' + str(num_val_pairs_per_example))

    start_time = datetime.now()

    # Where to store finetuned model
    model_path = os.path.join(run_path, "finetuned_model")

    try:

        model = SentenceTransformer(base_model, device=device)

    except:

        # Above sometimes crashes, but this gives equivalent result
        transformer = models.Transformer(base_model)
        pooling = models.Pooling(transformer.get_word_embedding_dimension(), pooling_mode="mean")
        model = SentenceTransformer(modules=[transformer, pooling], device=device)
    
    model.max_seq_length=ANSWER_LENGTH

    df_train_paired, df_val_paired, df_test_paired = get_paired_data_from_dataframes(df_train=df_train, df_val=df_val, df_test=df_test, target_column=target_column)
    print(len(df_train_paired))

    # Downsample training
    if num_training_pairs_per_example is not None:
        
        if len(df_train) * num_training_pairs_per_example * num_epochs < len(df_train_paired):

            df_train_paired = df_train_paired.sample(len(df_train) * num_epochs * num_training_pairs_per_example, random_state=random_state) 

    if num_val_pairs_per_example is not None:

        df_val_paired = df_val_paired.sample(num_val_pairs_per_example * len(df_val), random_state=random_state)

    train_dataset = datasets.Dataset.from_dict({
        'text1': list(df_train_paired[answer_column + '_1']),
        'text2': list(df_train_paired[answer_column + '_2']),
        'label': list(df_train_paired[target_column])
    })
    eval_dataset = datasets.Dataset.from_dict({
        'text1': list(df_val_paired[answer_column + '_1']),
        'text2': list(df_val_paired[answer_column + '_2']),
        'label': list(df_val_paired[target_column])
    })


    loss = losses.OnlineContrastiveLoss(model)

    early_stop = EarlyStoppingCallback(early_stopping_patience=3)

    args = SentenceTransformerTrainingArguments(
        output_dir=run_path,
        load_best_model_at_end=True,
        per_device_train_batch_size=batch_size,
        per_device_eval_batch_size=batch_size,
        eval_strategy='steps',
        save_strategy='steps',
        eval_steps=int(len(df_train)/batch_size),
        max_steps=int((len(df_train)/batch_size)*num_epochs),
        save_steps=int(len(df_train)/batch_size),
        # num_train_epochs=num_epochs,
        # eval_strategy='epoch',
        # save_strategy='epoch',
        # metric_for_best_model='spearman_cosine'
    )

    dev_evaluator = evaluation.EmbeddingSimilarityEvaluator(df_val_paired[answer_column + "_1"].tolist(), df_val_paired[answer_column + "_2"].tolist(), df_val_paired[target_column].tolist(), write_csv=True)
    dev_evaluator(model)

    trainer = SentenceTransformerTrainer(
        model=model,
        args=args,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        loss=loss,
        evaluator=dev_evaluator,
        callbacks=[early_stop]
    )

    trainer.train()

    logging.info("SBERT number of epochs: "+str(num_epochs))
    logging.info("SBERT batch size: "+str(batch_size))
    logging.info("SBERT evaluator: "+str(dev_evaluator.__class__)+" Batch size: "+str(dev_evaluator.batch_size)+" Main similarity:"+str(dev_evaluator.primary_metric))
    logging.info("SBERT loss: "+str(loss.__class__))
    
    # Obtain test predictions
    model.eval()
    
    with torch.no_grad():

        df_train_copy = deepcopy(df_train)
        df_test_copy = deepcopy(df_test)

        df_train_copy['embedding'] = df_train_copy[answer_column].apply(model.encode)
        df_test_copy['embedding'] = df_test_copy[answer_column].apply(model.encode)

    df_inference = cross_dataframes(df=df_test_copy, df_ref=df_train_copy)
    df_inference['sim'] = df_inference.apply(lambda row: row['embedding_1'] @ row['embedding_2'], axis=1)
    test_answers, test_true_scores, test_predictions, test_predictions_max = get_preds_from_pairs(df=df_inference, id_column=id_column+'_1', pred_column='sim', ref_label_column=target_column+'_2', true_label_column=target_column+'_1')

    df_test_aggregated = pd.DataFrame({id_column: test_answers, 'pred': test_predictions, target_column: test_true_scores, 'pred_max': test_predictions_max})
    df_test_aggregated.to_csv(os.path.join(run_path, 'preds.csv'))

    for checkpoint in glob.glob(os.path.join(run_path, 'checkpoint*')):
        shutil.rmtree(checkpoint)

    logging.info('Training duration:\t' + str(datetime.now() - start_time))

    model.save_pretrained(model_path)

    return test_true_scores, test_predictions_max, test_predictions
