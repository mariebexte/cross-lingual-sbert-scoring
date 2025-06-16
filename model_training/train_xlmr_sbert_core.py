import gc
import logging
import os
import shutil
import sys
import torch

import pandas as pd
import numpy as np

from datetime import datetime
from transformers import Trainer, TrainingArguments, AutoConfig, EarlyStoppingCallback
from model_training.sbert_for_classification import SbertForSequenceClassification
from model_training.utils import encode_labels, get_device, Dataset, compute_metrics, WriteCsvCallback, GetTestPredictionsCallback


def train_xlmr(run_path, df_train, df_val, df_test, answer_column, target_column, base_model, num_epochs, batch_size, save_model=False, from_pretrained=False):

    gc.collect()

    # Clear logger from previous runs
    log = logging.getLogger()
    handlers = log.handlers[:]

    for handler in handlers:

        log.removeHandler(handler)
        handler.close()

    device = get_device()

    print('**** Running XLMR with SBERT base on:', device)

    if not os.path.exists(run_path):

        os.makedirs(run_path)

    logging.basicConfig(filename=os.path.join(run_path, datetime.now().strftime('logs_%H_%M_%d_%m_%Y.log')), filemode='w', level=logging.DEBUG)
    logging.info('base_model:\t' + base_model)
    logging.info('number of epochs:\t' + str(num_epochs))
    logging.info('batch size:\t' + str(batch_size))

    logging.info('num train:\t' + str(len(df_train)))
    logging.info('num val:\t' + str(len(df_val)))
    logging.info('num test:\t' + str(len(df_test)))
    
    start_time = datetime.now()

    # If all training instances have the same label: Return this label as prediction for all testing instances
    if len(df_train[target_column].unique()) == 1:

        target_label = list(df_train[target_column].unique())
        logging.warn("All training instances have the same label '"+str(target_label[0])+"'. Predicting this label for all testing instances!")
        print("All training instances have the same label '"+str(target_label[0])+"'. Predicting this label for all testing instances!")
        
        return target_label*len(df_test)

    # Model evaluation throws error if val/test data contains more labels than train
    labels_in_training = df_train[target_column].unique().tolist()
    labels_in_validation = df_val[target_column].unique().tolist()
    labels_in_test = df_test[target_column].unique().tolist()
    label_set = set(labels_in_training + labels_in_validation + labels_in_test)

    # If the labels are not integers: Map them to integers
    labels_are_string = False

    if(df_train[target_column].dtype == object):

        labels_are_string = True

        int_to_label = {}
        label_index = 0

        # Assign each label its designated integer
        for label in label_set:

            int_to_label[label_index] = label
            label_index += 1
                    
        # Create reversed map from label to integer
        label_to_int = {v: k for k, v in int_to_label.items()}
        logging.info("Mapping labels to integers: "+str(label_to_int))

        # Replace labels with their respective integers, use copies to avoid changing the original df
        df_train = df_train.copy()
        df_val = df_val.copy()
        df_test = df_test.copy()

        df_train[target_column] = [label_to_int[label] for label in df_train[target_column]]
        df_val[target_column] = [label_to_int[label] for label in df_val[target_column]]
        df_test[target_column] = [label_to_int[label] for label in df_test[target_column]]

    # Grab X,y for train, val and test
    train_texts = list(df_train.loc[:, answer_column])
    train_labels = encode_labels(df_train, label_column=target_column)
    valid_texts = list(df_val.loc[:, answer_column])
    valid_labels = encode_labels(df_val, label_column=target_column)
    test_texts = list(df_test.loc[:, answer_column])
    test_labels = encode_labels(df_test, label_column=target_column)

    # Load model and pass to device
    if from_pretrained:

        config = AutoConfig.from_pretrained('/models/'+SBERT_BASE_MODEL)
        config.sbert_path = '/models/'+SBERT_BASE_MODEL
        config.num_labels = len(df_train[target_column].unique())
        model = SbertForSequenceClassification(config).to(device)
        model.load_state_dict(torch.load(base_model))
    
    else:

        config = AutoConfig.from_pretrained(base_model)
        config.num_labels=len(label_set)
        config.sbert_path = base_model
        model = SbertForSequenceClassification(config).to(device)

    # Tokenize the dataset, truncate if longer than max_length, pad with 0's when less than `max_length`
    train_encodings = model.sbert.tokenize(train_texts)
    valid_encodings = model.sbert.tokenize(valid_texts)
    test_encodings = model.sbert.tokenize(test_texts)

    # Convert tokenized data into a torch Dataset
    train_dataset = Dataset(train_encodings, train_labels)
    valid_dataset = Dataset(valid_encodings, valid_labels)
    test_dataset = Dataset(test_encodings, test_labels)

    logging.info('Labels: ' + str(label_set))

    model.train()

    if save_model:

        training_args = TrainingArguments(
            output_dir=os.path.join(run_path, 'checkpoints'),
            num_train_epochs=num_epochs,             
            per_device_train_batch_size=batch_size,
            per_device_eval_batch_size=batch_size,  
            load_best_model_at_end=True,
            eval_strategy="epoch",
            logging_strategy="epoch",
            save_strategy="epoch",
            save_total_limit=5,
        )

    else:

        training_args = TrainingArguments(
            output_dir=os.path.join(run_path, 'checkpoints'),
            num_train_epochs=num_epochs,             
            per_device_train_batch_size=batch_size,
            per_device_eval_batch_size=batch_size,  
            evaluation_strategy="epoch",
            logging_strategy="epoch",
        )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=valid_dataset,
        compute_metrics=compute_metrics,
    )

    dict_val_loss = {}
    dict_test_preds = {}

    trainer.add_callback(WriteCsvCallback(csv_train=os.path.join(run_path, "train_stats.csv"), csv_eval=os.path.join(run_path, "eval_stats.csv"), dict_val_loss=dict_val_loss))
    trainer.add_callback(GetTestPredictionsCallback(dict_test_preds=dict_test_preds, save_path=os.path.join(run_path, "test_stats.csv"), trainer=trainer, test_data=test_dataset))
    trainer.add_callback(EarlyStoppingCallback(early_stopping_patience=3))
    trainer.train()

    # Determine epoch with lowest validation loss
    best_epoch = min(dict_val_loss, key=dict_val_loss.get)

    # For this epoch, return test predictions
    predictions = dict_test_preds[best_epoch]

    if labels_are_string:

        predictions = [int_to_label[pred] for pred in predictions]

    if save_model == True:

        # trainer.save_model(os.path.join(run_path, "best_model"))
        # model.save_pretrained(os.path.join(run_path, 'best_model'))
        torch.save(model, os.path.join(run_path, 'best_model'))

    # Delete model checkpoints to save space
    if os.path.exists(os.path.join(run_path, "checkpoints")):

        shutil.rmtree(os.path.join(run_path, "checkpoints"), ignore_errors=True)

    df_test['pred'] = predictions
    df_test.to_csv(os.path.join(run_path, 'preds.csv'))

    logging.info('Training duration:\t' + str(datetime.now() - start_time))

    del model

    return test_labels, predictions
