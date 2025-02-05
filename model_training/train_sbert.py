import os
from sentence_transformers import SentenceTransformer, InputExample, losses, evaluation, models
from sentence_transformers.evaluation import SimilarityFunction
import pandas as pd
from torch.utils.data import DataLoader
import torch
import shutil
import sys
import shutil
import logging
from model_training.utils import encode_labels, eval_sbert, get_device
from datetime import datetime
from tqdm import tqdm
import random


random_state = 3456478

# For larger amounts of training data: Do not create all possible pairs, but limit to a fixed number per epoch (if possible, have different pairs across different epochs)
def train_sbert(run_path, df_train, df_val, df_test, answer_column, target_column, id_column, base_model="all-MiniLM-L6-v2", batch_size=8, num_epochs=20, num_pairs_per_example=None, do_warmup=False, save_model=False, num_val_pairs=None):

    # Clear logger from previous runs
    log = logging.getLogger()
    handlers = log.handlers[:]

    for handler in handlers:
    
        log.removeHandler(handler)
        handler.close()

    device = get_device()

    print('**** Running SBERT on:', device)

    if not os.path.exists(run_path):

        os.makedirs(run_path)
    
    logging.basicConfig(filename=os.path.join(run_path, datetime.now().strftime('logs_%H_%M_%d_%m_%Y.log')), filemode='w', level=logging.DEBUG)

    logging.info('base_model:\t' + base_model)
    logging.info('number of epochs:\t' + str(num_epochs))
    logging.info('batch size:\t' + str(batch_size))
    logging.info('warmup:\t' + str(do_warmup))

    logging.info('num train:\t' + str(len(df_train)))
    logging.info('num val:\t' + str(len(df_val)))
    logging.info('num test:\t' + str(len(df_test)))
    logging.info('num pairs per training example:\t' + str(num_pairs_per_example))
    logging.info('num validation pairs:\t' + str(num_val_pairs))

    start_time = datetime.now()

    num_batches_per_round = None

    if num_pairs_per_example is not None:

        num_samples = len(df_train) * num_pairs_per_example
        num_batches_per_round = int(num_samples/batch_size)
        logging.info("LIMITING SBERT TRAINING PAIRS: "+str(num_pairs_per_example)+" pairs per sample!")

    try:

        model = SentenceTransformer(base_model, device=device)

    except:

        # Above sometimes crashes, but this gives equivalent result
        transformer = models.Transformer(base_model)
        pooling = models.Pooling(transformer.get_word_embedding_dimension(), pooling_mode="mean")
        model = SentenceTransformer(modules=[transformer, pooling], device=device)

    # Where to store finetuned model
    model_path = os.path.join(run_path, "finetuned_model")

    # Define list of training pairs: Create only as many as needed
    if num_epochs * num_pairs_per_example < len(df_train):

        seeds = random.Random(random_state).sample(range(1, 100000), len(df_train))

        train_examples = []

        for idx_1, example_1 in tqdm(df_train.iterrows(), total=len(df_train)):

            df_subsample = df_train.sample(num_epochs * num_pairs_per_example, random_state=seeds[idx_1])

            for _, example_2 in df_subsample.iterrows():

                if not example_1[id_column] == example_2[id_column]:

                    label = 0

                    if example_1[target_column] == example_2[target_column]:

                        label = 1

                    train_examples.append(InputExample(texts=[example_1[answer_column], example_2[answer_column]], label=label*1.0))
    
    # Define list of training pairs: Create as many as possible
    else:

        train_examples = []

        for _, example_1 in tqdm(df_train.iterrows(), total=len(df_train)):

            for _, example_2 in df_train.iterrows():

                if not example_1[id_column] == example_2[id_column]:

                    label = 0

                    if example_1[target_column] == example_2[target_column]:

                        label = 1

                    train_examples.append(InputExample(texts=[example_1[answer_column], example_2[answer_column]], label=label*1.0))

    # Define validation pairs: Create as many as possible
    val_example_dict = {}
    val_example_index = 0

    for _, example_1 in tqdm(df_val.iterrows(), total=len(df_val)):

        for _, example_2 in df_train.iterrows():

            if not example_1[id_column] == example_2[id_column]:

                label = 0

                if example_1[target_column] == example_2[target_column]:

                    label = 1

                val_example_dict[val_example_index] = {"text_1": example_1[answer_column], "text_2": example_2[answer_column], "sim_label": label}
                val_example_index += 1

    val_examples = pd.DataFrame.from_dict(val_example_dict, "index")

    if num_val_pairs is not None:

        val_examples = val_examples.sample(num_val_pairs, random_state=random_state)

    # Define train dataset, dataloader, train loss
    train_dataloader = DataLoader(train_examples, shuffle=True, batch_size=batch_size)
    train_loss = losses.OnlineContrastiveLoss(model)

    # Define evaluator
    evaluator = evaluation.EmbeddingSimilarityEvaluator(val_examples["text_1"].tolist(), val_examples["text_2"].tolist(), val_examples["sim_label"].tolist())

    num_warm_steps = 0

    if do_warmup == True:

        if num_pairs_per_example is not None:

            steps_per_epoch = num_batches_per_round

        else:
            steps_per_epoch = len(train_examples)/batch_size

        total_num_steps = steps_per_epoch * num_epochs
        num_warm_steps = round(0.1*total_num_steps)

    # Tune the model
    if num_pairs_per_example is not None:

        model.fit(train_objectives=[(train_dataloader, train_loss)], epochs=num_epochs, warmup_steps=num_warm_steps, evaluator=evaluator, output_path=model_path, save_best_model=True, show_progress_bar=True, steps_per_epoch=num_batches_per_round)
    
    else:

        model.fit(train_objectives=[(train_dataloader, train_loss)], epochs=num_epochs, warmup_steps=num_warm_steps, evaluator=evaluator, output_path=model_path, save_best_model=True, show_progress_bar=True)

    logging.info("SBERT number of epochs: "+str(num_epochs))
    logging.info("SBERT batch size: "+str(batch_size))
    logging.info("SBERT warmup steps: "+str(num_warm_steps))
    logging.info("SBERT evaluator: "+str(evaluator.__class__)+" Batch size: "+str(evaluator.batch_size)+" Main similarity:"+str(evaluator.main_similarity))
    logging.info("SBERT loss: "+str(train_loss.__class__))

    # Evaluate best model: Can only do this if training was sucessful, otherwise keep pretrained
    if os.path.exists(os.path.join(model_path, "pytorch_model.bin")):

        model = SentenceTransformer(model_path)

    else:

        try:

            model = SentenceTransformer(base_model)
        
        except:

            # Above sometimes crashes, but this gives equivalent result
            transformer = models.Transformer(base_model)
            pooling = models.Pooling(transformer.get_word_embedding_dimension(), pooling_mode="mean")
            model = SentenceTransformer(modules=[transformer, pooling], device=device)

    # Eval testing data: Get sentence embeddings for all testing and reference answers
    df_test['embedding'] = df_test[answer_column].apply(model.encode)

    df_ref = pd.concat([df_val, df_train])
    df_ref['embedding'] = df_ref[answer_column].apply(model.encode)

    # Copy training statistic into run folder
    if os.path.exists(model_path):

        shutil.copyfile(os.path.join(model_path, "eval", "similarity_evaluation_results.csv"), os.path.join(run_path, "eval_training.csv"))

    # Delete model to save space
    if os.path.exists(model_path) and save_model==False:

        shutil.rmtree(model_path)

    gold, max_pred, avg_pred = eval_sbert(run_path=run_path, df_test=df_test, df_ref=df_ref, id_column=id_column, answer_column=answer_column, target_column=target_column)

    logging.info('Training duration:\t' + str(datetime.now() - start_time))

    return gold, max_pred, avg_pred
    
