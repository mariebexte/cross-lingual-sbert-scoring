import os
import torch
import pandas as pd
import numpy as np

from sklearn.metrics import classification_report, accuracy_score, cohen_kappa_score, f1_score
from transformers import TrainerCallback, Trainer

from scipy import spatial

from config import ANSWER_LENGTH


def read_data(path, answer_column, target_column):

    df = pd.read_csv(path)
    df = df.fillna('')
    df[answer_column] = df[answer_column].astype(str)
    df[target_column] = df[target_column].astype(int)

    # Shuffle dataframe
    df = df.sample(frac = 1)

    return df


## Strip labels from dataframe and return list of integers
def encode_labels(df, label_column):

    labels = list(df.loc[:, label_column])
    labels = [int(label) for label in labels]

    return labels
    

## Which metrics to compute on evaluation
def compute_metrics(pred):

    labels = pred.label_ids
    preds = pred.predictions.argmax(-1)
    acc = accuracy_score(labels, preds)
    f1 = f1_score(labels, preds, average="weighted")
    kappa = cohen_kappa_score(labels, preds) 
    qwk = cohen_kappa_score(labels, preds, weights='quadratic') 

    return {
      'acc': acc,
      'weighted f1': f1,
      'kappa': kappa,
      'qwk': qwk,
    }


def average_qwk(df):

    high = 0.999

    try:

        df['qwk_smooth'] = df['qwk'].apply(lambda x: x if x < high else high)

    except:

        df['qwk_smooth'] = df['qwk_within_val'].apply(lambda x: x if x < high else high)

    # Arctanh == FISHER
    df_preds_fisher = np.arctanh(df)
    # print(df_preds_fisher)
    test_scores_mean_fisher = np.nanmean(df_preds_fisher, axis=0)
    # Tanh == FISHERINV
    test_scores_mean = np.tanh(test_scores_mean_fisher)

    return test_scores_mean


def get_device():

    device = 'cpu'
    if torch.cuda.is_available():
        device = 'cuda'
    
    return device


## Write classification statistics to specified output directory, if desired provide suffix for filename
def write_classification_statistics(filepath, y_true, y_pred, suffix=''):

    qwk = cohen_kappa_score(y1=y_true, y2=y_pred, weights='quadratic')
    kappa = cohen_kappa_score(y1=y_true, y2=y_pred)

    with open(os.path.join(filepath, 'test_performance' + suffix + '.txt'), 'w') as eval_stats:

        eval_stats.write(classification_report(y_true=y_true, y_pred=y_pred)+"\n\n")
        true_series = pd.Series(y_true, name='Actual')
        pred_series = pd.Series(y_pred, name='Predicted')
        eval_stats.write(str(pd.crosstab(true_series, pred_series))+"\n\n")
        eval_stats.write("Kappa:\t"+str(kappa)+"\n")
        eval_stats.write("QWK:\t"+str(qwk))


## Evaluate provided pretrained BERT model on a dataframe with test data
def eval_bert(model, tokenizer, df_test, answer_column, target_column):
    
    model.eval()

    test_texts = list(df_test.loc[:, answer_column])
    test_labels = encode_labels(df_test, label_column=target_column)
    test_encodings = tokenizer(test_texts, truncation=True, padding=True, max_length=ANSWER_LENGTH)
    test_dataset = Dataset(test_encodings, test_labels)

    trainer = Trainer(model=model)

    preds = trainer.predict(test_dataset) 
    predictions = preds.predictions.argmax(axis=1)

    return test_labels, predictions


## Evaluate provided pretrained BERT model on a dataframe with test data
def eval_sbert_classification(model, df_test, answer_column, target_column):
    
    model.eval()

    test_texts = list(df_test.loc[:, answer_column])
    test_labels = encode_labels(df_test, label_column=target_column)
    test_encodings = model.sbert.tokenize(test_texts)
    test_dataset = Dataset(test_encodings, test_labels)

    trainer = Trainer(model=model)

    preds = trainer.predict(test_dataset) 
    predictions = preds.predictions.argmax(axis=1)

    return test_labels, predictions


# _2 is ref!
def cross_dataframes(df, df_ref):

    return pd.merge(left=df, right=df_ref, how='cross', suffixes=('_1', '_2'))


def get_preds_from_pairs(df, id_column, pred_column, ref_label_column, true_label_column, threshold=0.9):

    answer_ids = []
    pred_labels = []
    pred_labels_max = []
    pred_labels_hybrid = []
    true_labels = []

    # For each test instance
    for answer, df_answer in df.groupby(id_column):

        true_label = list(df_answer[true_label_column].unique())
        if len(true_label) > 1:
            print('True label not unique!', true_label)
            sys.exit(0)

        else:
            true_label = true_label[0]

        score_probs = {}

        for label, df_label in df_answer.groupby(ref_label_column):

            score_probs[label] = df_label[pred_column].mean()
        
        pred_label = max(score_probs, key=score_probs.get)
    
        # Find most similar answer and its label
        max_idx = df_answer[pred_column].idxmax()
        pred_label_max = df_answer.loc[max_idx][ref_label_column]

        answer_ids.append(answer)
        pred_labels.append(pred_label)
        pred_labels_max.append(pred_label_max)
        true_labels.append(true_label)

        if df_answer.loc[max_idx][pred_column] >= threshold:

            pred_labels_hybrid.append(pred_label_max)
        
        else:

            pred_labels_hybrid.append(pred_label)

    return answer_ids, true_labels, pred_labels, pred_labels_max, pred_labels_hybrid


def calculate_sim(row):

    return (1 - spatial.distance.cosine(row['embedding_1'], row['embedding_2']))


def eval_sbert(run_path, df_test, df_ref, id_column, answer_column, target_column):

    df_inference = cross_dataframes(df=df_test, df_ref=df_ref)

    df_inference['sim'] = df_inference.apply(calculate_sim, axis=1)

    test_answers, test_true_scores, test_predictions, test_predictions_max, test_predictions_hybrid = get_preds_from_pairs(df=df_inference, id_column=id_column+'_1', pred_column='sim', ref_label_column=target_column+'_2', true_label_column=target_column+'_1')

    df_test_aggregated = pd.DataFrame({id_column: test_answers, 'pred': test_predictions, target_column: test_true_scores, 'pred_max': test_predictions_max, 'pred_hybrid': test_predictions_hybrid})
    df_test_aggregated.to_csv(os.path.join(run_path, 'preds.csv'))

    return test_true_scores, test_predictions_max, test_predictions, test_predictions_hybrid


class Dataset(torch.utils.data.Dataset):

    def __init__(self, encodings, labels):

        self.encodings = encodings
        self.labels = labels

    def __getitem__(self, idx):

        item = {k: v[idx] for k, v in self.encodings.items()}
        # item = {k: v[idx].clone().detach() for k, v in self.encodings.items()}
        item["labels"] = torch.tensor(self.labels[idx])

        return item

    def __len__(self):

        return len(self.labels)


## Callback to monitor performance
class GetTestPredictionsCallback(TrainerCallback):

    def __init__(self, *args, dict_test_preds, save_path, trainer, test_data, **kwargs):

        super().__init__(*args, **kwargs)

        self.dict_test_preds = dict_test_preds
        self.save_path = save_path
        self.trainer = trainer
        self.test_data=test_data
        self.df_test_stats = pd.DataFrame()

    def on_log(self, args, state, control, logs=None, **kwargs):

        pred = self.trainer.predict(self.test_data)
        predictions = pred.predictions.argmax(axis=1)
        self.dict_test_preds[logs['epoch']] = predictions
        self.df_test_stats = pd.concat([self.df_test_stats, pd.DataFrame(pred.metrics, index=[int(logs['epoch'])])])

    def on_train_end(self, args, state, control, **kwargs):

        self.df_test_stats.to_csv(self.save_path, index_label='epoch')


## Callback to log loss of training and evaluation to file
class WriteCsvCallback(TrainerCallback):

    def __init__(self, *args, csv_train, csv_eval, dict_val_loss, **kwargs):

        super().__init__(*args, **kwargs)

        self.csv_train_path = csv_train
        self.csv_eval_path = csv_eval
        self.df_eval = pd.DataFrame()
        self.df_train_eval = pd.DataFrame()
        self.dict_val_loss = dict_val_loss

    def on_log(self, args, state, control, logs=None, **kwargs):

        df_log = pd.DataFrame([logs])

        # Has info about performance on training data
        if "loss" in logs:

            self.df_train_eval = pd.concat([self.df_train_eval, df_log])
        
        # Has info about performance on validation data
        else:

            best_model = state.best_model_checkpoint
            df_log["best_model_checkpoint"] = best_model
            self.df_eval = pd.concat([self.df_eval, df_log])

            if 'eval_loss' in logs:

                self.dict_val_loss[logs['epoch']] = logs['eval_loss']

    def on_train_end(self, args, state, control, **kwargs):

        self.df_eval.to_csv(self.csv_eval_path)
        self.df_train_eval.to_csv(self.csv_train_path)
