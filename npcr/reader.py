import re
import sys
import nltk
import npcr.utils as utils
import logging

import numpy as np

from transformers import XLMRobertaTokenizer, BertTokenizer
from sentence_transformers import SentenceTransformer


url_replacer = '<url>'
logger = utils.get_logger("Loading data...")


def tokenize(string):
    tokens = nltk.word_tokenize(string)
    for index, token in enumerate(tokens):
        if token == '@' and (index+1) < len(tokens):
            tokens[index+1] = '@' + re.sub('[0-9]+.*', '', tokens[index+1])
            tokens.pop(index)
    return tokens


def offcenter_labels(labels):
    min_label = -1
    max_label = 1
    offcentered_labels = (labels - min_label) / (max_label - min_label)
    return offcentered_labels


def get_model_friendly_scores_adversarial(scores_array, min_label=None, max_label=None, prompts_array=None):

    if (min_label is None) and (max_label is None) and (prompts_array is None):
        logging.info('Cannot get model friendly scores: Neither min score, nor max score or prompts array was set!')

    if (min_label is not None) and (max_label is not None):
        # logging.info('Using min and max score to rescale labels! ' + str(min_label) + ' ' + str(max_label))
        scores_array = (scores_array - min_label) / (max_label - min_label)
        return scores_array

    else:
        # logging.info('Using prompt information to rescale labels! ' + str(prompts_array))
        scaled_scores = []
        for i in range(len(scores_array)):
            current_score = scores_array[i]
            current_prompt = prompts_array[i]
            min_label = utils.SCORES[current_prompt][0]
            max_label = utils.SCORES[current_prompt][1]
            scaled_scores.append((current_score - min_label) / (max_label - min_label))
        return np.array(scaled_scores)


def replace_url(text):
    replaced_text = re.sub('(http[s]?://)?((www)\.)?([a-zA-Z0-9]+)\.{1}((com)(\.(cn))?|(org))', url_replacer, text)
    return replaced_text


def text_tokenizer(text):
   
    text = replace_url(text)
    text = text.replace(u'"', u'')
    if "..." in text:
        text = re.sub(r'\.{3,}(\s+\.{3,})*', '...', text)
        # print text
    if "??" in text:
        text = re.sub(r'\?{2,}(\s+\?{2,})*', '?', text)
        # print text
    if "!!" in text:
        text = re.sub(r'\!{2,}(\s+\!{2,})*', '!', text)
        # print text

    tokens = tokenize(text)
    punctuation = '.!,;:?"\'、，；'
    text = " ".join(tokens)

    text_nopun = re.sub(r'[{}]+'.format(punctuation), '', text)
    sent_tokens = text_nopun

    return sent_tokens


def get_tokenizer_output(row, tokenizer, text_col, max_length):

    sent_tokens = text_tokenizer(row[text_col])
    tokenizer_output = tokenizer(sent_tokens, padding='max_length', truncation=True, max_length=max_length)

    return tokenizer_output['input_ids'], tokenizer_output['attention_mask']


def read_dataset_adversarial(df, base_model, col_score, col_answer, max_length=1024):

    if base_model == 'paraphrase-multilingual-MiniLM-L12-v2':
        model = SentenceTransformer(base_model)
        tokenizer = model.tokenizer

    elif base_model =='xlm-roberta-base':
        tokenizer = XLMRobertaTokenizer.from_pretrained(base_model)
    
    elif base_model == 'bert-base-uncased':
        tokenizer = BertTokenizer.from_pretrained(base_model, sep_token='[SEP]')

    else:
        print('Unknown model name!', model_name)
        sys.exit(0)

    if df is not None:
        df[col_answer] = df[col_answer].apply(lambda row: row.strip())
        df[col_score] = df[col_score].apply(lambda row: float(row))
        df['input_ids'], df['attention_mask'] = zip(*df.apply(get_tokenizer_output, tokenizer=tokenizer, text_col=col_answer, max_length=max_length, axis='columns'))
    
    return df


def get_data(df_train, df_val, df_test, max_length, base_model, col_score, col_answer):

    df_train = read_dataset_adversarial(df_train, max_length=max_length, base_model=base_model, col_score=col_score, col_answer=col_answer)
    df_val = read_dataset_adversarial(df_val, max_length=max_length, base_model=base_model, col_score=col_score, col_answer=col_answer)
    df_test = read_dataset_adversarial(df_test, max_length=max_length, base_model=base_model, col_score=col_score, col_answer=col_answer)

    return df_train, df_val, df_test