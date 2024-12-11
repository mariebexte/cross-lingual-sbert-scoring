import pandas as pd
import sys
import copy
import os


## Analyze answer overlap

def eval_overlap(df):

    for prompt, df_prompt in df.groupby('Variable'):

        print(prompt)

        lang_dict = {}

        for lang, df_lang in df_prompt.groupby('Language'):

            lang_dict[lang] = set(list(df_lang['Value']))
        
        for lang1, lang1_answers in lang_dict.items():

            for lang2, lang2_answers in lang_dict.items():

                if not lang1 == lang2:

                    overlap = lang1_answers.intersection(lang2_answers)

                    if len(overlap) > 10:

                        print(prompt, lang1, lang2, len(overlap), overlap)


print('Starting evaluation')

## Read raw data
data_path = '/data/data_newest.CSV'
df = pd.read_csv(data_path, sep=';')
# print(df.columns)

print('Read data')

## Remove NaN answers
df_clean = df[df['score'] != 9.0]

## Remove languages that haw too few answers & calculate stats
df_clean = df_clean[~ df_clean['Language'].isin(['az', 'fr', 'nn'])]

eval_overlap(df_clean)