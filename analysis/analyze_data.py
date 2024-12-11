import pandas as pd
import sys
import copy
import os


## Explore ePIRLS data

output_folder = '/results/data-analysis'

if not os.path.exists(output_folder):

    os.mkdir(output_folder)


## Write statistics of input dataframe to file
def get_stats(df_clean, out_name, remove_duplicates=True):

    df_overview = pd.DataFrame()

    for lang, df_lang in df_clean.groupby('Language'):

        # Store for which prompt there are how many answers
        lang_dict = {}

        for prompt, df_prompt in df_lang.groupby('Variable'):

            # Reduce to unique answers, if an answer occurs with different labels: Keep both
            if remove_duplicates:

                df_prompt = df_prompt.drop_duplicates(subset=['score', 'Value'], keep='first')

            lang_dict[prompt] = len(df_prompt)

        df_lang_dict = pd.DataFrame.from_dict(lang_dict, orient='index').T
        df_lang_dict.index = [lang]
        df_overview = pd.concat([df_overview, df_lang_dict])


    ## For each prompt: For each label: Determine lowest answer count across languages
    # Sum these counts up to the highest possible number of answers with constant label distribution
    languages = df_overview.index
    max_answer_counts = {}

    for prompt, df_prompt in df_clean.groupby('Variable'):

        answer_count = 0

        for label, df_label in df_prompt.groupby('score'):

            if remove_duplicates:
                
                df_label = df_label.drop_duplicates(subset=['Language', 'Value'], keep="first")

            fewest = df_label['Language'].value_counts().min() 
            answer_count = answer_count + fewest
            # print(fewest, df_label['Language'].value_counts())

        # print(prompt, answer_count)
        max_answer_counts[prompt] = answer_count
        # print(prompt, answer_count)

    df_max = pd.DataFrame.from_dict(max_answer_counts, orient='index').T
    df_max.index = ['MAX_TOTAL']
    df_overview = pd.concat([df_overview, df_max])
    df_overview.to_csv(out_name)


## Read raw data
data_path = '/data/data_newest.CSV'
df = pd.read_csv(data_path, sep=';')
# print(df.columns)

## Calculate stats of raw data
get_stats(df_clean=df, out_name=os.path.join(output_folder, 'overview.csv'), remove_duplicates=False)

## Remove NaN answers & calculate stats
df_clean = df[df['score'] != 9.0]
get_stats(df_clean=df_clean, out_name=os.path.join(output_folder, 'overview_noNanAnswers.csv'), remove_duplicates=False)

## Remove languages that haw too few answers & calculate stats
df_clean = df_clean[~ df_clean['Language'].isin(['az', 'fr', 'nn'])]
get_stats(df_clean=df_clean, out_name=os.path.join(output_folder, 'overview_noNanAnswers_noLowCountLanguages.csv'), remove_duplicates=False)

## Calculate stats without duplicate answers
get_stats(df_clean=df_clean, out_name=os.path.join(output_folder, 'overview_noNanAnswers_noLowCountLanguages_noDuplicates.csv'))
