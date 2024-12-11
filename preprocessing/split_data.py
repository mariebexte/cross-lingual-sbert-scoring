import pandas as pd
import sys
import copy
import os
import math

seed = 59348605

## Split ePIRLS data

## Make a folder for each prompt, containing train/val/test csvs
output_folder = '/data'

if not os.path.exists(output_folder):
    
    os.mkdir(output_folder)


## Split data to a prompt into train/val/test
def split_data(prompt, df_prompt):
    
    print('Processing', prompt)

    languages = df_prompt['Language'].unique()

    ## Dict to hold data as {label: count}
    ## Count is max achievable number of answers with same label distribution across all languages
    max_answer_counts = {}

    for label, df_label in df_prompt.groupby('score'):

        fewest = df_label['Language'].value_counts().min() 
        max_answer_counts[label] = fewest

    ## Determine distribution of values
    total = sum(max_answer_counts.values())
    max_answer_counts_normalized = {label: value/total for label, value in max_answer_counts.items()}
    print(max_answer_counts_normalized)

    ## Sample instances according to distribution
    for language, df_lang in df_prompt.groupby('Language'):

        # Initialize dfs to hold splits
        df_lang_train = pd.DataFrame()
        df_lang_val = pd.DataFrame()
        df_lang_test = pd.DataFrame()

        # Sample desired amounts label-wise: Num val/test = 100, Num Train = 600
        for label, df_label in df_lang.groupby('score'):

            # Sample test
            df_lang_test_sample = df_label.sample(round(max_answer_counts_normalized[label]*100), random_state=seed)
            df_label = df_label.drop(df_lang_test_sample.index)
            df_lang_test = pd.concat([df_lang_test, df_lang_test_sample])

            # Sample val
            df_lang_val_sample = df_label.sample(round(max_answer_counts_normalized[label]*100), random_state=seed)
            df_label = df_label.drop(df_lang_val_sample.index)
            df_lang_val = pd.concat([df_lang_val, df_lang_val_sample])

            # If it has enough answers this is a prompt to be used in experiments:
            # Sample train
            if total >= 800:

                df_lang_train_sample = df_label.sample(round(max_answer_counts_normalized[label]*600), random_state=seed)
                df_lang_train = pd.concat([df_lang_train, df_lang_train_sample])
            
            # Other prompts are used for hyperparameter search: grab as many as possible, while keeping equal amount across languages
            else:

                df_lang_train_sample = df_label.sample(math.floor(max_answer_counts_normalized[label]*(total-200)), random_state=seed)
                df_lang_train = pd.concat([df_lang_train, df_lang_train_sample])


        print(language, len(df_lang_train), len(df_lang_val), len(df_lang_test))
        
        # Assign to split according to answer count
        if total >= 800:

            target_folder = 'exp'
        
        else:

            target_folder = 'dev'

        
        target_folder = os.path.join(output_folder, target_folder, prompt, language)

        if not os.path.exists(target_folder):

            os.makedirs(target_folder)

        df_lang_train.to_csv(os.path.join(target_folder, 'train.csv'))
        df_lang_val.to_csv(os.path.join(target_folder, 'val.csv'))
        df_lang_test.to_csv(os.path.join(target_folder, 'test.csv'))


### Read data, discard what we do not need, process into splits prompt-wise
data_path = '/data/data_newest.CSV'
df = pd.read_csv(data_path, sep=';')
df['id'] = df.index

# Remove NaN answers
df_clean = df[df['score'] != 9.0]
# Remove languages where answer count is too low
df_clean = df_clean[~ df_clean['Language'].isin(['az', 'fr', 'nn'])]

for prompt, df_prompt in df_clean.groupby('Variable'):

    split_data(prompt, df_prompt)
