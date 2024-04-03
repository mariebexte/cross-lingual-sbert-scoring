import pandas as pd
import sys
import copy
import os
import math

seed = 59348605

## Make a folder for each prompt, containing train/val/test csvs
output_folder = '/data/ASAP/split'
target_column = 'Score1'

if not os.path.exists(output_folder):
    os.mkdir(output_folder)


## Split data to a prompt into train/val/test
def split_data(prompt, df_prompt):
    
    print('Processing', prompt)

    ## Determine distribution of values
    label_dist = dict(df_prompt[target_column].value_counts())
    total = sum(label_dist.values())
    label_dist_frac = {label: count/total for label, count in label_dist.items()}

    # Initialize dfs to hold splits
    df_lang_train = pd.DataFrame()
    df_lang_val = pd.DataFrame()
    df_lang_test = pd.DataFrame()

    if len(df_prompt)< 800:    
        print('Not enough anwers for prompt', prompt)
        sys.exit(0)

    # Sample desired amounts label-wise: Num val/test = 100, Num Train = 600
    for label, df_label in df_prompt.groupby(target_column):

        # Sample test
        df_lang_test_sample = df_label.sample(round(label_dist_frac[label]*100), random_state=seed)
        df_label = df_label.drop(df_lang_test_sample.index)
        df_lang_test = pd.concat([df_lang_test, df_lang_test_sample])

        # Sample val
        df_lang_val_sample = df_label.sample(round(label_dist_frac[label]*100), random_state=seed)
        df_label = df_label.drop(df_lang_val_sample.index)
        df_lang_val = pd.concat([df_lang_val, df_lang_val_sample])

        df_lang_train_sample = df_label.sample(round(label_dist_frac[label]*600), random_state=seed)
        df_lang_train = pd.concat([df_lang_train, df_lang_train_sample])


    print(len(df_lang_train), len(df_lang_val), len(df_lang_test))
    print(df_lang_train[target_column].value_counts(), df_lang_val[target_column].value_counts(), df_lang_test[target_column].value_counts())
        
    target_folder = os.path.join(output_folder, prompt, 'en')
    if not os.path.exists(target_folder):
        os.makedirs(target_folder)

    print(target_folder)

    df_lang_train.to_csv(os.path.join(target_folder, 'train.csv'))
    df_lang_val.to_csv(os.path.join(target_folder, 'val.csv'))
    df_lang_test.to_csv(os.path.join(target_folder, 'test.csv'))




### Read data, discard what we do not need, process into splits prompt-wise
data_path = '/data/ASAP'

for prompt in os.listdir(data_path):

    if 'allAnswers' in prompt:

        df_prompt = pd.read_csv(os.path.join(data_path, prompt), sep='\t')
        prompt = prompt[22:prompt.index('.')]

        # print(prompt, df_prompt)
        split_data(prompt, df_prompt)
