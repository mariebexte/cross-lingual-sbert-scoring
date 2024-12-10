import os
import pandas as pd


data_path = '/data/exp'
score_ranges = {}

for prompt in os.listdir(data_path):

    if os.path.isdir(os.path.join(data_path, prompt)):

        dfs = []

        for language in ['ar', 'da', 'en', 'he', 'it', 'ka', 'nb', 'pt', 'sl', 'sv', 'zh']:

            for file in ['train.csv', 'val.csv', 'test.csv']:

                df_temp = pd.read_csv(os.path.join(data_path, prompt, language, file))
                dfs.append(df_temp)
        
        df_overall = pd.concat(dfs)
        scores = df_overall['score'].unique()
        min_score = scores.min()
        max_score = scores.max()
        print(prompt, min_score, max_score)
        print(df_overall['score'].value_counts())

        score_ranges[prompt] = [int(min_score), int(max_score)]

print(score_ranges)




