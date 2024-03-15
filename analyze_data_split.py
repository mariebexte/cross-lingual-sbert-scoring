import pandas as pd
import os


data_folder = '/data/exp'

results = {} 
results_idx = 0

frequencies = {}

for prompt in os.listdir(data_folder):

    for language in os.listdir(os.path.join(data_folder, prompt)):

        df_train = pd.read_csv(os.path.join(data_folder, prompt, language, 'train.csv'))
        df_val = pd.read_csv(os.path.join(data_folder, prompt, language, 'val.csv'))
        df_test = pd.read_csv(os.path.join(data_folder, prompt, language, 'test.csv'))

        df_full = pd.concat([df_train, df_val, df_test])

        avg_len = 0
        all_answers = list(df_full['Value'])

        for answer in all_answers:

            avg_len += len(str(answer))
        
        avg_len = avg_len/len(df_full)

        all_answers_set = set(all_answers)

        ratio_unique = len(all_answers_set)/len(all_answers)

        results[results_idx] = {'prompt': prompt, 'lang': language, 'avg_len': avg_len, 'ratio_unique': ratio_unique}
        results_idx += 1

        if language == 'ar':

            score_dist = dict(df_full['score'].value_counts())
            total = sum(list(score_dist.values()))
            score_dist_percentage = {score: freq/total for score, freq in score_dist.items()}
            # print(prompt, language, dict(df_full['score'].value_counts()), score_dist_percentage)
            frequencies[prompt] = score_dist_percentage


df_results = pd.DataFrame.from_dict(results, orient='index')
print(df_results)
for lang, df_lang in df_results.groupby('lang'):

    print(lang, df_lang['avg_len'].mean(), df_lang['ratio_unique'].mean(), len(df_lang))

df_frequencies = pd.DataFrame(frequencies).T
# print(df_frequencies)
df_frequencies.to_csv('/data/label_distributions_exp.csv')