import pandas as pd
import os

from config import EPIRLS, ASAP_T, ASAP_M


def process_dataset(data_folder, dataset_name, languages, answer_column):

    results = {} 
    results_idx = 0

    frequencies = {}

    for prompt in os.listdir(data_folder):

        for language in languages:

            df_train = pd.read_csv(os.path.join(data_folder, prompt, language, 'train.csv'))
            df_val = pd.read_csv(os.path.join(data_folder, prompt, language, 'val.csv'))
            df_test = pd.read_csv(os.path.join(data_folder, prompt, language, 'test.csv'))

            df_full = pd.concat([df_train, df_val, df_test])

            avg_len = 0
            all_answers = list(df_full[answer_column])

            for answer in all_answers:

                avg_len += len(str(answer))
            
            avg_len = avg_len/len(df_full)

            all_answers_set = set(all_answers)

            ratio_unique = len(all_answers_set)/len(all_answers)

            results[results_idx] = {'prompt': prompt, 'lang': language, 'avg_len': avg_len, 'ratio_unique': ratio_unique}
            results_idx += 1

            # Exemplary processing of first language (score distributions are balanced)
            if language == languages[0]:

                score_dist = dict(df_full['score'].value_counts())
                total = sum(list(score_dist.values()))
                score_dist_percentage = {score: freq/total for score, freq in score_dist.items()}
                # print(prompt, language, dict(df_full['score'].value_counts()), score_dist_percentage)
                frequencies[prompt] = score_dist_percentage


    df_results = pd.DataFrame.from_dict(results, orient='index')

    with open('/data/' + dataset_name + '_answer_stats.tsv', 'w') as out_file:

        out_file.write('language\tavg_len\tratio_unique\tlength\n')

        for lang, df_lang in df_results.groupby('lang'):

            print(lang, df_lang['avg_len'].mean(), df_lang['ratio_unique'].mean(), len(df_lang))
            out_file.write(lang+'\t'+str(df_lang['avg_len'].mean())+'\t'+str(df_lang['ratio_unique'].mean())+'\t'+str(len(df_lang)))

    df_frequencies = pd.DataFrame(frequencies).T
    df_frequencies.to_csv('/data/' + dataset_name + '_scores.csv')


# ePIRLS
for dataset in [EPIRLS, ASAP_T, ASAP_M]:

    process_dataset(data_folder=dataset['dataset_path'], dataset_name=dataset['dataset_name'], languages=dataset['languages'], answer_column=dataset['answer_column'])