import os
import pandas as pd
import sys
from nltk.probability import FreqDist as FD


# def calc_xlmr(result_path='/results/exp_1_zero_shot'):
def calc_xlmr(result_path='/Users/mariebexte/Coding/Projects/cross-lingual/exp_3_lolo/combine_all_other'):

    avg_e = 0
    num_runs = 0
    list_e = []

    for prompt in os.listdir(result_path):

        if not '.' in prompt:

            for lang in os.listdir(os.path.join(result_path, prompt)):

                if len(lang) == 2:

                    result_file = os.path.join(result_path, prompt, lang, 'XLMR', 'eval_stats.csv')
                    df_results = pd.read_csv(result_file)

                    best = list(df_results['best_model_checkpoint'])[-1]
                    best_e = int(best[best.index('-')+1:])
                    
                    avg_e += best_e
                    list_e.append(best_e)
                    num_runs += 1

    print(avg_e)
    print(num_runs)
    print(avg_e/num_runs)

    print('+++ +++ +++')

    frequencies = FD(list_e)
    for freq, amount in frequencies.items():

        print(freq, amount)


def calc_sbert(result_path='/Users/mariebexte/Coding/Projects/cross-lingual/exp_3_lolo/combine_all_other'):

    avg_e = 0
    num_runs = 0
    list_e = []

    for prompt in os.listdir(result_path):

        if not '.' in prompt:

            for lang in os.listdir(os.path.join(result_path, prompt)):

                if len(lang) == 2:

                    best_e = 0
                    current_e = 0

                    for log in os.listdir(os.path.join(result_path, prompt, lang, 'SBERT')):

                        if log.startswith('log'):

                            df_log = pd.read_csv(os.path.join(result_path, prompt, lang, 'SBERT', log))
                            
                            for idx, line in df_log.iterrows():

                                line = line.iloc[0]
                                
                                if 'Evaluating the model on' in line:
                                    
                                    current_e = int(line[line.index('epoch ')+6:-1])
                                
                                if 'Save model to' in line:

                                    best_e = current_e
                        
                    avg_e += best_e
                    list_e.append(best_e)
                    num_runs += 1
            

    print(avg_e)
    print(num_runs)
    print(avg_e/num_runs)

    print('+++ +++ +++')

    frequencies = FD(list_e)
    for freq, amount in frequencies.items():

        print(freq, amount)


calc_xlmr()
calc_sbert()
