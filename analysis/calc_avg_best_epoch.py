import os
import pandas as pd
import sys
from nltk.probability import FreqDist as FD


## Analyse average best epoch across runs

# def calc_xlmr(result_path='/results/exp_1_zero_shot'):
def calc_xlmr(result_path='/results/FINAL_PAPER/exp_3_lolo_RUN1/combine_downsampled', model='XLMR'):

    avg_e = 0
    num_runs = 0
    list_e = []

    for prompt in os.listdir(result_path):

        if not '.' in prompt and os.path.isdir(os.path.join(result_path, prompt)):

            for lang in os.listdir(os.path.join(result_path, prompt)):

                if len(lang) == 2:

                    if os.path.exists(os.path.join(result_path, prompt, lang, model, 'eval_stats.csv')):

                        result_file = os.path.join(result_path, prompt, lang, model, 'eval_stats.csv')
                        df_results = pd.read_csv(result_file)

                        best_e = df_results['eval_loss'].idxmin() + 1
                        print(best_e)
                        
                        avg_e += best_e
                        list_e.append(best_e)
                        num_runs += 1
                    
                    else:

                        for fold in range(1, 8):

                            result_file = os.path.join(result_path, prompt, lang, model, 'fold_' + str(fold), 'eval_stats.csv')
                            df_results = pd.read_csv(result_file)

                            best_e = df_results['eval_loss'].idxmin() + 1
                            print(best_e)
                            
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


def calc_npcr(result_path='/results/FINAL_PAPER/exp_3_lolo_RUN1/combine_downsampled', model='NPCR_XLMR'):

    avg_e = 0
    num_runs = 0
    list_e = []

    for prompt in os.listdir(result_path):

        if not '.' in prompt and os.path.isdir(os.path.join(result_path, prompt)):

            for lang in os.listdir(os.path.join(result_path, prompt)):

                if len(lang) == 2:

                    if os.path.exists(os.path.join(result_path, prompt, lang, model, 'training_stats.csv')):

                        result_file = os.path.join(result_path, prompt, lang, model, 'training_stats.csv')
                        df_results = pd.read_csv(result_file)

                        best_e = df_results['val best epoch'].max() + 1
                        print(best_e)
                        
                        avg_e += best_e
                        list_e.append(best_e)
                        num_runs += 1
                    
                    else:

                        for fold in range(1, 8):

                            result_file = os.path.join(result_path, prompt, lang, model, 'fold_' + str(fold), 'training_stats.csv')
                            df_results = pd.read_csv(result_file)

                            best_e = df_results['val best epoch'].max() + 1
                            print(best_e)
                            
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


# def calc_xlmr(result_path='/results/FINAL_PAPER/exp_3_lolo_RUN1/combine_downsampled'):

#     avg_e = 0
#     num_runs = 0
#     list_e = []

#     for prompt in os.listdir(result_path):

#         if not '.' in prompt and os.path.isdir(os.path.join(result_path, prompt)):

#             for lang in os.listdir(os.path.join(result_path, prompt)):

#                 if len(lang) == 2:

#                     result_file = os.path.join(result_path, prompt, lang, 'XLMR', 'eval_stats.csv')
#                     df_results = pd.read_csv(result_file)

#                     print(df_results)
#                     best = list(df_results['best_model_checkpoint'])[-1]
#                     print(best)
#                     best_e = int(best[best.index('-')+1:])
                    
#                     avg_e += best_e
#                     list_e.append(best_e)
#                     num_runs += 1

#     print(avg_e)
#     print(num_runs)
#     print(avg_e/num_runs)

#     print('+++ +++ +++')

#     frequencies = FD(list_e)
#     for freq, amount in frequencies.items():

#         print(freq, amount)


def calc_sbert(result_path='/Users/mariebexte/Coding/Projects/cross-lingual/exp_3_lolo/combine_all_other', model='SBERT'):

    avg_e = 0
    num_runs = 0
    list_e = []

    for prompt in os.listdir(result_path):

        if not '.' in prompt:

            for lang in os.listdir(os.path.join(result_path, prompt)):

                if len(lang) == 2:

                    best_e = 0
                    current_e = 0

                    if not 'multilingual' in result_path:

                        for log in os.listdir(os.path.join(result_path, prompt, lang, model)):

                            if log.startswith('log'):

                                df_log = pd.read_csv(os.path.join(result_path, prompt, lang, model, log))
                                
                                for idx, line in df_log.iterrows():

                                    line = line.iloc[0]
                                    
                                    if 'Evaluating the model on' in line:
                                        
                                        current_e = int(line[line.index('epoch ')+6:-1])
                                    
                                    if 'Save model to' in line:

                                        best_e = current_e
                            
                        avg_e += best_e
                        list_e.append(best_e)
                        num_runs += 1
                    
                    else:

                        for fold in range(1, 8):

                            for log in os.listdir(os.path.join(result_path, prompt, lang, model, 'fold_' + str(fold))):

                                if log.startswith('log'):

                                    df_log = pd.read_csv(os.path.join(result_path, prompt, lang, model, 'fold_' + str(fold), log))
                                    
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


# calc_xlmr()
# calc_sbert()
calc_xlmr(result_path='/results/EXP_1_BS_16_20e_RUN1/ASAP_multilingual', model='XLMR')

# calc_xlmr(result_path='/results/exp_1_zero_shot_RUN1/ePIRLS', model='XLMR_SBERTcore')
# calc_npcr(result_path='/results/FINAL_PAPER/exp_1_zero_shot_RUN1/ASAP_translated', model='NPCR_XLMR')
# calc_sbert(result_path='/results/FINAL_PAPER/exp_1_zero_shot_RUN1/ePIRLS', model='SBERT')

# calc_xlmr(result_path='/results/FINAL_PAPER/exp_3_lolo_RUN1/combine_downsampled/ASAP_multilingual', model='XLMR_SBERTcore')
# calc_npcr(result_path='/results/FINAL_PAPER/exp_3_lolo_RUN1/combine_downsampled/ASAP_multilingual', model='NPCR_SBERT')
# calc_sbert(result_path='/results/FINAL_PAPER/exp_3_lolo_RUN1/combine_downsampled/ePIRLS', model='SBERT')
