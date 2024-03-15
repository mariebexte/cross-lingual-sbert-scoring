import os
import pandas as pd
from sklearn.metrics import cohen_kappa_score
import sys
from utils import average_qwk


def eval_condition(results_path='/Users/mariebexte/Coding/Projects/cross-lingual/exp_3_lolo/combine_all_other'):

    results = {}
    results_idx = 0

    for prompt in os.listdir(results_path):
    
        if not '.' in prompt:

            for language in os.listdir(os.path.join(results_path, prompt)):

                if len(language) == 2:

                    for model in [('SBERT', '_avg'), ('SBERT', '_max')]:
                    # for model in [('XLMR', ''), ('SBERT', '_avg'), ('SBERT', '_max')]:

                        try:
                            preds = pd.read_csv(os.path.join(results_path, prompt, language, model[0], 'preds.csv'))
                            qwk = cohen_kappa_score(list(preds['score']), list(preds['pred'+model[1]]), weights='quadratic')

                            results[results_idx] = {
                                'prompt': prompt,
                                'test_lang': language,
                                'model': model[0] + model[1],
                                'qwk': qwk,
                            }

                            results_idx += 1

                        except:
                            print(prompt, model, language)
                            print('HERE')
                            # sys.exit(0)
                            qwk = -1


    df_results = pd.DataFrame.from_dict(results, orient='index')
    df_results.to_csv(os.path.join(results_path, 'overall.csv'))

    print(df_results)

    for model, df_model in df_results.groupby('model'):

        aggregated_dict = {}
        aggregated_dict_idx = 0

        for lang, df_lang in df_model.groupby('test_lang'):

            aggregated_dict[aggregated_dict_idx] = {'test_lang': lang, 'support': len(df_lang), 'qwk_fisher': average_qwk(df_lang[['qwk']])[1]}
            aggregated_dict_idx += 1
        
        df_model_results = pd.DataFrame.from_dict(aggregated_dict, orient='index')
        df_model_results.to_csv(os.path.join(results_path, model + '.csv'))



eval_condition(results_path='/Users/mariebexte/Coding/Projects/cross-lingual/exp_3_lolo/combine_all_other-sbert_pairs')
eval_condition(results_path='/Users/mariebexte/Coding/Projects/cross-lingual/exp_3_lolo/combine_downsampled_other-sbert_pairs')
# eval_condition(results_path='/Users/mariebexte/Coding/Projects/cross-lingual/exp_3_lolo/combine_all_other-sbert-3')
# eval_condition(results_path='/Users/mariebexte/Coding/Projects/cross-lingual/exp_3_lolo/combine_all_other-sbert-5')
# eval_condition(results_path='/Users/mariebexte/Coding/Projects/cross-lingual/exp_3_lolo/combine_downsampled_other')
# eval_condition(results_path='/Users/mariebexte/Coding/Projects/cross-lingual/exp_3_lolo/combine_downsampled_other_epochs-halved')