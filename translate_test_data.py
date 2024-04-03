import pandas as pd
import os, sys
import copy
import time
import gc
from easynmt import EasyNMT
import nltk
from torch.cuda import OutOfMemoryError
import torch


nltk.download('punkt')

data_path = '/data/exp'
languages = ['ar', 'da', 'en', 'he', 'it', 'ka', 'nb', 'pt', 'sl', 'sv', 'zh']
language_codes = {'ar': 'ar', 'da': 'da', 'en': 'en', 'he': 'he', 'it': 'it', 'ka': 'ka', 'nb': 'no', 'pt': 'pt', 'sl': 'sl', 'sv': 'sv', 'zh': 'zh'}

translation_model = EasyNMT('m2m_100_1.2B', max_loaded_models=1)

# translation_model.stop_multi_process_pool(process_pool)
# process_pool = translation_model.start_multi_process_pool(['cuda', 'cuda'])

for prompt in os.listdir(data_path):

    print(prompt)

    for language in languages:

        df_test = pd.read_csv(os.path.join(data_path, prompt, language, 'test.csv'))

        other_languages = copy.deepcopy(languages)
        other_languages.remove(language)

        for other_language in other_languages:

            if not os.path.exists(os.path.join(data_path, prompt, language, 'test_translated_m2m_100_1.2B_' + other_language + '.csv')):

                df_test_copy = copy.deepcopy(df_test)
                df_test_copy['Value'] = df_test_copy['Value'].apply(str)

                # if prompt == 'E011T02C' and language == 'he' and other_language == 'ka':
                # try:
                #     df_test_copy['Value'] = translation_model.translate(df_test_copy['Value'], source_lang=language_codes[language], target_lang=language_codes[other_language])
                # except:
                #     del translation_model
                #     gc.collect()
                #     torch.cuda.empty_cache()
                #     time.sleep(3)
                #     translation_model = EasyNMT('m2m_100_1.2B', max_loaded_models=1)
                #     df_test_copy['Value'] = [translation_model.translate(sentence, source_lang=language_codes[language], target_lang=language_codes[other_language]) for sentence in list(df_test_copy['Value'])]
                #     print('Processed sentence-by-sentence')

                # df_test_copy['Value'] = translation_model.translate(df_test_copy['Value'], source_lang=language_codes[language], target_lang=language_codes[other_language])
                # df_test_copy['Value'] = [translation_model.translate(sentence, source_lang=language_codes[language], target_lang=language_codes[other_language]) for sentence in list(df_test_copy['Value'])]
                
                # translations = []
                # original = list(df_test_copy['Value'])
                # translations = translations + (translation_model.translate(original[:10], source_lang=language_codes[language], target_lang=language_codes[other_language]))
                # translations = translations + (translation_model.translate(original[10:20], source_lang=language_codes[language], target_lang=language_codes[other_language]))
                # translations = translations + (translation_model.translate(original[20:30], source_lang=language_codes[language], target_lang=language_codes[other_language]))
                # translations = translations + (translation_model.translate(original[30:40], source_lang=language_codes[language], target_lang=language_codes[other_language]))
                # translations = translations + (translation_model.translate(original[40:50], source_lang=language_codes[language], target_lang=language_codes[other_language]))
                # translations = translations + (translation_model.translate(original[50:60], source_lang=language_codes[language], target_lang=language_codes[other_language]))
                # translations = translations + (translation_model.translate(original[60:70], source_lang=language_codes[language], target_lang=language_codes[other_language]))
                # translations = translations + (translation_model.translate(original[70:80], source_lang=language_codes[language], target_lang=language_codes[other_language]))
                # translations = translations + (translation_model.translate(original[80:90], source_lang=language_codes[language], target_lang=language_codes[other_language]))
                # translations = translations + (translation_model.translate(original[90:], source_lang=language_codes[language], target_lang=language_codes[other_language]))
                # # print(len(translations))
                # df_test_copy['Value'] = translations
                # # print(translations[0], original[0])

                translations = []
                original = list(df_test_copy['Value'])
                translations = translations + (translation_model.translate(original[:5], source_lang=language_codes[language], target_lang=language_codes[other_language]))
                translations = translations + (translation_model.translate(original[5:10], source_lang=language_codes[language], target_lang=language_codes[other_language]))
                translations = translations + (translation_model.translate(original[10:15], source_lang=language_codes[language], target_lang=language_codes[other_language]))
                translations = translations + (translation_model.translate(original[15:20], source_lang=language_codes[language], target_lang=language_codes[other_language]))
                translations = translations + (translation_model.translate(original[20:25], source_lang=language_codes[language], target_lang=language_codes[other_language]))
                translations = translations + (translation_model.translate(original[25:30], source_lang=language_codes[language], target_lang=language_codes[other_language]))
                translations = translations + (translation_model.translate(original[30:35], source_lang=language_codes[language], target_lang=language_codes[other_language]))
                translations = translations + (translation_model.translate(original[35:40], source_lang=language_codes[language], target_lang=language_codes[other_language]))
                translations = translations + (translation_model.translate(original[40:45], source_lang=language_codes[language], target_lang=language_codes[other_language]))
                translations = translations + (translation_model.translate(original[45:50], source_lang=language_codes[language], target_lang=language_codes[other_language]))
                translations = translations + (translation_model.translate(original[50:55], source_lang=language_codes[language], target_lang=language_codes[other_language]))
                translations = translations + (translation_model.translate(original[55:60], source_lang=language_codes[language], target_lang=language_codes[other_language]))
                translations = translations + (translation_model.translate(original[60:65], source_lang=language_codes[language], target_lang=language_codes[other_language]))
                translations = translations + (translation_model.translate(original[65:70], source_lang=language_codes[language], target_lang=language_codes[other_language]))
                translations = translations + (translation_model.translate(original[70:75], source_lang=language_codes[language], target_lang=language_codes[other_language]))
                translations = translations + (translation_model.translate(original[75:80], source_lang=language_codes[language], target_lang=language_codes[other_language]))
                translations = translations + (translation_model.translate(original[80:85], source_lang=language_codes[language], target_lang=language_codes[other_language]))
                translations = translations + (translation_model.translate(original[85:90], source_lang=language_codes[language], target_lang=language_codes[other_language]))
                translations = translations + (translation_model.translate(original[90:95], source_lang=language_codes[language], target_lang=language_codes[other_language]))
                translations = translations + (translation_model.translate(original[95:], source_lang=language_codes[language], target_lang=language_codes[other_language]))
                # print(len(translations))
                df_test_copy['Value'] = translations
                # print(translations[0], original[0])

                # df_test_copy['Value'] = translation_model.translate_multi_process(process_pool, df_test_copy['Value'], source_lang=language_codes[language], target_lang=language_codes[other_language])
            
                df_test_copy.to_csv(os.path.join(data_path, prompt, language, 'test_translated_m2m_100_1.2B_' + other_language + '.csv'))
                print('Translated ', prompt, language, other_language)
            # except OSError:
            #     print('No translation ', prompt, language, other_language)
            
            else:
                # print('Already translated ', prompt, language, other_language)
                pass

# translation_model.stop_multi_process_pool(process_pool)
