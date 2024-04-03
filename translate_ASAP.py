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

data_path = '/data/ASAP/split'
other_languages = ['ar', 'da', 'he', 'it', 'ka', 'nb', 'pt', 'sl', 'sv', 'zh']
language_codes = {'ar': 'ar', 'da': 'da', 'en': 'en', 'he': 'he', 'it': 'it', 'ka': 'ka', 'nb': 'no', 'pt': 'pt', 'sl': 'sl', 'sv': 'sv', 'zh': 'zh'}

translation_model = EasyNMT('m2m_100_1.2B', max_loaded_models=1)


for prompt in os.listdir(data_path):

    print(prompt)

    for split in ['train.csv', 'val.csv', 'test.csv']:

        # All original data is English
        df = pd.read_csv(os.path.join(data_path, prompt, 'en', split))

        for other_language in other_languages:

            if not os.path.exists(os.path.join(data_path, prompt, other_language, split)):

                if not os.path.exists(os.path.join(data_path, prompt, other_language)):
                    os.mkdir(os.path.join(data_path, prompt, other_language))

                df_copy = copy.deepcopy(df)
                df_copy['AnswerText'] = df_copy['AnswerText'].apply(str)

                # df_test_copy['Value'] = translation_model.translate(df_test_copy['Value'], source_lang=language_codes[language], target_lang=language_codes[other_language])
                df_copy['AnswerText'] = [translation_model.translate(sentence, source_lang=language_codes['en'], target_lang=language_codes[other_language]) for sentence in list(df_copy['AnswerText'])]
                
                df_copy.to_csv(os.path.join(data_path, prompt, other_language, split))
                print('Translated', prompt, language, other_language)
