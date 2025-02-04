EPIRLS = {
    'dataset_path': '/data/exp',
    'dataset_name': 'ePIRLS',
    'id_column': 'id',
    'prompt_column': 'Variable',
    'answer_column': 'Value',
    'target_column': 'score',
    'languages': ['da', 'nb', 'sv', 'en', 'it', 'pt', 'ar', 'he', 'ka', 'sl', 'zh'],
    'translate_test': True,
    'language_column': 'Language',
    'prompts': ['E011B03C', 'E011B08C', 'E011B12C', 'E011B14C', 'E011M03C', 'E011M08C', 'E011M11C', 'E011M15C', 'E011R05C', 'E011R09C', 'E011R14C',
    'E011R16C', 'E011T05C', 'E011T09C', 'E011T17C', 'E011Z04C', 'E011Z12C', 'E011B04C', 'E011B09C', 'E011B13C', 'E011M02C', 'E011M04C', 'E011M09C',
    'E011M13C', 'E011R02C', 'E011R08C', 'E011R11C', 'E011R15C', 'E011T02C', 'E011T08C', 'E011T10C', 'E011Z02C', 'E011Z09C', 'E011Z14C']
}

ASAP_T = {
    'dataset_path': '/data/ASAP/split',
    'dataset_name': 'ASAP_translated',
    'id_column': 'ItemId',
    'prompt_column': 'PromptId',
    'answer_column': 'AnswerText',
    'target_column': 'Score1',
    'languages': ['da', 'nb', 'sv', 'en', 'it', 'pt', 'ar', 'he', 'ka', 'sl', 'zh'],
    'translate_test': False,
    'language_column': None,
    'prompts': ['1', '2', '3', '4', '5', '6', '7', '8', '9', '10']
}

ASAP_M = {
    'dataset_path': '/data/ASAP_crosslingual/split',
    'dataset_name': 'ASAP_multilingual',
    'id_column': 'id',
    'prompt_column': 'prompt',
    'answer_column': 'text',
    'target_column': 'score',
    'languages': ['de', 'en', 'es', 'fr', 'zh'],
    'translate_test': True,
    'num_folds': 7,
    'language_column': None,
    'prompts': ['1', '2', '10'] 
}
