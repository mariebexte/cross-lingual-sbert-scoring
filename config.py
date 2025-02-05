EPIRLS = {
    'dataset_path': '/data/exp',
    'dataset_name': 'ePIRLS',
    'id_column': 'id',
    'prompt_column': 'Variable',
    'answer_column': 'Value',
    'target_column': 'score',
    'languages': ['ar', 'da', 'en', 'he', 'it', 'ka', 'nb', 'pt', 'sl', 'sv', 'zh'],
    'translate_test': True,
    'language_column': 'Language'
}

ASAP_T = {
    'dataset_path': '/data/ASAP/split',
    'dataset_name': 'ASAP_translated',
    'id_column': 'ItemId',
    'prompt_column': 'PromptId',
    'answer_column': 'AnswerText',
    'target_column': 'Score1',
    'languages': ['ar', 'da', 'en', 'he', 'it', 'ka', 'nb', 'pt', 'sl', 'sv', 'zh'],
    'translate_test': False,
    'language_column': None
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
    'language_column': None
}

#SBERT_BASE_MODEL = 'sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2'
SBERT_BASE_MODEL = 'paraphrase-multilingual-MiniLM-L12-v2'
XLMR_BASE_MODEL = 'xlm-roberta-base'

RESULT_PATH_EXP_1 = '/results/exp_1_zero_shot'
RESULT_PATH_EXP_2 = '/results/exp_2_tradeoff'
RESULT_PATH_EXP_3 = '/results/exp_3_lolo'

SBERT_NUM_EPOCHS = 8
BERT_NUM_EPOCHS = 10

SBERT_NUM_PAIRS = 25
SBERT_NUM_VAL_PAIRS = 1000

NPCR_ANSWER_LENGTH = 128
