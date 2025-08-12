EPIRLS = {
    'dataset_path': '/data/exp',
    'dataset_name': 'ePIRLS',
    'id_column': 'id',
    'prompt_column': 'Variable',
    'answer_column': 'Value',
    'target_column': 'score',
    'languages': ['ar', 'da', 'en', 'he', 'it', 'ka', 'nb', 'pt', 'sl', 'sv', 'zh'],
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
    'languages': ['ar', 'da', 'en', 'he', 'it', 'ka', 'nb', 'pt', 'sl', 'sv', 'zh'],
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

RANDOM_SEED=8742341

ANSWER_LENGTH = 128
PATIENCE = None

SBERT_BASE_MODEL = 'paraphrase-multilingual-MiniLM-L12-v2'
XLMR_BASE_MODEL = 'xlm-roberta-base'

RESULT_PATH_EXP_1 = '/results/fair_final/exp_1_zero_shot'
RESULT_PATH_EXP_2 = '/results/fair_final/exp_2_tradeoff'
RESULT_PATH_EXP_3 = '/results/fair_final/exp_3_lolo'

SBERT_NUM_EPOCHS = 50
NPCR_NUM_EPOCHS = 50
BERT_NUM_EPOCHS = 50

SBERT_BATCH_SIZE = 16
NPCR_BATCH_SIZE = 16
BERT_BATCH_SIZE = 16

SBERT_BATCH_SIZE_ASAP_M = 16
NPCR_BATCH_SIZE_ASAP_M = 16
BERT_BATCH_SIZE_ASAP_M = 16

SBERT_NUM_VAL_PAIRS = 10
NPCR_NUM_VAL_PAIRS = 10
NPCR_NUM_TEST_PAIRS = 50
