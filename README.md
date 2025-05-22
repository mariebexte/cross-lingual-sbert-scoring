# Cross-Lingual Content Scoring
### Comparison of two similarity-based approaches to instance-based classification
- Similarty-based scoring using SBERT: [Bexte et al., 2022](https://aclanthology.org/2022.bea-1.16.pdf)
- Neural Pairwise Contrastrive Regression (NPCR): [Xie et al., 2022](https://aclanthology.org/2022.coling-1.240.pdf)

### Base models
- SBERT: [paraphrase-multilingual-MiniLM-L12-v2](https://huggingface.co/sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2)
- XLMR: [xlmr-roberta-base](https://huggingface.co/FacebookAI/xlm-roberta-base)

### Data
- ePIRLS
- ASAP_Translated: [ASAP short answer scoring dataset](https://www.kaggle.com/competitions/asap-sas) translated using [EasyNMT](https://github.com/UKPLab/EasyNMT) (in `data/ASAP`)
- ASAP_Multilingual: [Horbach et al., 2023](https://link.springer.com/article/10.1007/s40593-023-00370-1) (in `data/ASAP_crosslingual`)

### Languages
- Danish, Norwegian Bokmål, Swedish, English, German, Italian, Portuguese, Spanish, French, Arabic, Hebrew, Georgian, Slovenian, Chinese

### Experiment 1 
- Monolingual baseline
- Zero-shot cross-lingual transfer
- Cross-lingual transfer via translated test data

### Experiment 2
- Mixing data in a source lanuage with gradually increasing amounts of target language data

### Experiment 3
- Combining multiple languages in the training data
