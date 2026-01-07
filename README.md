# Cross-Lingual Content Scoring 🌐

This repository contains code for our paper *The Benefits of Similarity-Based Content Scoring in a Massively Cross-Lingual Setting Based on Large-Scale Assessment Data*.
We're experimenting with cross-lingual content scoring on three datasets, comparing three scoring approaches and two base models.

### Data
- ePIRLS: Can not be shared
- ASAP_Translated: [ASAP short answer scoring dataset](https://www.kaggle.com/competitions/asap-sas) translated using [EasyNMT](https://github.com/UKPLab/EasyNMT) (in `data/ASAP`)
- ASAP_Multilingual: [Horbach et al., 2023](https://link.springer.com/article/10.1007/s40593-023-00370-1) (in `data/ASAP_crosslingual`)

### Similarity-based approaches we compare to instance-based classification
- Similarty-based scoring using SBERT: [Bexte et al., 2022](https://aclanthology.org/2022.bea-1.16.pdf)
- Neural Pairwise Contrastrive Regression (NPCR): [Xie et al., 2022](https://aclanthology.org/2022.coling-1.240.pdf)

### Base models
- SBERT: [paraphrase-multilingual-MiniLM-L12-v2](https://huggingface.co/sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2)
- XLMR: [xlmr-roberta-base](https://huggingface.co/FacebookAI/xlm-roberta-base)

### Languages
- Arabic
- Chinese
- Danish
- English
- French
- Georgian
- German
- Hebrew
- Italian
- Norwegian Bokmål
- Portuguese
- Slovenian
- Spanish
- Swedish

### Setup
In `config.py`, set the `dataset_path` attribute of the datasets you indend to use. Do also adjust the `MODEL_PATH` and `RESULT_PATH*` variables to suit your setup. To run the SBERT model in the instance-based scoring approach, execute `python3 download_model.py` once before starting the experiments.

### Experiments
For all experiments, pick which models to run by setting the respective flag in the bottom of the file to `TRUE`.

#### Experiment 1  (Monolingual baseline, zero-shot cross-lingual transfer and cross-lingual transfer via translated test data)
- `exp_1_zero_shot.py` for instance-based XLMR and Bexte et al. SBERT.
- `exp_1_zero_shot_model_swap.py` for Bexte et al. XLMR and instance-based SBERT.
- `exp_1_NPCR.py` for NPCR XLMR and NPCR SBERT.
- `exp_1_pretrained.py` for pretrained SBERT.
  
Each file has a standard function for ePIRLS and ASAP_Translated and a `*_cross_validated` one for ASAP_Multilingual.

#### Experiment 2 (Mixing data in a source language with gradually increasing amounts of target language data; only ePIRLS)
- `exp_2_tradeoff.py` (all models).

#### Experiment 3 (Combining multiple languages in the training data)
- `exp_3_lolo.py` for ePIRLS, ASAP_Translated (all models).
- `exp_3_lolo_cross_validated.py` for ASAP_Multilingual (all models).

### Analysis
Run `eval_exp_*.py` to evaluate the respective experiment.
- For experiment 1, this includes producing the cross-lingual scoring heatmaps shown in our paper.
- For experiment 2, this creates the matrixes we plot to show tradeoff between source and target languages.
- For experiment 3, this calculates the per-language and overall performance for the *sampled* and *full* condition.
