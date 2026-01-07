import os

from sentence_transformers import SentenceTransformer, models
from transformers import XLMRobertaModel, BertModel
from config import MODEL_PATH


# Save model to disk (required for use in classification setup)
model = SentenceTransformer('paraphrase-multilingual-MiniLM-L12-v2')
model.save(os.path.join(MODEL_PATH, 'paraphrase-multilingual-MiniLM-L12-v2'))