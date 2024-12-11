from sentence_transformers import SentenceTransformer, models
from transformers import XLMRobertaModel, BertModel


# Save model to disk (required for use in classification setup)
model = SentenceTransformer('paraphrase-multilingual-MiniLM-L12-v2')
model.save('/models/paraphrase-multilingual-MiniLM-L12-v2')