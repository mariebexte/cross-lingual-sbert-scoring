from sentence_transformers import SentenceTransformer

model = SentenceTransformer('paraphrase-multilingual-MiniLM-L12-v2')
print(model.tokenize('this is a sentence'))
model.save('/models/paraphrase-multilingual-MiniLM-L12-v2')

print('load')
model = SentenceTransformer('/models/paraphrase-multilingual-MiniLM-L12-v2')
print('loaded')