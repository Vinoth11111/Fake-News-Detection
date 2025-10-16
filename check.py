from sentence_transformers import SentenceTransformer
import joblib

models = SentenceTransformer('all-MiniLM-L6-v2')
texts = 'i love programming,python is great for programming and its widly used language'
embeddings = models.encode([texts])
pretrained_model = joblib.load('/Users/vinothkumar/Documents/Fake_News_Detection/model/fake_news_trained_model')
predic = pretrained_model.predict([embeddings[0]])
print(predic)
