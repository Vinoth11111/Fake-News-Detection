"""from sentence_transformers import SentenceTransformer
import joblib

models = SentenceTransformer('all-MiniLM-L6-v2')
texts = 'i love programming,python is great for programming and its widly used language'
embeddings = models.encode([texts])
pretrained_model = joblib.load('/Users/vinothkumar/Documents/Fake_News_Detection/model/fake_news_trained_model')
predic = pretrained_model.predict([embeddings[0]])
print(predic)
"""
import pandas as pd
import sklearn
import numpy as np
import streamlit as st
import joblib

print('pandas version:', pd.__version__)#2.3.3
import sys
print(sys.version)
print('sklearn version:', sklearn.__version__)
print('numpy version:', np.__version__)
print('streamlit version:', st.__version__)
print('joblib version:', joblib.__version__)
import sentence_transformers
print(sentence_transformers.__version__)
