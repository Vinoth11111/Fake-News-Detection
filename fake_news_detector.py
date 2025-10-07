import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import nltk
from nltk.corpus import stopwords
import re
import pickle

class FakeNewsDetector:
    def __init__(self):
        self.vectorizer = TfidfVectorizer(
            max_features=5000, 
            stop_words='english',
            ngram_range=(1, 2),
            min_df=1
        )
        self.model = None
        
    def preprocess_text(self, text):
        """Preprocess text by removing special characters and converting to lowercase"""
        if pd.isna(text):
            return ""
        text = str(text).lower()
        text = re.sub(r'[^a-zA-Z\s]', '', text)
        text = re.sub(r'\s+', ' ', text).strip()
        return text
    
    def train(self, X_train, y_train, model_type='logistic'):
        """Train the model on the training data"""
        # Preprocess texts
        X_train_processed = [self.preprocess_text(text) for text in X_train]
        
        # Vectorize the text
        X_train_vectors = self.vectorizer.fit_transform(X_train_processed)
        
        # Train the model
        if model_type == 'logistic':
            self.model = LogisticRegression(
                max_iter=1000, 
                random_state=42, 
                C=0.5,
                solver='liblinear'
            )
        elif model_type == 'naive_bayes':
            self.model = MultinomialNB(alpha=0.1)
        else:
            raise ValueError("model_type must be 'logistic' or 'naive_bayes'")
        
        self.model.fit(X_train_vectors, y_train)
        
    def predict(self, X_test):
        """Predict labels for test data"""
        if self.model is None:
            raise ValueError("Model not trained yet. Call train() first.")
        
        # Preprocess texts
        X_test_processed = [self.preprocess_text(text) for text in X_test]
        
        # Vectorize the text
        X_test_vectors = self.vectorizer.transform(X_test_processed)
        
        # Make predictions
        predictions = self.model.predict(X_test_vectors)
        return predictions
    
    def evaluate(self, X_test, y_test):
        """Evaluate model performance"""
        predictions = self.predict(X_test)
        
        accuracy = accuracy_score(y_test, predictions)
        report = classification_report(y_test, predictions, target_names=['Fake', 'Real'])
        conf_matrix = confusion_matrix(y_test, predictions)
        
        return {
            'accuracy': accuracy,
            'classification_report': report,
            'confusion_matrix': conf_matrix
        }
    
    def save_model(self, filepath='fake_news_model.pkl'):
        """Save the trained model and vectorizer"""
        if self.model is None:
            raise ValueError("Model not trained yet. Call train() first.")
        
        with open(filepath, 'wb') as f:
            pickle.dump({'model': self.model, 'vectorizer': self.vectorizer}, f)
    
    def load_model(self, filepath='fake_news_model.pkl'):
        """Load a trained model and vectorizer"""
        with open(filepath, 'rb') as f:
            data = pickle.load(f)
            self.model = data['model']
            self.vectorizer = data['vectorizer']
