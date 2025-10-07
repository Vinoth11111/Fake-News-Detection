import pandas as pd
import numpy as np
from fake_news_detector import FakeNewsDetector
from sklearn.model_selection import train_test_split

def create_sample_dataset():
    """Create a sample dataset for demonstration"""
    fake_news = [
        "Scientists discover aliens living among us, government covers it up",
        "Miracle cure for all diseases found in common household item",
        "President secretly planning to move capital to Mars",
        "New study shows eating rocks makes you immortal",
        "Breaking: Time travel invented by teenager in garage",
        "Shocking revelation: Birds aren't real, they're government drones",
        "Celebrity admits to being a robot from the future",
        "Local man wins lottery 50 times in a row, officials baffled",
        "Doctors hate this one weird trick to lose weight instantly",
        "World leaders are secretly lizard people from another dimension",
        "Vaccines contain microchips to control your mind",
        "Moon landing was completely fake and filmed in Hollywood",
        "Drinking bleach cures coronavirus says unlicensed expert",
        "Earth is flat and NASA has been lying to us",
        "5G towers cause cancer and control weather patterns",
        "Chocolate is actually a vegetable according to new study",
        "Breaking: Bigfoot captured and held in secret government facility",
        "Miracle pill makes you younger overnight doctors shocked",
        "Secret society controls all world governments from shadows",
        "Tap water contains chemicals turning frogs gay"
    ]
    
    real_news = [
        "Stock market closes higher amid positive economic data",
        "New research suggests climate change affecting weather patterns",
        "Government announces new infrastructure spending plan",
        "Scientists make progress in cancer treatment research",
        "Technology company releases quarterly earnings report",
        "International summit addresses global trade concerns",
        "University study examines effects of social media usage",
        "City council approves budget for next fiscal year",
        "Federal Reserve announces interest rate decision",
        "New employment data shows job market growth",
        "Health officials recommend annual flu vaccinations",
        "Transportation department plans highway expansion project",
        "Education board reviews curriculum standards",
        "Environmental agency releases pollution report",
        "Trade negotiations continue between nations",
        "Supreme court hears arguments on constitutional case",
        "Agricultural department issues crop yield forecast",
        "Space agency announces next satellite launch schedule",
        "Public health officials monitor disease outbreak",
        "Central bank publishes economic growth projections"
    ]
    
    # Create dataframe
    data = {
        'text': fake_news + real_news,
        'label': [0] * len(fake_news) + [1] * len(real_news)  # 0 = Fake, 1 = Real
    }
    
    return pd.DataFrame(data)

def load_dataset(filepath=None):
    """Load dataset from file or create sample dataset"""
    if filepath:
        try:
            df = pd.read_csv(filepath)
            return df
        except FileNotFoundError:
            print(f"File {filepath} not found. Using sample dataset.")
            return create_sample_dataset()
    else:
        return create_sample_dataset()

def train_model(dataset_path=None, model_type='naive_bayes'):
    """Train the fake news detection model"""
    # Load dataset
    df = load_dataset(dataset_path)
    
    print(f"Dataset loaded: {len(df)} samples")
    print(f"Fake news: {sum(df['label'] == 0)}, Real news: {sum(df['label'] == 1)}")
    
    # Split data
    X = df['text'].values
    y = df['label'].values
    
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )
    
    print(f"\nTraining set: {len(X_train)} samples")
    print(f"Test set: {len(X_test)} samples")
    
    # Initialize and train detector
    detector = FakeNewsDetector()
    print(f"\nTraining {model_type} model...")
    detector.train(X_train, y_train, model_type=model_type)
    
    # Evaluate
    print("\nEvaluating model...")
    results = detector.evaluate(X_test, y_test)
    
    print(f"\nAccuracy: {results['accuracy']:.4f}")
    print("\nClassification Report:")
    print(results['classification_report'])
    print("\nConfusion Matrix:")
    print(results['confusion_matrix'])
    
    # Save model
    model_filename = f'fake_news_{model_type}_model.pkl'
    detector.save_model(model_filename)
    print(f"\nModel saved to {model_filename}")
    
    return detector

def predict_news(text, model_path='fake_news_naive_bayes_model.pkl'):
    """Predict if a news article is fake or real"""
    detector = FakeNewsDetector()
    detector.load_model(model_path)
    
    prediction = detector.predict([text])[0]
    
    return "Real" if prediction == 1 else "Fake"

if __name__ == "__main__":
    import sys
    
    if len(sys.argv) > 1 and sys.argv[1] == 'train':
        # Training mode
        dataset_path = sys.argv[2] if len(sys.argv) > 2 else None
        model_type = sys.argv[3] if len(sys.argv) > 3 else 'naive_bayes'
        train_model(dataset_path, model_type)
    elif len(sys.argv) > 1 and sys.argv[1] == 'predict':
        # Prediction mode
        if len(sys.argv) < 3:
            print("Usage: python train.py predict <text> [model_path]")
            sys.exit(1)
        
        text = sys.argv[2]
        model_path = sys.argv[3] if len(sys.argv) > 3 else 'fake_news_naive_bayes_model.pkl'
        
        result = predict_news(text, model_path)
        print(f"\nPrediction: {result}")
    else:
        # Default: train with sample data
        print("Training with sample dataset...")
        print("Usage: python train.py [train|predict] [args]")
        print("\nTraining:")
        print("  python train.py train [dataset_path] [logistic|naive_bayes]")
        print("\nPrediction:")
        print("  python train.py predict <text> [model_path]")
        print("\n" + "="*50 + "\n")
        
        train_model()
