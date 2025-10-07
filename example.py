#!/usr/bin/env python3
"""
Example usage of the Fake News Detection system
"""

from fake_news_detector import FakeNewsDetector

def example_usage():
    """Demonstrate the fake news detection system"""
    
    print("=" * 60)
    print("FAKE NEWS DETECTION - Example Usage")
    print("=" * 60)
    
    # Example news texts to classify
    news_samples = [
        "Scientists discover aliens living among us, government covers it up",
        "Stock market closes higher amid positive economic data",
        "Miracle cure for all diseases found in common household item",
        "Government announces new infrastructure spending plan",
        "Breaking: Time travel invented by teenager in garage",
        "University study examines effects of social media usage",
        "Shocking revelation: Birds aren't real, they're government drones",
        "International summit addresses global trade concerns"
    ]
    
    # Load the pre-trained model
    print("\n1. Loading pre-trained model...")
    detector = FakeNewsDetector()
    
    try:
        detector.load_model('fake_news_naive_bayes_model.pkl')
        print("   ✓ Model loaded successfully")
    except FileNotFoundError:
        print("   ✗ Model not found. Training new model with sample data...")
        from train import train_model
        detector = train_model()
    
    # Make predictions
    print("\n2. Making predictions on sample news articles:")
    print("-" * 60)
    
    for i, news in enumerate(news_samples, 1):
        prediction = detector.predict([news])[0]
        label = "REAL NEWS" if prediction == 1 else "FAKE NEWS"
        
        print(f"\n{i}. {news[:50]}...")
        print(f"   → {label}")
    
    print("\n" + "=" * 60)
    print("\n3. Try it yourself!")
    print("   Run: python main.py \"Your news text here\"")
    print("=" * 60)

if __name__ == "__main__":
    example_usage()
