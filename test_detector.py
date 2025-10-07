#!/usr/bin/env python3
"""
Test script for Fake News Detection system
"""

from fake_news_detector import FakeNewsDetector
import numpy as np

def test_preprocessing():
    """Test text preprocessing"""
    detector = FakeNewsDetector()
    
    # Test with normal text
    text = "This is a TEST with CAPS and numbers 123!"
    processed = detector.preprocess_text(text)
    assert processed == "this is a test with caps and numbers", f"Expected 'this is a test with caps and numbers', got '{processed}'"
    
    # Test with None
    processed = detector.preprocess_text(None)
    assert processed == "", f"Expected empty string for None, got '{processed}'"
    
    print("✓ Preprocessing tests passed")

def test_training_and_prediction():
    """Test model training and prediction"""
    detector = FakeNewsDetector()
    
    # Create simple dataset
    X_train = [
        "aliens are among us government conspiracy",
        "miracle cure found in kitchen",
        "stock market rises on good news",
        "government announces new policy"
    ]
    y_train = [0, 0, 1, 1]  # 0 = fake, 1 = real
    
    # Train
    detector.train(X_train, y_train, model_type='naive_bayes')
    
    # Test predictions
    X_test = [
        "aliens conspiracy theory",
        "economic report released"
    ]
    predictions = detector.predict(X_test)
    
    assert len(predictions) == 2, f"Expected 2 predictions, got {len(predictions)}"
    assert predictions[0] in [0, 1], f"Prediction should be 0 or 1, got {predictions[0]}"
    
    print("✓ Training and prediction tests passed")

def test_model_persistence():
    """Test saving and loading models"""
    import os
    import tempfile
    
    detector1 = FakeNewsDetector()
    
    # Train a simple model
    X_train = ["fake news here", "real news here"]
    y_train = [0, 1]
    detector1.train(X_train, y_train, model_type='naive_bayes')
    
    # Save to temp file
    with tempfile.NamedTemporaryFile(delete=False, suffix='.pkl') as tmp:
        tmp_path = tmp.name
    
    try:
        detector1.save_model(tmp_path)
        
        # Load in new detector
        detector2 = FakeNewsDetector()
        detector2.load_model(tmp_path)
        
        # Compare predictions
        test_text = ["fake news"]
        pred1 = detector1.predict(test_text)
        pred2 = detector2.predict(test_text)
        
        assert np.array_equal(pred1, pred2), "Loaded model should give same predictions"
        
        print("✓ Model persistence tests passed")
    finally:
        # Cleanup
        if os.path.exists(tmp_path):
            os.remove(tmp_path)

def test_evaluation():
    """Test model evaluation"""
    detector = FakeNewsDetector()
    
    # Train with simple data
    X_train = ["fake", "real"]
    y_train = [0, 1]
    detector.train(X_train, y_train, model_type='naive_bayes')
    
    # Evaluate
    X_test = ["fake", "real"]
    y_test = [0, 1]
    results = detector.evaluate(X_test, y_test)
    
    assert 'accuracy' in results, "Results should contain accuracy"
    assert 'classification_report' in results, "Results should contain classification report"
    assert 'confusion_matrix' in results, "Results should contain confusion matrix"
    assert 0 <= results['accuracy'] <= 1, f"Accuracy should be between 0 and 1, got {results['accuracy']}"
    
    print("✓ Evaluation tests passed")

if __name__ == "__main__":
    print("Running Fake News Detector Tests...")
    print("=" * 50)
    
    test_preprocessing()
    test_training_and_prediction()
    test_model_persistence()
    test_evaluation()
    
    print("\n" + "=" * 50)
    print("All tests passed! ✓")
