from fake_news_detector import FakeNewsDetector
import sys

def main():
    """Main entry point for fake news detection"""
    if len(sys.argv) < 2:
        print("Fake News Detection System")
        print("=" * 50)
        print("\nUsage:")
        print("  python main.py <news_text>")
        print("\nExample:")
        print('  python main.py "Scientists discover cure for all diseases"')
        print("\nNote: Make sure to train the model first using train.py")
        sys.exit(1)
    
    # Get text from command line
    text = ' '.join(sys.argv[1:])
    
    # Load model and predict
    try:
        detector = FakeNewsDetector()
        detector.load_model('fake_news_naive_bayes_model.pkl')
        
        prediction = detector.predict([text])[0]
        
        print("\n" + "=" * 50)
        print("FAKE NEWS DETECTION RESULT")
        print("=" * 50)
        print(f"\nNews text: {text}")
        print(f"\nPrediction: {'REAL NEWS' if prediction == 1 else 'FAKE NEWS'}")
        print("\n" + "=" * 50)
        
    except FileNotFoundError:
        print("\nError: Model file not found!")
        print("Please train the model first by running:")
        print("  python train.py train")
        sys.exit(1)

if __name__ == "__main__":
    main()
