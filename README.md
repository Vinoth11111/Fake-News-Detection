# Fake News Detection

A machine learning-based system for detecting fake news articles using Natural Language Processing (NLP) techniques.

## Features

- Text preprocessing and cleaning
- TF-IDF vectorization for feature extraction
- Multiple ML models support (Logistic Regression, Naive Bayes)
- Model training and evaluation
- Prediction on new text
- Model persistence (save/load)

## Installation

1. Clone the repository:
```bash
git clone https://github.com/Vinoth11111/Fake-News-Detection.git
cd Fake-News-Detection
```

2. Install dependencies:
```bash
pip install -r requirements.txt
```

## Usage

### Quick Start - Run Example

See the system in action with pre-defined examples:
```bash
python example.py
```

### Training the Model

Train with sample dataset:
```bash
python train.py train
```

Train with custom dataset:
```bash
python train.py train path/to/dataset.csv
```

Train with different model:
```bash
python train.py train path/to/dataset.csv naive_bayes
```

### Making Predictions

Using the main script:
```bash
python main.py "Your news text here"
```

Using the train script:
```bash
python train.py predict "Your news text here"
```

## Dataset Format

If you want to use your own dataset, it should be a CSV file with the following columns:
- `text`: The news article text
- `label`: 0 for fake news, 1 for real news

Example:
```csv
text,label
"Breaking news about something fake",0
"Actual news about real events",1
```

## Model Performance

The system uses TF-IDF features combined with machine learning classifiers:
- **Logistic Regression**: Good baseline performance
- **Naive Bayes**: Fast training and prediction (default)

## Testing

Run the test suite to verify functionality:
```bash
python test_detector.py
```

## Project Structure

```
Fake-News-Detection/
├── fake_news_detector.py   # Main detector class
├── train.py                 # Training and prediction script
├── main.py                  # Main entry point for predictions
├── example.py               # Example usage script
├── test_detector.py         # Unit tests
├── requirements.txt         # Python dependencies
├── .gitignore              # Git ignore file
└── README.md               # This file
```

## Example

```python
from fake_news_detector import FakeNewsDetector

# Create and train detector
detector = FakeNewsDetector()
detector.train(X_train, y_train, model_type='logistic')

# Make predictions
predictions = detector.predict(X_test)

# Evaluate
results = detector.evaluate(X_test, y_test)
print(f"Accuracy: {results['accuracy']}")
```

## License

This project is open source and available for educational purposes.
