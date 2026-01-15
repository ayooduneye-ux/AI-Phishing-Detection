# AI-Driven Phishing Detection System (TF-IDF + Logistic Regression)

## Overview
This repository contains a supervised machine learning system that detects phishing-style messages using Natural Language Processing (NLP).
It is implemented using TF-IDF feature extraction and a Logistic Regression classifier.

## Dataset
This project uses the **SMS Spam Collection** dataset. The original labels are:
- **spam** (treated as *phishing* for this coursework)
- **ham** (treated as *legitimate*)

The dataset is stored locally as:
- `data/phishing_dataset.csv`

> Note: The Kaggle/UCI version commonly uses columns `v1` (label) and `v2` (message).

## Method
- Text cleaning (lowercasing, noise removal, URL/email normalisation)
- TF-IDF vectorisation (unigrams + bigrams)
- Logistic Regression (regularised)
- Evaluation using accuracy, confusion matrix, precision, recall, F1-score

## How to Run
1. Install dependencies:
```bash
pip install -r requirements.txt
```

2. Run:
```bash
python src/phishing_detection.py
```

## Outputs
The script prints:
- Accuracy
- Confusion matrix
- Classification report (precision/recall/F1)

## Ethical Considerations
This system is for educational purposes. Any real-world deployment should comply with data protection regulations (e.g., GDPR) and include human oversight.

## Author
<Your Name> — University College Birmingham
