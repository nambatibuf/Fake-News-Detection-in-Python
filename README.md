# Fake-News-Detection-in-Python

## Overview
This project focuses on detecting fake news using machine learning techniques. It includes preprocessing, feature extraction, and training various classifiers to predict the authenticity of news articles.

## Project Structure
- **Dataset**: The dataset (`WELFake_Dataset.csv`) contains information about news articles, including titles, text, and labels.

- **Code**: The main code (`fake_news_detection.ipynb` or `.py`) includes the following:
  - Importing necessary libraries and modules.
  - Loading and preprocessing the dataset.
  - Exploratory data analysis (EDA) and visualization.
  - Text cleaning and feature extraction using TF-IDF.
  - Training different classifiers (Naive Bayes, Passive Aggressive, SVM).
  - Evaluating and comparing the performance of classifiers.

## Requirements
- Python
- Libraries: pandas, scikit-learn, wordcloud, matplotlib, numpy, prml

## How to Run
1. Ensure you have the required dependencies installed (`pip install pandas scikit-learn wordcloud matplotlib numpy prml`).
2. Open the Jupyter Notebook or run the Python script.
3. Follow the code cells step by step to preprocess data, train classifiers, and evaluate performance.

## Results
The project evaluates the performance of three classifiers: Naive Bayes, Passive Aggressive, and Support Vector Machine (SVM). The evaluation metrics include accuracy, precision, recall, F1 score, and confusion matrices.

## Future Improvements
- Fine-tuning hyperparameters for better model performance.
- Experimenting with different feature extraction techniques.
- Handling imbalanced datasets for improved model generalization.

Feel free to explore the code and experiment with different parameters to enhance the performance of the fake news detection model.

