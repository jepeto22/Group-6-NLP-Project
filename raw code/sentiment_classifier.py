# model1_sentiment_classifier.py

"""
Model 1 - Sentiment Classifier

This script provides functions to prepare features, train, select, and evaluate sentiment classification models for Amazon product reviews.
"""

from sklearn.linear_model import LogisticRegression
from sklearn.svm import LinearSVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.naive_bayes import MultinomialNB
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.feature_extraction.text import TfidfVectorizer
from scipy.sparse import hstack
import numpy as np
import pandas as pd

def prepare_sentiment_features(df, text_col='lemmatized_str', label_col='sentiment'):
    """
    Prepares features and labels for sentiment classification.
    - Splits data into train/test sets.
    - Vectorizes text using TF-IDF.
    - Adds review length as a feature.
    Returns: X_train_features, X_test_features, y_train, y_test, X_test_text, tfidf_vectorizer
    """
    X_text = df[text_col]
    y = df[label_col]
    X_train_text, X_test_text, y_train, y_test = train_test_split(
        X_text, y, test_size=0.2, random_state=42, stratify=y
    )
    tfidf_vectorizer = TfidfVectorizer(ngram_range=(1,2), min_df=3, max_df=0.9)
    X_train_tfidf = tfidf_vectorizer.fit_transform(X_train_text)
    X_test_tfidf = tfidf_vectorizer.transform(X_test_text)
    train_length = np.array([len(x.split()) for x in X_train_text]).reshape(-1, 1)
    test_length = np.array([len(x.split()) for x in X_test_text]).reshape(-1, 1)
    X_train_features = hstack([X_train_tfidf, train_length])
    X_test_features = hstack([X_test_tfidf, test_length])
    return X_train_features, X_test_features, y_train, y_test, X_test_text, tfidf_vectorizer

def get_sentiment_models():
    """
    Returns a list of tuples: (model name, model instance, parameter grid)
    """
    return [
        (
            "Logistic Regression",
            LogisticRegression(max_iter=2000, random_state=42),
            {'C': [0.01, 0.1, 1, 10]}
        ),
        (
            "Linear SVC",
            LinearSVC(max_iter=1000, random_state=42),
            {'C': [0.01, 0.1, 1, 10]}
        ),
        (
            "Random Forest",
            RandomForestClassifier(random_state=42),
            {'n_estimators': [50, 100], 'max_depth': [None, 10, 20]}
        ),
        (
            "Multinomial NB",
            MultinomialNB(),
            {'alpha': [0.5, 1.0, 2.0]}
        )
    ]

def train_best_sentiment_model(X_train, y_train, models_and_grids):
    """
    Trains models using GridSearchCV and selects the best one based on F1-macro score.
    Returns: (best_model_name, best_score, best_model, results_list)
    """
    results = []
    for name, model, param_grid in models_and_grids:
        print(f"\n{name} - GridSearchCV")
        grid = GridSearchCV(model, param_grid, cv=3, scoring='f1_macro', n_jobs=-1)
        try:
            grid.fit(X_train, y_train)
            print("Best params:", grid.best_params_)
            print("Best CV F1-macro:", grid.best_score_)
            results.append((name, grid.best_score_, grid.best_estimator_))
        except Exception as e:
            print(f"Error with {name}: {e}")
    if results:
        best_name, best_score, best_model = max(results, key=lambda x: x[1])
        return best_name, best_score, best_model, results
    else:
        return None, None, None, results

def evaluate_sentiment_model(model, X_test, y_test, X_test_text):
    """
    Evaluates the model on the test set and prints classification report and confusion matrix, as well as showing the missclassified examples.
    Returns: y_pred, misclassified_df
    """
    y_pred = model.predict(X_test)
    print(classification_report(y_test, y_pred))
    print("Confusion Matrix:\n", confusion_matrix(y_test, y_pred))
    mis_idx = np.where(y_pred != y_test)[0]
    y_test_reset = y_test.reset_index(drop=True)
    misclassified_df = pd.DataFrame({
        'actual_sentiment': y_test_reset.iloc[mis_idx].values,
        'predicted_sentiment': y_pred[mis_idx],
        'combined_reviews': X_test_text.iloc[mis_idx].values
    })
    pd.set_option('display.max_colwidth', None)
    print(misclassified_df.head())
    return y_pred, misclassified_df