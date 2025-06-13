# --- 0. Import Libraries ---
import pandas as pd
import numpy as np
import re
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA

# --- 1. Data Preprocessing Functions ---


def drop_duplicates_empty (df):
    df = df.dropna(subset=['reviews.text', 'reviews.rating'])
    df = df.drop_duplicates(subset=['asins', 'reviews.text'])
    return df

def clean_text(text):
    text = re.sub(r"[^a-zA-Z0-9\s]", '', str(text))
    text = re.sub(r'\s+', ' ', text)
    return text.strip().lower()

def preprocess_reviews(df):
    # Combine title and text, clean, drop NA
    df['combined_reviews'] = df['reviews.title'].fillna('') + ' ' + df['reviews.text'].fillna('')
    df['combined_reviews'] = df['combined_reviews'].apply(clean_text)
    df = df.drop(['reviews.title', 'reviews.text'], axis=1)
    df = df.dropna(subset=['combined_reviews', 'reviews.rating'])
    return df

def tokenize_reviews(df, col='combined_reviews', new_col='tokens'):
    df[new_col] = df[col].apply(word_tokenize)
    return df

def remove_stopwords(df, tokens_col='tokens', new_col='tokens_nostop', language='english'):
    stop_words = set(stopwords.words(language))
    df[new_col] = df[tokens_col].apply(lambda tokens: [w for w in tokens if w.lower() not in stop_words])
    return df

def lemmatize_tokens(df, tokens_col='tokens_nostop', new_col='tokens_lemmatized'):
    lemmatizer = WordNetLemmatizer()
    df[new_col] = df[tokens_col].apply(lambda tokens: [lemmatizer.lemmatize(token) for token in tokens])
    return df

def add_sentiment_column(df):
    def categorize_rating(rating):
        if rating >= 4:
            return 'positive'
        elif rating == 3:
            return 'neutral'
        else:
            return 'negative'
    df['sentiment'] = df['reviews.rating'].apply(categorize_rating)
    return df

def preprocess_and_lemmatize_names_categories(df):
    """
    Preprocesses and lemmatizes the 'name' and 'categories' columns:
    - Fills missing values
    - Combines into a new column 'name_category'
    - Cleans text (lowercase, removes unwanted characters)
    - Tokenizes, removes stopwords, and lemmatizes
    - Joins lemmatized tokens into a string in 'name_category_lemmatized'
    Returns the DataFrame with new columns.
    """
    import re
    from nltk.corpus import stopwords
    from nltk.tokenize import word_tokenize
    from nltk.stem import WordNetLemmatizer

    # Fill missing values
    df['name'] = df['name'].fillna('')
    df['name'] = df['name'].str.replace('Includes Special Offers', '', regex=False).str.strip()
    df['categories'] = df['categories'].fillna('')

    # Combine into a single column
    df['name_category'] = df['name'] + ' ' + df['categories']

    # Clean text: lowercase, remove unwanted characters, strip
    def clean_text(text):
        text = text.lower()
        text = re.sub(r"[^a-zA-Z0-9\s]", '', text)
        text = re.sub(r'\s+', ' ', text)
        return text.strip()
    df['name_category_clean'] = df['name_category'].apply(clean_text)

    # Tokenize
    df['name_category_tokens'] = df['name_category_clean'].apply(word_tokenize)

    # Remove stopwords
    stop_words = set(stopwords.words('english'))
    df['name_category_nostop'] = df['name_category_tokens'].apply(
        lambda tokens: [word for word in tokens if word not in stop_words]
    )

    # Lemmatize
    lemmatizer = WordNetLemmatizer()
    df['name_category_lemmatized'] = df['name_category_nostop'].apply(
        lambda tokens: [lemmatizer.lemmatize(token) for token in tokens]
    )

    # Join lemmatized tokens into a string
    df['name_category_lemmatized'] = df['name_category_lemmatized'].apply(lambda tokens: ' '.join(tokens))

    return df

def preprocess_pipeline(df):
    df = drop_duplicates_empty(df)
    df = preprocess_reviews(df)
    df = add_sentiment_column(df)
    df = tokenize_reviews(df)
    df = remove_stopwords(df)
    df = lemmatize_tokens(df)
    df['lemmatized_str'] = df['tokens_lemmatized'].apply(lambda tokens: ' '.join(tokens))
    return df