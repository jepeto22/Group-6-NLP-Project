# 0. Imports and Setup

import os
import re
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from collections import Counter
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from sklearn.model_selection import train_test_split, GridSearchCV, cross_val_score
from sklearn.pipeline import Pipeline
from sklearn.svm import LinearSVC
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.metrics import silhouette_score
from kneed import KneeLocator
from scipy.sparse import hstack
import joblib
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer
from transformers import pipeline, AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
import torch
import gradio as gr
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import classification_report, confusion_matrix

nltk.download('punkt')
nltk.download('punkt_tab')
nltk.download('stopwords')
nltk.download('wordnet')

import gradio as gr
from dotenv import load_dotenv

from data_preprocessing import preprocess_pipeline, preprocess_and_lemmatize_names_categories
from clustering import cluster_and_label_products
from blog_generation import generate_blogposts_for_all_categories



# Load environment variables
load_dotenv()
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
HF_KEY = os.getenv("HF_KEY")

# Load trained sentiment model and vectorizer
sentiment_model = joblib.load("models/best_sentiment_model.pkl")
tfidf_vectorizer = joblib.load("models/tfidf_vectorizer_sentiment.pkl")

# Load blog generator model/tokenizer
def load_blog_generator():
    model_path = r"models/blog_generator"
    bnb_config = BitsAndBytesConfig(
        load_in_8bit=True,
        bnb_8bit_quant_type="nf8",
        bnb_8bit_compute_dtype=torch.float16,
        bnb_8bit_use_double_quant=False,
    )
    model = AutoModelForCausalLM.from_pretrained(
        model_path,
        quantization_config=bnb_config,
        device_map="auto"
    )
    tokenizer = AutoTokenizer.from_pretrained(model_path)
    return pipeline("text-generation", model=model, tokenizer=tokenizer)

blog_generator = load_blog_generator()

def full_pipeline(file):
    # 1. Load CSV
    df = pd.read_csv(file)
    
    # 2. Preprocess reviews (clean, tokenize, lemmatize, etc.)
    df = preprocess_pipeline(df)
    
    # 3. Predict sentiment using trained model
    X_tfidf = tfidf_vectorizer.transform(df['lemmatized_str'])
    review_length = np.array([len(x.split()) for x in df['lemmatized_str']]).reshape(-1, 1)
    X_features = hstack([X_tfidf, review_length])
    df['sentiment'] = sentiment_model.predict(X_features)

    # 4. Cluster products (returns df with 'cluster' and 'clustered_category' columns)
    df, kmeans, vectorizer, X, cluster_name_map = cluster_and_label_products(
        df,
        text_col='name_category_lemmatized',
        k_range=range(2, 8),
        random_state=42,
        show_samples=False,
        show_plot=False
    )
    
    # 5. Generate blogposts for each cluster/category
    blogposts = generate_blogposts_for_all_categories(df, generator=blog_generator)
    
    # 6. Return blogposts as a single string
    return "\n\n".join(blogposts)

iface = gr.Interface(
    fn=full_pipeline,
    inputs=gr.File(label="Upload your Amazon reviews CSV. Ensure it has at least the following columns: asins, name, categories, reviews.rating, reviews.title, reviews.text"),
    outputs=gr.Textbox(label="Generated Blogposts"),
    title="Amazon Product Review Blog Generator",
    description="Upload a CSV of product reviews. The app will preprocess, analyze sentiment, cluster products, and generate a blogpost for each category with top picks and the worst product."
)

if __name__ == "__main__":
    iface.launch(share=True)