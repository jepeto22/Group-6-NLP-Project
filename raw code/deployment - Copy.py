import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))


import gradio as gr
import nltk
import re
import string
import numpy as np
from scipy.sparse import hstack
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.stem import PorterStemmer, WordNetLemmatizer
from nltk import pos_tag
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
import pandas as pd
import nltk
nltk.download('punkt')  # 🔑 Required for sentence and word tokenization
nltk.download('stopwords')
nltk.download('wordnet')
from nltk.tokenize import word_tokenize, sent_tokenize
import joblib
from src.data_preprocessing import clean_text, preprocess_pipeline, preprocess_and_lemmatize_names_categories
from src.clustering import cluster_and_visualize_products
from src.blog_generation import generate_blogposts_for_all_categories
from transformers import AutoTokenizer, AutoModelForCausalLM, pipeline

# Load pre-trained sentiment model and vectorizer
sentiment_model = joblib.load(os.path.join(
    '/content/drive/Othercomputers', 
    'My laptop', 
    'Escritorio', 
    'Ironhack', 
    'Ironhack candela', 
    'Week 6', 
    'Project', 
    'product-recommendation-finetune', 
    'outputs', 
    'linear_svc_model_sentiment.pkl'))
tfidf_vectorizer = joblib.load(os.path.join(
    '/content/drive/Othercomputers', 
    'My laptop', 
    'Escritorio', 
    'Ironhack', 
    'Ironhack candela', 
    'Week 6', 
    'Project', 
    'product-recommendation-finetune', 
    'outputs', 
    'tfidf_vectorizer_sentiment.pkl'))


# Load pre-trained blog generator
blog_model = AutoModelForCausalLM.from_pretrained(os.path.join(
    '/content/drive/Othercomputers', 
    'My laptop', 
    'Escritorio', 
    'Ironhack', 
    'Ironhack candela', 
    'Week 6', 
    'Project', 
    'product-recommendation-finetune', 
    'outputs', 
    'blog_generator'), trust_remote_code=True)
blog_tokenizer = AutoTokenizer.from_pretrained(os.path.join('/content/drive/Othercomputers', 
    'My laptop', 
    'Escritorio', 
    'Ironhack', 
    'Ironhack candela', 
    'Week 6', 
    'Project', 
    'product-recommendation-finetune', 
    'outputs', 
    'blog_generator'))
blog_generator = pipeline("text-generation", model=blog_model, tokenizer=blog_tokenizer)

def process_csv_and_generate_blog(file):
    # 1. Load CSV
    df = pd.read_csv(file.name)
    
    # 2. Clean and preprocess review text
    
    df = preprocess_pipeline(df)
    
    # 3. Predict sentiment
    X_tfidf = tfidf_vectorizer.transform(df['lemmatized_str'])
    review_length = np.array([len(x.split()) for x in df['lemmatized_str']]).reshape(-1, 1)
    X_vec = hstack([X_tfidf, review_length])
    df['sentiment'] = sentiment_model.predict(X_vec)
    sentiment_map = {'positive': 2, 'neutral': 1, 'negative': 0}
    df['sentiment_points'] = df['sentiment'].map(sentiment_map)
    
    # 4. Preprocess names/categories for clustering
    df = preprocess_and_lemmatize_names_categories(df)
    
    # 5. Cluster products
    df, _, _, _ = cluster_and_visualize_products(df, show_samples=False, show_plot=False)
    
    # 6. Generate blog posts for each cluster/category
    import io
    import sys
    output = io.StringIO()
    sys.stdout = output  # Capture print output from blog generation
    generate_blogposts_for_all_categories(df, generator=blog_generator)
    sys.stdout = sys.__stdout__
    blogposts = output.getvalue()
    
    return blogposts

iface = gr.Interface(
    fn=process_csv_and_generate_blog,
    inputs=gr.File(label="Upload your product reviews CSV"),
    outputs=gr.Textbox(label="Generated Blog Posts"),
    title="Product Review Blog Generator",
    description="Upload a CSV of product reviews and get blog posts for each product category!"
)

iface.launch(share=True)