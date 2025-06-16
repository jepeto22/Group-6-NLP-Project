# Amazon Product Review Classification & Recommendation

## Project Aim

This project analyzes Amazon product reviews to:
- **Classify review sentiment** (positive, neutral, negative)
- **Cluster products** into meaningful meta-categories using text features
- **Generate blog-style product recommendations** for each category, highlighting the best and worst products based on customer feedback

The final output is a set of consumer-friendly blog posts that help users make informed purchase decisions.

---

## Project Structure

```
.
├── README.md
├── requirements.txt
├── data/
│   ├── raw/                # Original Amazon review data
│   └── Processed/          # Cleaned and combined review data
├── deployment/
│   ├── app.py              # Flask web app for model deployment
│   ├── static/             # Static files for the web app
│   └── templates/          # HTML templates for the web app
├── models/
│   ├── linear_svc_model_sentiment.pkl         # Trained sentiment classifier
│   └── tfidf_vectorizer_sentiment.pkl         # TF-IDF vectorizer for sentiment
├── notebooks/
│   ├── utils.ipynb         # Data preprocessing and utility functions
│   ├── model1_sentiment_classifier.ipynb  # Sentiment model training/evaluation
│   ├── model2_clustering.ipynb             # Product clustering with KMeans
│   ├── model2.5_cluster_names_GPT_3.5.ipynb # Cluster naming with GPT-3.5
│   ├── model3_blog_generator.ipynb         # Blog post generation
│   └── final.ipynb         # Full pipeline: data prep, modeling, blog generation
├── raw code/
│   ├── blog_generation.py  # (Legacy) Blog post generation script
│   ├── clustering.py       # (Legacy) Clustering script
│   ├── data_preprocessing.py # (Legacy) Data cleaning functions
│   └── deployment - Copy.py # (Legacy) Gradio deployment script
└── RoboReviews Group 6 Presentation.pdf      # Project presentation
```

---

## File & Folder Descriptions

### **data/**
- **raw/**: Contains the original, unprocessed Amazon review datasets.
- **Processed/**: Contains cleaned and combined review data (e.g., `combined_reviews.zip`).

### **deployment/**
- **app.py**: Flask web application for serving blog post recommendations. Main entry point for deployment.
- **static/**: Static assets (CSS, JS, images) for the web interface.
- **templates/**: HTML templates for rendering web pages.

### **models/**
- **linear_svc_model_sentiment.pkl**: Trained SVM model for sentiment classification.
- **tfidf_vectorizer_sentiment.pkl**: TF-IDF vectorizer fitted on review text for sentiment analysis.

### **notebooks/**
- **utils.ipynb**: Functions for text cleaning, tokenization, lemmatization, and pipeline assembly.
- **model1_sentiment_classifier.ipynb**: Trains and evaluates sentiment classifiers (SVM, Logistic Regression, etc.).
- **model2_clustering.ipynb**: Clusters products using KMeans and visualizes clusters.
- **model2.5_cluster_names_GPT_3.5.ipynb**: Uses OpenAI GPT-3.5 to generate human-readable cluster/category names.
- **model3_blog_generator.ipynb**: Generates blog-style product recommendations using a language model.
- **final.ipynb**: Runs the full pipeline: data prep, sentiment modeling, clustering, cluster naming, and blog generation.

### **raw code/**
- **blog_generation.py, clustering.py, data_preprocessing.py**: Early scripts for blog generation, clustering, and data cleaning (now superseded by notebook code).
- **deployment - Copy.py**: Prototype Gradio app for interactive blog post generation.

### **requirements.txt**
- Python dependencies for running the project and web app.

### **RoboReviews Group 6 Presentation.pdf**
- Project summary and presentation slides.

---

## How to Run the Project

### **1. Install Dependencies**

```bash
pip install -r requirements.txt
```

### **2. Prepare Data**

- Place your raw review files in `data/raw/`.
- Run the preprocessing pipeline in [`notebooks/utils.ipynb`](notebooks/utils.ipynb) or use the provided scripts to generate `data/Processed/combined_reviews.zip`.

### **3. Train Models (Optional)**

- Use the notebooks in [`notebooks/`](notebooks/) to train or retrain the sentiment classifier and clustering models.
- Pre-trained models are provided in [`models/`](models/).

### **4. Run the Web App**

From the project root, start the Flask app:

```bash
cd deployment
python app.py
```

- The app will be available at [http://localhost:5000](http://localhost:5000).
- Visit `/chatgpt_blog` or `/mistral_blog` routes to see generated blog posts for each product category.

---

## Example Usage

- **Fine-tune sentiment model:** See [`notebooks/model1_sentiment_classifier.ipynb`](notebooks/model1_sentiment_classifier.ipynb)
- **Cluster products:** See [`notebooks/model2_clustering.ipynb`](notebooks/model2_clustering.ipynb)
- **Generate blog posts:** See [`notebooks/model3_blog_generator.ipynb`](notebooks/model3_blog_generator.ipynb)

---

## Notes

- For OpenAI or Hugging Face API access, set your API keys in environment variables or `.env` files as shown in the notebooks.
- The `raw code/` folder contains legacy scripts; use the `notebooks/` and `deployment/` folders for the latest pipeline.

---

## Authors

Group 6 - Ironhack Data Science Bootcamp

---