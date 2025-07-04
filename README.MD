# Amazon Product Review Blog Generator

## Overview

This project is an end-to-end pipeline for analyzing Amazon product reviews, performing sentiment analysis, clustering products into meaningful categories, and generating human-like blog posts summarizing the best and worst products in each category. The pipeline leverages NLP techniques, machine learning models, and large language models (LLMs) such as Mistral-7B (for blog creation) and OpenAI GPT-3.5 Turbo to assign dynamically representative category names to product clusters. 

---

## Features

- **Data Preprocessing:** Cleans, tokenizes, removes stopwords, and lemmatizes review texts and product metadata.
- **Sentiment Analysis:** Trains and applies a sentiment classifier to label reviews as positive, neutral, or negative.
- **Product Clustering:** Groups products using TF-IDF vectorization and KMeans clustering based on product names and categories.
- **Category Naming:** Uses OpenAI GPT-3.5 Turbo to generate concise, human-readable names for each product cluster (categories).
- **Blog Generation:** Utilizes a quantized Mistral-7B model (via Hugging Face) to generate structured blog posts for each category, highlighting top products, common complaints, and products to avoid.
- **Web App Deployment:** Provides a user-friendly web interface (via Gradio and Flask) for uploading review data and viewing generated blog content.

---

## Project Structure

```
requirements.txt
data/
    Processed/
        combined_reviews.csv
    raw/
        1429_1.zip
deployment/
    app.py
    static/
    templates/
models/
    best_sentiment_model.pkl
    tfidf_vectorizer_sentiment.pkl
notebooks/
    Data_preprocessing.ipynb
    final.ipynb
    model1_sentiment_classifier.ipynb
    model2_clustering.ipynb
    model3_blog_generator.ipynb
raw code/
    blog_generation.py
    clustering.py
    data_preprocessing.py
    deployment dynamic.py
    sentiment_classifier.py
```

---

## Setup Instructions

### 1. Clone the Repository

```sh
git clone <repo-url>
cd <repo-directory>
```

### 2. Install Dependencies

Install all required Python packages:

```sh
pip install -r requirements.txt
```

**Key dependencies:**
- pandas, numpy, scikit-learn, nltk
- matplotlib, seaborn
- flask, gradio
- openai, transformers, huggingface_hub, bitsandbytes
- tqdm, dotenv, joblib

### 3. Download NLTK Data

The first run will download required NLTK corpora (punkt, stopwords, wordnet):

```python
import nltk
nltk.download('punkt')
nltk.download('punkt_tab')
nltk.download('stopwords')
nltk.download('wordnet')
```

### 4. Environment Variables

Create a `.env` file in the root directory with your API keys:

```
OPENAI_API_KEY=your-openai-api-key
HF_KEY=your-huggingface-api-key
```

---

## Usage

### A. Data Preprocessing

- Use [`notebooks/Data_preprocessing.ipynb`](notebooks/Data_preprocessing.ipynb) or [`raw code/data_preprocessing.py`](raw code/data_preprocessing.py) to clean and preprocess your Amazon reviews dataset.
- Output: Preprocessed data CSV file (e.g., `data/Processed/pre_processed_data.csv`). --> It is in zip format, because the size was too big to add it to the repo in Github. 

### B. Model Training

- **Sentiment Classifier:** Train and evaluate models in [`notebooks/model1_sentiment_classifier.ipynb`](notebooks/model1_sentiment_classifier.ipynb) or [`notebooks/final.ipynb`](notebooks/final.ipynb).
- Save the best model and vectorizer to `models/`.

### C. Product Clustering

- Cluster products using [`notebooks/model2_clustering.ipynb`](notebooks/model2_clustering.ipynb) or the pipeline in [`notebooks/final.ipynb`](notebooks/final.ipynb).
- Assign cluster/category names using OpenAI GPT-3.5 Turbo.

### D. Blog Generation

- Generate blog posts for each category using the Mistral-7B Instruct model as shown in [`notebooks/model3_blog_generator.ipynb`](notebooks/model3_blog_generator.ipynb) or [`raw code/blog_generation.py`](raw code/blog_generation.py).
- Save the quantized model and tokenizer locally in the `blog_generator/` directory. (We haven't included the saved Mistral 7B model in the repo, as it exceeded the maximum 100MB size from Github).

### E. Web App Deployment

#### 1. Gradio App

Run the dynamic pipeline with Gradio:

```sh
python "raw code/deployment dynamic.py"
```

- Upload a CSV of product reviews.
- The app will preprocess, analyze sentiment, cluster products, and generate blog posts for each category. It is dynamic, as no matter which dataset you upload, it will output blog entries for each of the product categories from the dataset. The only requirement is to ensure that the csv file uploaded contains at least the following columns: asins, name, categories, reviews.rating, reviews.title, reviews.text.

#### 2. Flask App

Run the Flask web app:

```sh
python deployment/app.py
```

- Access the app at `http://localhost:5000/`
- View blog posts generated by ChatGPT and Mistral. These results are static. In other words, we just displayed what we got in one iteration, using Mistral 7B Instruct model and another one generating the blog entries using GPT 3.5 Turbo model through the OPEN AI API. 

---

## File Descriptions

- **notebooks/**: Jupyter notebooks for each pipeline stage (preprocessing, sentiment, clustering, blog generation).
- **raw code/**: Modular Python scripts for each pipeline component.
- **deployment/app.py**: Flask web app for serving blog content.
- **deployment/templates/**: HTML templates for the web interface.
- **models/**: Saved ML models (sentiment classifier, vectorizer, blog generator).
- **data/**: Raw and processed datasets.

---

## Example Output

- **Blog posts** summarizing top products, common complaints, and warnings for each product category, ready for publication or integration into a website.

---

## Credits

- Built by Group 6 (Alejandro, Darius & Jef) for Ironhack Week 6 NLP Project.
- Uses [Mistral-7B-Instruct-v0.2](https://huggingface.co/mistralai/Mistral-7B-Instruct-v0.2) and OpenAI GPT-3.5 Turbo.

---

## License

This project is for educational purposes. See `LICENSE` for details.

---

## References

- [notebooks/final.ipynb](notebooks/final.ipynb)
- [raw code/deployment dynamic.py](raw code/deployment dynamic.py)
- [raw code/blog_generation.py](raw code/blog_generation.py)
- [deployment/app.py](deployment/app.py)
