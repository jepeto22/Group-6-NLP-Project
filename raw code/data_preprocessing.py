# data_preprocessing.py

"""
Data Preprocessing Pipeline for Amazon Product Reviews

This script provides functions to clean, preprocess, tokenize, lemmatize, and label sentiment for Amazon product reviews.
"""


def drop_duplicates_empty(df):
    """
    Cleans a DataFrame of Amazon product reviews by:
      - Removing rows where either 'reviews.text' or 'reviews.rating' are missing (NaN).
      - Removing duplicate reviews for the same product, keeping only the first occurrence of each unique combination of 'asins' and 'reviews.text'.
    """
    df = df.dropna(subset=['reviews.text', 'reviews.rating'])
    df = df.drop_duplicates(subset=['asins', 'reviews.text'])
    return df

def clean_text(text):
    """
    Cleans and standardizes input text by:
      - Removing all non-alphanumeric characters (except spaces).
      - Replacing multiple spaces with a single space.
      - Stripping leading/trailing whitespace and converting text to lowercase.
    """
    text = re.sub(r"[^a-zA-Z0-9\s]", '', str(text))
    text = re.sub(r'\s+', ' ', text)
    return text.strip().lower()

def clean_name(name):
    """
    Cleans the product name by:
      - Removing the phrase 'Includes Special Offers'
      - Replacing multiple spaces with a single space
      - Stripping leading/trailing whitespace
    """
    name = re.sub(r'Includes Special Offers', '', name)
    name = re.sub(r'\s+', ' ', name)
    return name.strip()

def preprocess_reviews(df):
    """
    Preprocesses Amazon review data by:
      - Combining 'reviews.title' and 'reviews.text' into a single 'combined_reviews' column.
      - Cleaning the combined text using the clean_text function.
      - Dropping the original 'reviews.title' and 'reviews.text' columns.
      - Removing rows with missing values in 'combined_reviews' or 'reviews.rating'.
    """
    df['combined_reviews'] = df['reviews.title'].fillna('') + ' ' + df['reviews.text'].fillna('')
    df['combined_reviews'] = df['combined_reviews'].apply(clean_text)
    df = df.drop(['reviews.title', 'reviews.text'], axis=1)
    df = df.dropna(subset=['combined_reviews', 'reviews.rating'])
    return df

def tokenize_reviews(df, col='combined_reviews', new_col='tokens'):
    """
    Tokenizes the text in a specified column of a DataFrame using NLTK's word_tokenize.
    """
    df[new_col] = df[col].apply(word_tokenize)
    return df

def remove_stopwords(df, tokens_col='tokens', new_col='tokens_nostop', language='english'):
    """
    Removes stopwords from tokenized text in a specified column of a DataFrame.
    """
    stop_words = set(stopwords.words(language))
    df[new_col] = df[tokens_col].apply(lambda tokens: [w for w in tokens if w.lower() not in stop_words])
    return df

def lemmatize_tokens(df, tokens_col='tokens_nostop', new_col='tokens_lemmatized'):
    """
    Lemmatizes tokens in a specified column of a DataFrame using NLTK's WordNetLemmatizer.
    """
    lemmatizer = WordNetLemmatizer()
    df[new_col] = df[tokens_col].apply(lambda tokens: [lemmatizer.lemmatize(token) for token in tokens])
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
    df['name'] = df['name'].fillna('')
    df['categories'] = df['categories'].fillna('')
    df['name_category'] = df['name'] + ' ' + df['categories']
    df['name_category'] = df['name_category'].apply(clean_name)
    df['name_category_tokens'] = df['name_category'].apply(word_tokenize)
    
    stop_words = set(stopwords.words('english'))
    df['name_category_tokens_nostop'] = df['name_category_tokens'].apply(
        lambda tokens: [token for token in tokens if token.lower() not in stop_words]
    )
    
    lemmatizer = WordNetLemmatizer()
    df['name_category_tokens_lemmatized'] = df['name_category_tokens_nostop'].apply(
        lambda tokens: [lemmatizer.lemmatize(token) for token in tokens]
    )
    
    df['name_category_lemmatized'] = df['name_category_tokens_lemmatized'].apply(lambda tokens: ' '.join(tokens))
    
    return df

def add_sentiment_column(df):
    """
    Adds a 'sentiment' column to the DataFrame based on the 'reviews.rating' value:
      - Ratings >= 4 are labeled as 'positive'
      - Ratings == 3 are labeled as 'neutral'
      - Ratings < 3 are labeled as 'negative'
    """
    def categorize_rating(rating):
        if rating >= 4:
            return 'positive'
        elif rating == 3:
            return 'neutral'
        else:
            return 'negative'
    df['sentiment'] = df['reviews.rating'].apply(categorize_rating)
    return df

def preprocess_pipeline(df):
    df = drop_duplicates_empty(df)
    df = preprocess_reviews(df)
    df = add_sentiment_column(df)
    df = tokenize_reviews(df)
    df = remove_stopwords(df)
    df = lemmatize_tokens(df)
    df['lemmatized_str'] = df['tokens_lemmatized'].apply(lambda tokens: ' '.join(tokens))
    df = preprocess_and_lemmatize_names_categories(df)
    return df