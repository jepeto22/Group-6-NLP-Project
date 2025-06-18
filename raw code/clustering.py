# model2_clustering.py

"""
Model 2 - Product Categories Clustering

This script provides functions to vectorize product text, find the optimal number of clusters, cluster products, visualize clusters, and assign category names using OpenAI GPT.
"""

import os
import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt
from kneed import KneeLocator
from dotenv import load_dotenv
import openai

load_dotenv()
openai_api_key = os.getenv("OPENAI_API_KEY")


def vectorize_text(df, text_col='name_category_lemmatized'):
    """
    Converts the text column of the dataframe into TF-IDF vectors.

    Parameters:
        df (DataFrame): The input data.
        text_col (str): Column containing preprocessed product text.

    Returns:
        X (sparse matrix): TF-IDF feature matrix.
        vectorizer (TfidfVectorizer): The fitted vectorizer object.
    """
    vectorizer = TfidfVectorizer(stop_words='english')
    X = vectorizer.fit_transform(df[text_col])
    return X, vectorizer

def find_optimal_k(X, k_range, random_state=42, plot=True):
    """
    Uses the elbow method and KneeLocator to determine the optimal number of clusters.

    Parameters:
        X (sparse matrix): Feature matrix from vectorization.
        k_range (range): Range of k values to test.
        random_state (int): Seed for reproducibility.
        plot (bool): Whether to show the elbow plot.

    Returns:
        optimal_k (int): Best number of clusters detected.
    """
    inertias = []
    for k in k_range:
        kmeans = KMeans(n_clusters=k, random_state=random_state, n_init=10)
        kmeans.fit(X)
        inertias.append(kmeans.inertia_)

    if plot:
        plt.figure(figsize=(8, 4))
        plt.plot(list(k_range), inertias, marker='o')
        plt.xlabel('Number of clusters (k)')
        plt.ylabel('Inertia')
        plt.title('Elbow Method For Optimal k')
        plt.show()

    kneedle = KneeLocator(list(k_range), inertias, curve='convex', direction='decreasing')
    optimal_k = kneedle.elbow - 1 if kneedle.elbow else k_range[0]
    print(f"Optimal number of clusters (automatic): {optimal_k}")
    return optimal_k

def apply_kmeans(X, n_clusters, random_state=42):
    """
    Applies KMeans clustering to the TF-IDF matrix.

    Parameters:
        X (sparse matrix): TF-IDF matrix.
        n_clusters (int): Number of clusters to use.
        random_state (int): Random seed.

    Returns:
        kmeans (KMeans): Fitted KMeans object.
        labels (ndarray): Cluster labels for each data point.
    """
    kmeans = KMeans(n_clusters=n_clusters, random_state=random_state, n_init=20)
    labels = kmeans.fit_predict(X)
    return kmeans, labels

def plot_clusters(X, labels, random_state=42):
    """
    Reduces high-dimensional data to 2D using PCA and visualizes clusters.

    Parameters:
        X (sparse matrix or ndarray): Feature matrix.
        labels (ndarray): Cluster assignments.
        random_state (int): PCA seed.
    """
    pca = PCA(n_components=2, random_state=random_state)
    X_pca = pca.fit_transform(X.toarray() if hasattr(X, "toarray") else X)
    plt.figure(figsize=(10, 6))
    for cluster_id in np.unique(labels):
        plt.scatter(
            X_pca[labels == cluster_id, 0],
            X_pca[labels == cluster_id, 1],
            label=f'Cluster {cluster_id}', alpha=0.5
        )
    plt.title('K-Means Clusters Visualization (PCA)')
    plt.xlabel('PCA Component 1')
    plt.ylabel('PCA Component 2')
    plt.legend()
    plt.show()

def display_sample_products(df, cluster_col='cluster'):
    """
    Displays a few sample products from each cluster.

    Parameters:
        df (DataFrame): Data with assigned clusters.
        cluster_col (str): Column name where cluster IDs are stored.
    """
    for cluster_id in sorted(df[cluster_col].unique()):
        print(f"\nCluster {cluster_id} sample products:")
        print(df[df[cluster_col] == cluster_id][['name', 'categories']].head(10))

def get_category_name_openai(product_names, used_names, openai_api_key, model="gpt-3.5-turbo"):
    """
    Calls OpenAI API to generate a unique, concise category name based on product names.

    Parameters:
        product_names (list): Product names within a cluster.
        used_names (set): Set of already-used category names to avoid duplicates.
        openai_api_key (str): Your OpenAI API key.
        model (str): Model to use (default: gpt-3.5-turbo).

    Returns:
        str: A unique category name.
    """

    prompt = (
        "Given the following list of product names, suggest a concise, precise category name that best describes the majority of products and is not generic. "
        f"Do NOT use any of these words: {', '.join(used_names)}\n"
        "Only return the category name, nothing else.\n\n"
        "Product names:\n" + "\n".join(product_names)
    )
    client = openai.OpenAI(api_key=openai_api_key)
    response = client.chat.completions.create(
        model=model,
        messages=[{"role": "user", "content": prompt}],
        temperature=0.5,
        max_tokens=15,
    )
    return response.choices[0].message.content.strip()

def assign_cluster_names(df, cluster_col='cluster', name_col='name', openai_api_key=None):
    """
    Assigns a descriptive category name to each cluster using OpenAI.

    Parameters:
        df (DataFrame): Data containing clusters.
        cluster_col (str): Name of the cluster column.
        name_col (str): Column to extract product names from.
        openai_api_key (str): OpenAI API key.

    Returns:
        df (DataFrame): Data with an added 'clustered_category' column.
        cluster_name_map (dict): Mapping of cluster ID to category name.
    """
    cluster_name_map = {}
    used_names = set()
    for cluster_id in sorted(df[cluster_col].unique()):
        product_names = df[df[cluster_col] == cluster_id][name_col].dropna().unique().tolist()
        if product_names:
            category_name = get_category_name_openai(product_names, used_names, openai_api_key)
            while category_name in used_names:
                category_name = get_category_name_openai(product_names, used_names, openai_api_key)
            used_names.add(category_name)
            cluster_name_map[cluster_id] = category_name
    df['clustered_category'] = df[cluster_col].map(cluster_name_map)
    return df, cluster_name_map

def cluster_and_label_products(
    df,
    text_col='name_category_lemmatized',
    k_range=range(2, 11),
    random_state=42,
    show_samples=True,
    show_plot=True
):
    """
    Full pipeline: vectorizes product text, clusters it, assigns category names via OpenAI.

    Parameters:
        df (DataFrame): Input dataframe.
        text_col (str): Column with product text.
        k_range (range): Range of cluster numbers to test.
        random_state (int): Random seed.
        show_samples (bool): Print sample products per cluster.
        show_plot (bool): Show elbow and PCA plots.

    Returns:
        df (DataFrame): Updated with clusters and categories.
        kmeans (KMeans): Fitted clustering model.
        vectorizer (TfidfVectorizer): Fitted vectorizer.
        X (sparse matrix): TF-IDF feature matrix.
        cluster_name_map (dict): Mapping of cluster IDs to category names.
    """
    # Step 1: Text vectorization
    X, vectorizer = vectorize_text(df, text_col)

    # Step 2: Find optimal number of clusters
    optimal_k = find_optimal_k(X, k_range, random_state, plot=show_plot)

    # Step 3: Cluster the data
    kmeans, labels = apply_kmeans(X, optimal_k, random_state)
    df['cluster'] = labels

    # Step 4: Visualize results
    if show_plot:
        plot_clusters(X, labels, random_state)
    if show_samples:
        display_sample_products(df, cluster_col='cluster')

    # Step 5: Name the clusters using OpenAI
    df, cluster_name_map = assign_cluster_names(df, openai_api_key=openai_api_key)
    print(cluster_name_map)
    return df, kmeans, vectorizer, X, cluster_name_map