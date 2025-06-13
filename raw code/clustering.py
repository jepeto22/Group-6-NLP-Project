def cluster_and_visualize_products(
    df,
    text_col='name_category_lemmatized',
    n_clusters=4,
    random_state=42,
    show_samples=True,
    show_plot=True
):
    """
    Cluster products using TF-IDF on a text column and visualize with PCA.
    Adds a 'cluster' column to the DataFrame.
    Returns: df (with cluster column), kmeans model, vectorizer, X (tfidf matrix)
    """
    import pandas as pd
    from sklearn.feature_extraction.text import TfidfVectorizer
    from sklearn.cluster import KMeans
    from sklearn.decomposition import PCA
    import matplotlib.pyplot as plt

    # Vectorize
    vectorizer = TfidfVectorizer(stop_words='english')
    X = vectorizer.fit_transform(df[text_col])
    print("TF-IDF matrix shape:", X.shape)

    # KMeans clustering
    kmeans = KMeans(n_clusters=n_clusters, random_state=random_state)
    df['cluster'] = kmeans.fit_predict(X)
    print("Number of products per cluster:")
    print(df['cluster'].value_counts())

    # Show sample products from each cluster
    if show_samples:
        for cluster_id in range(n_clusters):
            print(f"\nCluster {cluster_id} sample products:")
            display(df[df['cluster'] == cluster_id][['name', 'categories']].head(10))

    # Visualize clusters using PCA
    if show_plot:
        pca = PCA(n_components=2)
        X_pca = pca.fit_transform(X.toarray())
        plt.figure(figsize=(10, 6))
        for cluster_id in range(n_clusters):
            plt.scatter(
                X_pca[df['cluster'] == cluster_id, 0],
                X_pca[df['cluster'] == cluster_id, 1],
                label=f'Cluster {cluster_id}', alpha=0.5
            )
        plt.title('K-Means Clusters Visualization (PCA)')
        plt.xlabel('PCA Component 1')
        plt.ylabel('PCA Component 2')
        plt.legend()
        plt.show()
    
    cluster_name_map = {
    0: "Smart Home Essentials",
    1: "Batteries",
    2: "Smart Tablets",
    3: "E-readers"
    }
    df['clustered_category'] = df['cluster'].map(cluster_name_map)
    
    return df, kmeans, vectorizer, X

