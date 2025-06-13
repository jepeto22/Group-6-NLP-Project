def train_and_evaluate_svm_sentiment(
    df,
    text_col='lemmatized_str',
    label_col='sentiment',
    ngram_range=(1,2),
    min_df=3,
    max_df=0.9,
    C_grid=[0.01, 0.1, 1, 10],
    random_state=42
):
    """
    Trains and evaluates a LinearSVC sentiment classifier with TF-IDF (with ngrams) and review length as features.
    Returns: best_model, tfidf_vectorizer, X_test_features, y_test, misclassified_df
    """
    from sklearn.model_selection import train_test_split, GridSearchCV, cross_val_score
    from sklearn.pipeline import Pipeline
    from sklearn.svm import LinearSVC
    from sklearn.feature_extraction.text import TfidfVectorizer
    from scipy.sparse import hstack
    from sklearn.metrics import classification_report, confusion_matrix
    import numpy as np
    import pandas as pd

    # 1. Split
    X_text = df[text_col]
    y = df[label_col]
    X_train_text, X_test_text, y_train, y_test = train_test_split(
        X_text, y, test_size=0.2, random_state=random_state, stratify=y
    )

    # 2. Fit vectorizer only on training data
    tfidf_vectorizer = TfidfVectorizer(ngram_range=ngram_range, min_df=min_df, max_df=max_df)
    X_train_tfidf = tfidf_vectorizer.fit_transform(X_train_text)
    X_test_tfidf = tfidf_vectorizer.transform(X_test_text)

    # 3. Add review length as a feature
    train_length = np.array([len(x.split()) for x in X_train_text]).reshape(-1, 1)
    test_length = np.array([len(x.split()) for x in X_test_text]).reshape(-1, 1)
    X_train_features = hstack([X_train_tfidf, train_length])
    X_test_features = hstack([X_test_tfidf, test_length])

    # 4. Hyperparameter tuning with GridSearchCV
    pipeline = Pipeline([
        ('clf', LinearSVC())
    ])
    param_grid = {'clf__C': C_grid}
    grid = GridSearchCV(pipeline, param_grid, cv=3, scoring='f1_macro', n_jobs=-1)
    grid.fit(X_train_features, y_train)
    print("Best SVM parameters:", grid.best_params_)
    print("Best SVM F1-score (CV on train):", grid.best_score_)

    # 5. Cross-validation on training set
    svc = LinearSVC(C=grid.best_params_['clf__C'], max_iter=1000, random_state=random_state)
    scores = cross_val_score(svc, X_train_features, y_train, cv=5, scoring='f1_macro')
    print("LinearSVC 5-fold CV F1-macro (train):", scores.mean())
    print("All F1-macro scores (train):", scores)

    # 6. Final evaluation on test set
    best_model = grid.best_estimator_
    y_pred = best_model.predict(X_test_features)
    print(classification_report(y_test, y_pred))
    print(confusion_matrix(y_test, y_pred))

    # Find indices of misclassified samples
    mis_idx = np.where(y_pred != y_test)[0]
    y_test_reset = y_test.reset_index(drop=True)
    misclassified_df = pd.DataFrame({
        'actual_sentiment': y_test_reset.iloc[mis_idx].values,
        'predicted_sentiment': y_pred[mis_idx],
        'combined_reviews': X_test_text.iloc[mis_idx].values
    })
    print(misclassified_df.head())

    return best_model, tfidf_vectorizer, X_test_features, y_test, misclassified_df