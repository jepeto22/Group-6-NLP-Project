def generate_blogposts_for_all_categories(
    df_combined,
    generator,
    n_shot_prompt=None,
    n_top_products=3,
    n_complaints=3,
    max_new_tokens=800,
    temperature=0.4,
    top_p=0.9,
    repetition_penalty=1.1
):
    """
    For each unique category in df_combined['clustered_category'], generates a blogpost using the provided generator pipeline.
    Prints the blogpost for each category.
    """
    from nltk.corpus import stopwords
    from collections import Counter

    # Helper to extract top complaints
    def get_top_complaints(df, product_name, n=n_complaints):
        neg_reviews = df[(df['name'] == product_name) & (df['sentiment'] == 'negative')]['combined_reviews']
        words = ' '.join(neg_reviews).split()
        stop_words = set(stopwords.words('english'))
        filtered_words = [w.lower() for w in words if w.lower() not in stop_words and len(w) > 2]
        most_common = Counter(filtered_words).most_common(n)
        return [word for word, _ in most_common]

    # Default n-shot prompt if not provided
    if n_shot_prompt is None:
        n_shot_prompt = """
--- Blogpost for E-readers ---
(Example blog post here...)

--- Blogpost for Smart Tablets ---
(Example blog post here...)

--- Blogpost for Smart Home Essentials ---
(Example blog post here...)

--- Blogpost for Batteries ---
(Example blog post here...)
        """

    for category in df_combined['clustered_category'].unique():
        cat_df = df_combined[df_combined['clustered_category'] == category]
        if len(cat_df) < n_top_products:
            continue  # Skip small categories

        # Compute product scores
        product_scores = (
            cat_df.groupby('name')['sentiment_points']
            .agg(['mean', 'count'])
            .sort_values(by=['mean', 'count'], ascending=[False, False])
            .reset_index()
        )

        top_products = product_scores.head(n_top_products)
        worst_product = product_scores.tail(1)

        # Collect top complaints for top products
        complaints = {
            row['name']: get_top_complaints(cat_df, row['name'])
            for _, row in top_products.iterrows()
        }

        # Compose the specific prompt for the current category
        category_prompt = f"\n\n### Blogpost:\n\n"
        category_prompt += f"You are a product reviewer. Write a short, helpful blogpost for customers shopping for {category}.\n"
        category_prompt += "- The top 3 products are:\n"

        for i, row in top_products.iterrows():
            name = row['name']
            avg_rating = row['mean']
            review_count = int(row['count'])
            top_complaints = complaints[name]
            complaint_text = ', '.join(top_complaints) if top_complaints else 'Few complaints!'
            category_prompt += f"{i+1}. {name} (Avg. Rating: {avg_rating:.2f}, {review_count} reviews)\n"
            category_prompt += f"   Top complaints: {complaint_text}\n"

        # Include the worst product details
        worst_name = worst_product.iloc[0]['name']
        worst_rating = worst_product.iloc[0]['mean']
        category_prompt += f"\nThe worst product is {worst_name} (Avg. Rating: {worst_rating:.2f}).\n"
        category_prompt += "Explain why customers should avoid the worst product from the category, based on reviews.\n"
        category_prompt += "Write the blog entry in a friendly, informative tone.\n"

        # Final prompt
        prompt = n_shot_prompt + category_prompt

        # Generate blog post
        generated = generator(
            prompt,
            max_new_tokens=max_new_tokens,
            do_sample=True,
            temperature=temperature,
            top_p=top_p,
            repetition_penalty=repetition_penalty
        )[0]['generated_text']

        # Extract model output after prompt
        blog_output = generated.split("Write the blog entry in a friendly, informative tone.")[-1].strip()

        print(f"\n--- Blogpost for {category} ---\n")
        print(blog_output)

# Example usage
# clustered_data = pd.read_csv('path_to_your_clustered_data.csv')
# blog_content = generate_blog_post(clustered_data)
# save_blog_post(blog_content, 'outputs/blog_posts/recommendations_blog_post.md')