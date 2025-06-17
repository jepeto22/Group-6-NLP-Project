# model3_blog_generator.py

"""
Model 3 - Blog Entry Generation for Each Category

This script generates blog-like summaries that help users choose among products.
It takes clustered product data and creates human-readable reviews using a language model.
"""

import os
from dotenv import load_dotenv
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM, pipeline, BitsAndBytesConfig
from huggingface_hub import login

# Authenticate with Hugging Face
load_dotenv()
hf_key = os.getenv("HF_KEY")
login(token=hf_key, new_session=False)

# Model loading and quantization config
model_path = "mistralai/Mistral-7B-Instruct-v0.2"
bnb_config = BitsAndBytesConfig(
    load_in_8bit=True,
    bnb_8bit_quant_type="nf8",
    bnb_8bit_compute_dtype=torch.float16,
    bnb_8bit_use_double_quant=False,
)

# Load model and tokenizer with memory-efficient 8-bit settings
blog_model = AutoModelForCausalLM.from_pretrained(
    model_path,
    quantization_config=bnb_config,
    device_map="auto",
    use_auth_token=hf_key
)
blog_tokenizer = AutoTokenizer.from_pretrained(model_path, use_auth_token=hf_key)

# Save model and tokenizer locally for deployment
blog_model.save_pretrained('blog_generator')
blog_tokenizer.save_pretrained('blog_generator')

def generate_blogposts_for_all_categories(
    df,
    generator,
    n_shot_prompt=None,
    n_top_products=3,
    n_complaints=2,
    max_new_tokens=800,
    temperature=0.4,
    top_p=0.9,
    repetition_penalty=1.1
):
    """
    Generates blog-style posts for each product category using a text generation model.
    """
    from nltk.corpus import stopwords
    from collections import Counter

    # Map sentiment labels to numeric values
    sentiment_map = {'positive': 2, 'neutral': 1, 'negative': 0}
    df['sentiment_points'] = df['sentiment'].map(sentiment_map)
    blogposts = []

    def get_top_complaints(df, product_name, n=n_complaints):
        neg_reviews = df[(df['name'] == product_name) & (df['sentiment'] == 'negative')]['combined_reviews']
        words = ' '.join(neg_reviews).split()
        stop_words = set(stopwords.words('english'))
        filtered_words = [w.lower() for w in words if w.lower() not in stop_words and len(w) > 2]
        most_common = Counter(filtered_words).most_common(n)
        return [word for word, _ in most_common]

    # Example prompt for few-shot learning
    n_shot_prompt = n_shot_prompt or """
# Context:
You are a helpful assistant that writes concise, well-structured blogposts comparing consumer tech products. The user provides product review summaries, and you respond with a styled blogpost including top picks, key complaints, and a warning about the worst-rated product.

## Example 1

<|system|>
You are a product review blogger assistant.

<|user|>
Write a blogpost based on review data for e-readers:
1. Kindle Paperwhite (Avg. Rating: 4.8, 1200 reviews)
   Complaints: screen glare, slow refresh
2. Kobo Clara HD (Avg. Rating: 4.7, 950 reviews)
   Complaints: limited store, battery life
3. Kindle Oasis (Avg. Rating: 4.6, 800 reviews)
   Complaints: price, weight
Worst: Nook GlowLight 3 (Avg. Rating: 3.2), Complaints: slow performance, software glitches

<|assistant|>
--- Blogpost for E-readers ---
Looking for the perfect e-reader? Here are our top picks!

1. **Kindle Paperwhite** (Avg. Rating: 4.8, 1200 reviews)  
   *Top complaints:* screen glare, slow refresh

2. **Kobo Clara HD** (Avg. Rating: 4.7, 950 reviews)  
   *Top complaints:* limited store, battery life

3. **Kindle Oasis** (Avg. Rating: 4.6, 800 reviews)  
   *Top complaints:* price, weight

The **Kindle Paperwhite** stands out for its crisp display and waterproof design, making it ideal for reading anywhere. **Kobo Clara HD** is a great alternative with support for multiple formats, though some users wish for a larger store. The **Kindle Oasis** offers premium features, but its higher price and weight are noted by some reviewers.

**Worst product:** Nook GlowLight 3 (Avg. Rating: 3.2)  
Many customers report slow performance and frequent software glitches. Unless you’re a die-hard Nook fan, we recommend considering other options for a smoother reading experience.

---

## Example 2

<|user|>
Write a blogpost using this tablet review data:
1. iPad Air (4.9, 2100) - complaints: price, limited ports
2. Galaxy Tab S7 (4.8, 1800) - complaints: bloatware, charger speed
3. Fire HD 10 (4.5, 1600) - complaints: app selection, ads
Worst: Lenovo Tab M8 (3.4) - complaints: sluggish, weak display

<|assistant|>
--- Blogpost for Smart Tablets ---
Shopping for a smart tablet? Here are the top choices this year:

1. **Apple iPad Air** (Avg. Rating: 4.9, 2100 reviews)  
   *Top complaints:* price, limited ports

2. **Samsung Galaxy Tab S7** (Avg. Rating: 4.8, 1800 reviews)  
   *Top complaints:* bloatware, charger speed

3. **Amazon Fire HD 10** (Avg. Rating: 4.5, 1600 reviews)  
   *Top complaints:* app selection, ads

The **iPad Air** is praised for its speed and display quality, though it comes at a premium. The **Galaxy Tab S7** offers excellent multitasking, but some users dislike the pre-installed apps. The **Fire HD 10** is a budget-friendly option, but its app ecosystem is more limited.

**Worst product:** Lenovo Tab M8 (Avg. Rating: 3.4)  
Users mention sluggish performance and a lackluster screen. For a better experience, consider one of the top-rated tablets above.

---
"""

    for category in df['clustered_category'].unique():
        cat_df = df[df['clustered_category'] == category]
        unique_products = cat_df['name'].nunique()
        if unique_products == 0:
            continue

        # Compute product scores for ranking (by sentiment_points)
        product_scores = (
            cat_df.groupby('name')['sentiment_points']
            .agg(['mean', 'count'])
            .sort_values(by=['mean', 'count'], ascending=[False, False])
            .reset_index()
        )

        # For display: get average rating from 'reviews.rating'
        avg_ratings = cat_df.groupby('name')['reviews.rating'].mean()

        # Determine how many to show as top and which is worst
        if unique_products == 1:
            top_n = 1
            show_worst = False
        elif unique_products == 2:
            top_n = 1
            show_worst = True
        elif unique_products == 3:
            top_n = 2
            show_worst = True
        else:
            top_n = n_top_products
            show_worst = True

        top_products = product_scores.head(top_n)
        if show_worst:
            worst_product = product_scores.tail(1)
        else:
            worst_product = None

        # Collect top complaints for top products
        complaints = {
            row['name']: get_top_complaints(cat_df, row['name'])
            for _, row in top_products.iterrows()
        }

        # Compose the prompt
        category_prompt = f"\n\n### Blogpost:\n\n"
        category_prompt += f"You are a product reviewer. Write a short, helpful blogpost for customers shopping for {category}.\n"
        category_prompt += f"- The top {top_n} product{'s are' if top_n > 1 else ' is'}:\n"

        for i, row in top_products.iterrows():
            name = row['name']
            avg_rating = avg_ratings[name]
            review_count = int(row['count'])
            top_complaints = complaints[name]
            complaint_text = ', '.join(top_complaints) if top_complaints else 'Few complaints!'
            category_prompt += f"{i+1}. {name} (Avg. Rating: {avg_rating:.2f}, {review_count} reviews)\n"
            category_prompt += f"   Top complaints: {complaint_text}\n"

        # Include the worst product details if applicable
        if show_worst and worst_product is not None:
            worst_name = worst_product.iloc[0]['name']
            worst_avg_rating = avg_ratings[worst_name]
            category_prompt += f"\nThe worst product is {worst_name} (Avg. Rating: {worst_avg_rating:.2f}).\n"
            category_prompt += "Explain why customers should avoid the worst product from the category, based on reviews.\n"

        category_prompt += "Write the blog entry in a friendly, informative tone.\n"

        prompt = n_shot_prompt + category_prompt
        generated = generator(
            prompt,
            max_new_tokens=max_new_tokens,
            do_sample=True,
            temperature=temperature,
            top_p=top_p,
            repetition_penalty=repetition_penalty
        )[0]['generated_text']
        blog_output = generated.split("Write the blog entry in a friendly, informative tone.")[-1].strip()
        blogposts.append(f"--- Blogpost for {category} ---\n\n{blog_output}")

    return blogposts
