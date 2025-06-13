# Clasificación y Recomendación de Productos

Este proyecto clasifica reseñas de productos en categorías de sentimiento y agrupa los productos en meta-categorías, generando además contenido estilo blog para recomendar productos.

## Estructura de notebooks

- `main.ipynb`: Ejecución general del flujo completo.
- `utils.ipynb`: Funciones para procesar texto y preparar los datos.
- `model1_sentiment_classifier.ipynb`: Modelo de sentimiento con SVM.
- `model2_clustering.ipynb`: Agrupamiento con KMeans.
- `model3_blog_generator.ipynb`: Generación de artículos para consumidores.

## Estructura de datos

- `data/raw/`: Archivos de datos originales (reseñas, categorías, etc.).
- `data/Processed/`: Archivos procesados y limpios listos para análisis y entrenamiento.
- `data/README.md`: Documentación sobre la estructura y el preprocesamiento de los datos.

## Cómo ejecutar

1. Ejecuta `utils.ipynb` para preparar los datos.
2. Entrena y evalúa el modelo de sentimiento en `model1_sentiment_classifier.ipynb`.
3. Agrupa los productos por similitud en `model2_clustering.ipynb`.
4. Genera artículos informativos con `model3_blog_generator.ipynb`.
5. O usa `main.ipynb` para correr todo el proceso seguido.

## Requisitos

Instala dependencias:

```bash
pip install -r requirements.txt
```

---

# Product Recommendation Fine-Tuning Project

This project fine-tunes a language model to select the top 3 products from each category based on the average of positive, neutral, and negative reviews. It also generates a blog post recommending the best products, their features, downsides, and mentions the product with the worst reviews and reasons not to purchase it.

## Project Structure

- **data/**: Raw and processed data files.
  - **raw/**: Original product reviews and category data.
  - **Processed/**: Cleaned and transformed data for analysis and model training.
  - **README.md**: Data documentation and preprocessing steps.

- **notebooks/**: Jupyter notebooks for EDA and model fine-tuning.
  - **main.ipynb**: Full pipeline execution.
  - **utils.ipynb**: Text processing and data preparation functions.
  - **model1_sentiment_classifier.ipynb**: Sentiment classification with SVM.
  - **model2_clustering.ipynb**: Product clustering with KMeans.
  - **model3_blog_generator.ipynb**: Blog post generation.

- **src/**: Source code for data processing, model fine-tuning, and blog generation.
  - **__init__.py**: Package marker.
  - **data_preprocessing.py**: Data cleaning and preprocessing functions.
  - **clustering.py**: Clustering algorithms for product grouping.
  - **review_aggregation.py**: Review aggregation and average calculations.
  - **model_finetune.py**: Language model fine-tuning.
  - **blog_generation.py**: Blog post generation.
  - **utils.py**: Utility functions (logging, saving models, loading datasets).

- **outputs/**: Project results.
  - **models/**: Fine-tuned models.
  - **blog_posts/**: Generated blog posts.
  - **logs/**: Training and evaluation logs.

- **requirements.txt**: Project dependencies.

## Setup Instructions

1. Clone the repository:
   ```bash
   git clone <repository-url>
   cd product-recommendation-finetune
   ```

2. Install the required packages:
   ```bash
   pip install -r requirements.txt
   ```

3. Prepare the data:
   - Place raw data files in `data/raw/`.
   - Run the preprocessing scripts in `src/data_preprocessing.py` or use `utils.ipynb`.

4. Run the Jupyter notebooks for analysis and model training:
   ```bash
   jupyter notebook notebooks/main.ipynb
   ```
   O ejecuta cada notebook paso a paso como se describe arriba.

5. Tras el entrenamiento, revisa `outputs/models/` para los modelos y `outputs/blog_posts/` para los artículos generados.

## Usage Guidelines

- Use functions in `src/model_finetune.py` to fine-tune the model on your dataset.
- Use `src/blog_generation.py` to create blog posts based on recommendations.
- See `data/README.md` for data structure and preprocessing details.

## Ejemplo de uso

```python
from src.model_finetune import finetune_model
from src.blog_generation import generate_blog_post

# Fine-tune the model
finetune_model('data/Processed/combined_reviews.csv')

# Generate a blog post for a category
generate_blog_post(category='Tablets', top_n=3)
```
