# Customer Analytics Projects Overview

This repository contains documentation for three customer analytics projects focusing on predictive modeling, customer segmentation, and natural language processing with Retrieval-Augmented Generation (RAG).

---

## 1. Customer Churn Prediction Pipeline

### Goals
* Predict customer churn using machine learning classification models.
* Build an automated preprocessing and model selection pipeline.
* Export the optimal model as a production-ready file (`churn_champion_pipeline.joblib`).

### Key Changes & Implementation
* **Feature Engineering:** Added interaction metrics, including `Charge_Per_Tenure_Ratio` ($\frac{\text{TotalCharges}}{\text{tenure} + 1e-5}$) and `Tenure_Monthly_Interaction` ($\text{tenure} \times \text{MonthlyCharges}$).
* **Data Preprocessing:** Handled numerical features via median imputation and standard scaling, and categorical features via most-frequent imputation and one-hot encoding.
* **Class Imbalance:** Integrated SMOTE (Synthetic Minority Over-sampling Technique) inside an `ImbPipeline`.
* **Model Optimization:** Evaluated Logistic Regression, Random Forest, and XGBoost using `RandomizedSearchCV` with 5-fold `StratifiedKFold` cross-validation based on ROC-AUC scores.

---

## 2. RFM Customer Segmentation & Clustering

### Goals
* Segment customers into distinct groups using unsupervised learning based on historical transaction behavior.
* Map algorithmic cluster outputs to actionable business personas.

### Key Changes & Implementation
* **Data Cleaning:** Filtered out records missing `CustomerID`, product returns (negative quantities), and zero/negative prices.
* **RFM Aggregation:** Calculated `LineTotal` ($\text{Quantity} \times \text{UnitPrice}$) and aggregated data at the customer level to compute **Recency**, **Frequency**, and **Monetary** (RFM) metrics.
* **Transformation & Scaling:** Applied $\log(1+x)$ transformations to normalize skewed distributions prior to standard scaling.
* **Clustering Analysis:** Tested $K$-Means clustering across $K \in [2, 6]$ using Silhouette Scores to select $K=4$ as the optimal cluster count.
* **Persona Mapping:** Created composite score rules ($\text{Monetary} + \text{Frequency} - \text{Recency}$) to map clusters into business segments: **Champions**, **Loyal Customers**, **At-Risk**, and **Hibernating**.
* **Visualization:** Generated scatter plots mapping Recency vs. Monetary values segmented by cluster.

---

## 3. Review Text Mining & RAG Prototype

### Goals
* Build an NLP model integrated with a Retrieval-Augmented Generation (RAG) prototype.
* Process customer review corpora into optimal chunks to enhance retrieval accuracy and contextual generation.
* Extract sentiment insights and topic themes to inform RAG query augmentation.

### Key Changes & Implementation
* **Text Preprocessing & Chunking:** Cleaned customer feedback through lowercasing, regex stripping, and lemmatization, followed by document chunking strategies (overlap and size parameterization) to prepare text for vector indexing.
* **Vector Search & RAG Setup:** Indexed review chunks for semantic similarity search to support contextual retrieval for LLM prompt augmentation.
* **Sentiment Inference:** Evaluated review sentiment using Hugging Face's `distilbert-base-uncased-finetuned-sst-2-english` pipeline to attach sentiment metadata to retrieved text chunks.
* **Topic Extraction:** Vectorized text using `CountVectorizer` and fitted a Latent Dirichlet Allocation (LDA) model to extract word distributions across topics.
* **Data Visualization:** Generated plots showing sentiment distribution and confidence scores across the review dataset.