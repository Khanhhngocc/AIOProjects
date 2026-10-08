# Customer Segmentation with K-Means

An end-to-end customer segmentation project based on the **Online Retail dataset**.

The project transforms transaction-level retail data into customer-level behavioral features, applies feature preprocessing and redundancy reduction, performs **K-Means clustering**, and analyzes the resulting customer segments using visualization, SHAP-based interpretation, and cluster profiling.

---

## Project Overview

Customer segmentation helps businesses identify groups of customers with similar purchasing behaviors.

This project builds a clustering pipeline that goes beyond basic RFM analysis by engineering behavioral features across multiple dimensions, including:

- Purchase rhythm
- Spending behavior
- Basket behavior
- Purchase volume and bulk buying
- Product concentration
- Customer lifecycle
- Spending trends
- Seasonal concentration

The final customer representation is used as input for K-Means clustering.

---

## Project Pipeline

```text
Raw Transaction Data
        │
        ▼
Data Cleaning & EDA
        │
        ▼
Customer-Level Feature Engineering
        │
        ▼
Quantile Transformation
        │
        ▼
Correlation-Based Feature Selection
        │
        ▼
Standard Scaling
        │
        ▼
Outlier Clipping
        │
        ▼
Feature Family Weighting
        │
        ▼
K-Means Clustering
        │
        ▼
Cluster Interpretation
        │
        ├── SHAP Analysis
        │
        └── Customer Profiling
````

---

## Feature Engineering

The pipeline initially creates behavioral candidate features and removes highly redundant features before clustering.

### Purchase Rhythm

Examples:

* `Recency`
* `InterPurchaseCV`
* `ActiveSpanDays`
* `MonthlyOrderRate`

### Spending Shape

Examples:

* `AOV`
* `SpendGini`
* `PriceCV`
* `MaxSpendShare`
* `SpendAcceleration`

### Basket Behavior

Examples:

* `NewSKURate`
* `RepeatSKUFraction`
* `BasketSizeCV`

### Volume & Bulk Behavior

Examples:

* `BurstIndex`
* `BulkLineRate`
* `QuantityCV`
* `MedianBasketQty`
* `SKU_HHI`

### Customer Lifecycle & Seasonality

Examples:

* `ReturnRate`
* `ReturnValueRate`
* `QuarterConcentration`

---

## Feature Preprocessing

Several preprocessing steps are applied before clustering.

### Quantile Transformation

A `QuantileTransformer` with a normal output distribution is used to reduce strong skewness and the influence of extreme observations.

```python
QuantileTransformer(
    output_distribution="normal",
    n_quantiles=500,
    random_state=42
)
```

### Correlation Filtering

Highly correlated features are removed using a greedy pairwise correlation filter.

Features whose absolute correlation exceeds the configured threshold are considered redundant.

Some business-interpretable features are protected using a `FORCE_KEEP` rule.

### Standard Scaling

Selected features are standardized using:

```python
StandardScaler()
```

### Outlier Clipping

Standardized features are clipped to:

```text
[-3, 3]
```

to reduce excessive influence from extreme observations.

### Feature Family Weighting

Feature families may contain different numbers of retained features.

To avoid families with more features dominating Euclidean distance, each feature receives:

```text
weight = 1 / sqrt(number of retained features in its family)
```

This gives each feature family approximately equal total squared contribution to the clustering distance.

---

## Project Structure

```text
Customer_Segmentation/
│
├── data/
│   ├── raw/
│   │   └── online_retail.xlsx
│   │
│   └── processed/
│
├── figures/
│   ├── 01_eda/
│   ├── 02_features/
│   ├── 03_modeling/
│   ├── 04_shap/
│   └── 05_profiling/
│
├── notebooks/
│   ├── 01_cleaning_and_eda.ipynb
│   ├── 02_feature_engineering.ipynb
│   ├── 03_modeling.ipynb
│   ├── 04_shap_analysis.ipynb
│   └── 05_cluster_profiling.ipynb
│
├── outputs/
│
├── src/
│   ├── clustering_library/
│   │   ├── __init__.py
│   │   ├── cleaner.py
│   │   ├── clustering_with_rfm_features.py
│   │   ├── features.py
│   │   └── visualizer.py
│   │
│   ├── notebook_io.py
│   └── visual_style.py
│
├── tests/
│
└── README.md
```

---

## Main Modules

### `cleaner.py`

Handles transaction-level preprocessing, including:

* Missing-value handling
* Invalid transaction filtering
* Cancellation filtering
* Customer filtering
* Time-related features
* Transaction value calculation

### `features.py`

Implements customer-level behavioral feature engineering and preprocessing.

Main responsibilities include:

* Behavioral feature generation
* Correlation-based feature filtering
* Quantile transformation
* Standard scaling
* Outlier clipping
* Feature-family weighting
* Pipeline artifact saving

### `visualizer.py`

Contains reusable plotting utilities for:

* Revenue analysis
* Time patterns
* Product analysis
* Customer distributions
* RFM analysis
* Feature and cluster visualizations

### `clustering_with_rfm_features.py`

Provides RFM-based customer-level features for comparison and baseline analysis.

---

## Notebooks

The notebooks should be executed in the following order.

### 1. Data Cleaning and EDA

```text
01_cleaning_and_eda.ipynb
```

Loads the Online Retail dataset, cleans transactions, and explores customer and transaction behavior.

---

### 2. Feature Engineering

```text
02_feature_engineering.ipynb
```

Builds customer-level behavioral features and performs preprocessing.

Typical outputs include:

```text
customer_feature_candidates.csv
customer_features.csv
customer_features_transformed.csv
customer_features_scaled_unweighted.csv
customer_features_scaled.csv
```

---

### 3. Clustering

```text
03_modeling.ipynb
```

Uses the processed feature matrix for K-Means clustering.

The notebook includes model-selection analysis such as:

* Elbow analysis
* Silhouette analysis
* Cluster visualization
* Cluster assignment

---

### 4. SHAP Analysis

```text
04_shap_analysis.ipynb
```

Analyzes how individual features contribute to cluster membership and provides a more interpretable view of the clustering results.

---

### 5. Cluster Profiling

```text
05_cluster_profiling.ipynb
```

Creates business-oriented profiles for the discovered customer segments.

This step helps translate numerical cluster labels into meaningful behavioral customer groups.

---

## Setup

### 1. Clone or open the project

```bash
cd Customer_Segmentation
```

### 2. Create a virtual environment

```bash
python3 -m venv .venv
```

Activate it on macOS/Linux:

```bash
source .venv/bin/activate
```

On Windows:

```bash
.venv\Scripts\activate
```

### 3. Install dependencies

Install the required libraries used by the project, including:

```bash
pip install pandas numpy matplotlib seaborn scikit-learn scipy joblib openpyxl jupyter shap
```

---

## Running the Project

Start Jupyter from the project root:

```bash
jupyter notebook
```

Then execute the notebooks in order:

```text
01_cleaning_and_eda.ipynb
        ↓
02_feature_engineering.ipynb
        ↓
03_modeling.ipynb
        ↓
04_shap_analysis.ipynb
        ↓
05_cluster_profiling.ipynb
```

---

## Input Data

Place the Online Retail dataset at:

```text
data/raw/online_retail.xlsx
```

The dataset contains transaction-level information such as:

* Invoice number
* Product code
* Product description
* Quantity
* Invoice date
* Unit price
* Customer ID
* Country

---

## Generated Artifacts

Processed customer features are stored under:

```text
data/processed/
```

Visualization outputs are stored under:

```text
figures/
```

The feature engineering pipeline can also persist fitted preprocessing objects such as:

```text
scaler.pkl
quantile_transformer.pkl
pipeline_metadata.json
```

These artifacts allow the same preprocessing configuration to be reused later without refitting the transformers.

---

## Technologies

* Python
* Pandas
* NumPy
* Scikit-learn
* SciPy
* Matplotlib
* Seaborn
* SHAP
* Joblib
* Jupyter Notebook

---

## Key Learning Outcomes

This project demonstrates how to:

* Convert transaction-level data into customer-level representations
* Engineer behavioral features for unsupervised learning
* Handle skewed customer behavior distributions
* Reduce redundant features using correlation analysis
* Design feature weights for Euclidean-distance-based clustering
* Select and train K-Means clustering models
* Interpret and profile customer segments
* Organize a data science project into reusable modules and notebooks

---

## Future Improvements

Possible extensions include:

* Automated clustering model comparison
* Additional clustering algorithms such as DBSCAN or Gaussian Mixture Models
* Cluster stability analysis
* Automated customer-segment naming
* Production inference for new customers
* Interactive dashboards for cluster exploration

---


