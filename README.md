# Interpretable Machine Learning for Life Expectancy Prediction

## Authors
- Roman Dolgopolyi  
- Ioanna Amaslidou  
- Agrippina Margaritou  

---

## Overview

Life expectancy is a critical measure of population health and societal well-being, yet predicting it accurately remains a challenge due to the interplay of numerous demographic, environmental, and healthcare factors. This study explores the effectiveness of three machine learning models—**Linear Regression (LR)**, **Regression Decision Tree (RDT)**, and **Random Forest (RF)**—for life expectancy prediction using real-world data sourced from the **World Health Organization (WHO)** and the **United Nations (UN)**.

The project emphasizes both **predictive performance** and **model interpretability**, using statistical and visualization-based methods to identify the most influential features and to enhance the transparency of the predictive models.

---

## Data Source

- **Source**: WHO and UN datasets, accessed via [Kaggle](https://www.kaggle.com/datasets/kumarajarshi/life-expectancy-who)  
- **Period**: 2000–2015  
- **Size**: 2,938 rows × 22 columns  
- **Countries**: 193  

---

## Methodology

### 1. Data Preprocessing

- **Renaming**: Standardized column names for consistency.
- **Missing Values**: 
  - Removed features with >5% missing data: `alcohol`, `hepatitis_b`, `tot_expenditure`, `gdp`, `population`, `income_composition_of_resources`, `school_years`.
  - Dropped rows with any remaining missing values. Final dataset: **941 samples × 15 features**.
- **Inconsistencies**: Filtered unrealistic values (e.g., `bmi ≤ 100`, `infant_deaths ≤ 1000`).
- **Categorical Mapping**:
  - `country` replaced with `latitude` and `longitude` (from Google Earth).
  - `status` one-hot encoded into `status_developing` and `status_developed`.

### 2. Exploratory Data Analysis

- **Histograms**: Plotted key features (`year`, `infant_deaths`, `polio`, `diphtheria`) to examine distributions.
- **Life Expectancy Distribution**: Right-skewed, with most values between 60–75 years.
- **Feature Correlation**: Pearson heatmaps and ANOVA revealed:
  - Strong positive: `infant_deaths` ↔ `under_five_deaths` (r=0.99)
  - Strong negative: `adult_mortality` ↔ `diphtheria` (r=-0.47)

### 3. Clustering & PCA

- **Scaling Methods Tested**: `StandardScaler`, `MinMaxScaler`, `MaxAbsScaler`
- **Clustering Models**: `KMeans`, `AgglomerativeClustering`
- **Best Setup**: 
  - **MaxAbsScaler + KMeans**
  - **3 clusters** (Silhouette Score = 0.59)
- **Visualization**: PCA-reduced scatter plot confirmed clear cluster separation.

### 4. Regression Models

| Model | R² | MAE | RMSE | Training Time | Inference Time |
|-------|----|-----|------|----------------|----------------|
| Linear Regression | 0.7296 | 3.7398 | 4.8169 | 0.00235 s | 0.00002 s |
| Decision Tree | 0.8979 | 2.0902 | 2.9602 | 11.42 s | 0.00005 s |
| Random Forest | **0.9423** | **1.4799** | **2.2245** | 32.69 s | 0.00358 s |

#### Linear Regression (LR)

- Fastest and simplest.
- Key features by p-value: `adult_mortality`, `hiv_aids`, `bmi`, `diphtheria`, `polio`.

#### Decision Tree (RDT)

- Used grid search with 5-fold CV.
- Optimal config: `max_depth=15`, `min_samples_leaf=5`, `ccp_alpha=0.01`.
- Tree visualization showed key splits on: `hiv_aids`, `adult_mortality`, `under_five_deaths`.

#### Random Forest (RF)

- Best performing model.
- Tuned for: `n_estimators=200`, `bootstrap=False`, `min_samples_split=2`.
- Top features: `diphtheria`, `year`, `measles`.

---

## Key Findings

- **Best Model**: Random Forest (R² = 0.9423).
- **Top Predictors**:
  - Strong negative: `adult_mortality`, `hiv_aids`, `thinness_1to19years`
  - Strong positive: `bmi`, `diphtheria`, `polio`, `year`
- **Interpretability**: 
  - LR: P-values
  - RDT: Tree visualization
  - RF: Feature importance plots

---

## Requirements / Dependencies

```bash
pip install pandas numpy matplotlib seaborn plotly scikit-learn statsmodels
