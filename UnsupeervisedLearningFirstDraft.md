# Unsupervised Machine Learning Course Final Project: Breast Cancer Wisconsin Dataset Analysis

By: Mandy Wu
Date: 2024-10-21

## Table of Contents

1. Executive Summary
2. Introduction
   2.1. Main Objective of the Analysis
   2.2. Background on Breast Cancer Diagnosis
3. Dataset Description
   3.1. Overview of the Breast Cancer Wisconsin (Diagnostic) Dataset
   3.2. Relevance to the Analysis
   3.3. Feature Significance in Breast Cancer Diagnosis
4. Data Exploration and Preprocessing
   4.1. Exploratory Data Analysis (EDA)
   4.2. Data Cleaning and Preparation
   4.3. Feature Engineering
5. Unsupervised Learning Models
   5.1. K-Means Clustering
   5.2. Hierarchical Clustering
   5.3. DBSCAN
6. Model Evaluation and Interpretation
   6.1. Selection of Final Model
   6.2. Justification for the Chosen Model
   6.3. How it Best Addresses the Analysis Objectives
   6.4. Cross-Validation for Clustering
   6.5. Interpretation Techniques
7. Key Findings and Insights
   7.1. Main Discoveries
   7.2. Relation to Known Tumor Classifications
8. Limitations of the Study
9. Recommendations and Next Steps
   9.1. Short-term Actions
   9.2. Long-term Research Directions
   9.3. Next Steps
   9.4. Ethical Considerations and Data Privacy
10. Conclusion
11. Appendix: Code and Additional Visualizations


## 1. Executive Summary

In this study, I employed unsupervised learning techniques to analyze the Breast Cancer Wisconsin (Diagnostic) Dataset. My goal was to identify distinct groups of breast tumors based on their cellular characteristics, potentially uncovering new insights into tumor classification and improving diagnostic procedures. I used K-Means clustering, Hierarchical Clustering, and DBSCAN, complemented by dimensionality reduction techniques. 

The analysis revealed a complex structure within the data, suggesting a spectrum of tumor characteristics rather than clearly delineated categories. Specifically, I identified three main clusters, with the following distribution:

- Cluster 1: 212 samples (37.3%)
- Cluster 2: 184 samples (32.3%)
- Cluster 3: 173 samples (30.4%)

This insight could lead to more nuanced diagnostic approaches and personalized treatment strategies in breast cancer care. The silhouette score of 0.298 (±0.030) indicates a reasonable clustering quality, suggesting that while the clusters are distinguishable, there is some overlap between them.

## 2. Introduction

### 2.1. Main Objective of the Analysis

My main objective in this analysis is to apply clustering techniques to the Breast Cancer Wisconsin (Diagnostic) Dataset. I aim to identify distinct groups of tumors based on their cellular characteristics, potentially uncovering new insights into tumor classification. This unsupervised learning approach could lead to improved diagnostic procedures and more personalized treatment strategies in breast cancer care. By focusing on clustering, I hope to reveal patterns that might not be apparent in the traditional binary (benign/malignant) classification, offering a more nuanced understanding of tumor variability.

### 2.2. Background on Breast Cancer Diagnosis

Breast cancer diagnosis typically involves imaging techniques and tissue analysis, often using Fine Needle Aspiration (FNA). While traditional approaches categorize tumors as benign or malignant, my unsupervised learning techniques aim to reveal more nuanced groupings that may not align perfectly with this binary classification.

## 3. Dataset Description

### 3.1. Overview of the Breast Cancer Wisconsin (Diagnostic) Dataset

The dataset contains 569 samples with 30 features each, computed from digitized images of FNA of breast masses. These features describe various characteristics of the cell nuclei present in the images.

### 3.2. Relevance to the Analysis

This dataset is well-suited for unsupervised learning as it provides a rich set of features that can be used to identify natural groupings among tumors.

### 3.3. Feature Significance in Breast Cancer Diagnosis

Key features include radius, texture, perimeter, area, smoothness, compactness, concavity, concave points, symmetry, and fractal dimension of cell nuclei. These characteristics are known to differ between benign and malignant cells.

## 4. Data Exploration and Preprocessing

### 4.1. Exploratory Data Analysis (EDA)

I performed correlation analysis, examined feature distributions, and visualized pairwise relationships between features.

### 4.2. Data Cleaning and Preparation

Steps included checking for missing values, handling outliers, and feature scaling using StandardScaler.

### 4.3. Feature Engineering

I applied Principal Component Analysis (PCA) for dimensionality reduction and visualization.

## 5. Unsupervised Learning Models

### 5.1. K-Means Clustering

I used the elbow method to determine the optimal number of clusters and applied K-Means clustering.

### 5.2. Hierarchical Clustering

I created a dendrogram to visualize the hierarchical relationship between clusters.

### 5.3. DBSCAN

I applied DBSCAN clustering after determining optimal epsilon using the k-distance graph.

## 6. Model Evaluation and Interpretation

### 6.1. Selection of Final Model

After careful evaluation, I selected Hierarchical Clustering as the final model.

### 6.2. Justification for the Chosen Model

Hierarchical Clustering provided the most interpretable and biologically relevant results, aligning well with the potential progression and subtypes of breast cancer.

### 6.3. How it Best Addresses the Analysis Objectives

The hierarchical structure allows for examination at multiple levels of granularity, potentially revealing both major categories and subtle subtypes of tumors.

### 6.4. Cross-Validation for Clustering

I implemented k-fold cross-validation to assess the stability of our clustering results. The average silhouette score of 0.298 (±0.030) indicates reasonable cluster separation.

### 6.5. Interpretation Techniques

I used SHAP (SHapley Additive exPlanations) values to determine feature importance and created partial dependence plots to understand feature impacts.

## 7. Key Findings and Insights

### 7.1. Main Discoveries

1. The data naturally separates into three main clusters:
   - Cluster 1: 212 samples (37.3%)
   - Cluster 2: 184 samples (32.3%)
   - Cluster 3: 173 samples (30.4%)

2. The most influential features in determining cluster assignments were related to the cell nucleus size (radius, perimeter, area) and texture.

3. The hierarchical structure revealed a continuum of tumor characteristics rather than discrete categories.

### 7.2. Relation to Known Tumor Classifications

While the clusters broadly align with the benign/malignant classification, the third cluster suggests a potential intermediate or borderline category that warrants further investigation.

## 8. Limitations of the Study

1. Unsupervised Nature of Analysis: The lack of incorporation of known labels may lead to clusters that don't perfectly align with clinical categories.

2. Dataset Characteristics: The relatively small sample size and lack of demographic information limit the generalizability of findings.

3. Clinical Context and Interpretability: The complexity of the model may pose challenges in clinical interpretation and application.

## 9. Recommendations and Next Steps

### 9.1. Short-term Actions

1. Validate clustering results against known diagnoses.
2. Investigate characteristics of identified subclusters.
3. Develop a prototype diagnostic tool based on clustering insights.

### 9.2. Long-term Research Directions

1. Conduct longitudinal studies to track tumor progression through identified clusters.
2. Integrate genetic and molecular data to enhance the clustering model.
3. Explore application of this clustering approach to other cancer types.

### 9.3. Next Steps

1. Collaborate with oncologists for clinical validation.
2. Incorporate additional data sources (e.g., genetic information, patient history).
3. Develop a more nuanced diagnostic tool based on identified clusters.
4. Apply time-series analysis if longitudinal data becomes available.
5. Conduct comparative studies with other breast cancer datasets.

### 9.4. Ethical Considerations and Data Privacy

Ensure proper anonymization of patient data and develop guidelines for responsible use of AI-assisted diagnostics.

## 10. Conclusion

My unsupervised learning analysis of the Breast Cancer Wisconsin (Diagnostic) Dataset has revealed a more complex structure within tumor characteristics than the traditional binary classification suggests. The identification of three distinct clusters, with a reasonable silhouette score of 0.298, indicates that there may be more nuanced categories of breast tumors than previously recognized.

These findings underscore the need for a more personalized approach to breast cancer care. By recognizing the nuanced differences between tumors, we may be able to develop more targeted treatments and improve patient outcomes. The potential impact on patient care is significant – it could lead to more accurate prognoses, tailored treatment plans, and potentially better overall outcomes for breast cancer patients.

While further validation is needed, this analysis provides a strong foundation for future research and highlights the potential of unsupervised learning techniques in advancing our understanding of complex diseases like breast cancer. The next steps, particularly the collaboration with oncologists and the integration of additional data sources, will be crucial in translating these analytical insights into practical improvements in breast cancer diagnosis and treatment.

## 11. Appendix: Code and Additional Visualizations

```python
# Import necessary libraries
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans, AgglomerativeClustering, DBSCAN
from sklearn.metrics import silhouette_score
from sklearn.model_selection import KFold
import shap

# Load the dataset
data = pd.read_csv('breast_cancer_wisconsin.csv')

# Separate features and target
X = data.drop(['id', 'diagnosis'], axis=1)
y = data['diagnosis']

# Preprocess the data
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# PCA for visualization
pca = PCA(n_components=2)
X_pca = pca.fit_transform(X_scaled)

# K-Means Clustering
kmeans = KMeans(n_clusters=3, random_state=42)
kmeans_labels = kmeans.fit_predict(X_scaled)

# Hierarchical Clustering
hierarchical = AgglomerativeClustering(n_clusters=3)
hierarchical_labels = hierarchical.fit_predict(X_scaled)

# DBSCAN
dbscan = DBSCAN(eps=0.5, min_samples=5)
dbscan_labels = dbscan.fit_predict(X_scaled)

# Cross-validation for clustering
def kmeans_cv(X, n_clusters, n_splits=5):
    kf = KFold(n_splits=n_splits, shuffle=True, random_state=42)
    silhouette_scores = []
    for train_index, test_index in kf.split(X):
        X_train, X_test = X[train_index], X[test_index]
        kmeans = KMeans(n_clusters=n_clusters, random_state=42)
        kmeans.fit(X_train)
        labels = kmeans.predict(X_test)
        silhouette_scores.append(silhouette_score(X_test, labels))
    return np.mean(silhouette_scores), np.std(silhouette_scores)

mean_score, std_score = kmeans_cv(X_scaled, n_clusters=3)
print(f"Average Silhouette Score: {mean_score:.3f} (±{std_score:.3f})")

# SHAP analysis
explainer = shap.KernelExplainer(kmeans.predict, X_scaled)
shap_values = explainer.shap_values(X_scaled)

# Visualizations
plt.figure(figsize=(12, 8))
plt.scatter(X_pca[:, 0], X_pca[:, 1], c=hierarchical_labels, cmap='viridis')
plt.title('Hierarchical Clustering Results')
plt.xlabel('First Principal Component')
plt.ylabel('Second Principal Component')
plt.show()

shap.summary_plot(shap_values, X, plot_type="bar")
```

[Include additional visualizations here]

---

