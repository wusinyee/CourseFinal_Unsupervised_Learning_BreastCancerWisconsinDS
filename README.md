# Unsupervised Learning for Tumor Segmentation in the Breast Cancer Wisconsin Dataset

By: Mandy Wu

Date: 2024-10-21


| Section | Title |
|---------|-------|
| 1 | Executive Summary |
| 2 | Introduction |
| 2.1 | Main Objective of the Analysis |
| 2.2 | Background on Breast Cancer Diagnosis |
| 3 | Dataset Description |
| 3.1 | Overview of the Breast Cancer Wisconsin (Diagnostic) Dataset |
| 3.2 | Relevance to the Analysis |
| 3.3 | Feature Significance in Breast Cancer Diagnosis |
| 4 | Data Quality Assessment and Preprocessing |
| 4.1 | Data Quality Assessment |
| 4.2 | Data Cleaning and Preparation |
| 4.3 | Feature Engineering |
| 5 | Exploratory Data Analysis (EDA) |
| 6 | Unsupervised Learning Models |
| 6.1 | K-Means Clustering |
| 6.2 | Hierarchical Clustering |
| 6.3 | DBSCAN |
| 6.4 | Gaussian Mixture Models |
| 7 | Model Evaluation and Interpretation |
| 7.1 | Selection of Final Model |
| 7.2 | Justification for the Chosen Model |
| 7.3 | How it Best Addresses the Analysis Objectives |
| 7.4 | Cross-Validation for Clustering |
| 7.5 | Interpretation Techniques |
| 8 | Key Findings and Insights |
| 8.1 | Main Discoveries |
| 8.2 | Relation to Known Tumor Classifications |
| 9 | Limitations of the Study |
| 10 | Practical Implications |
| 11 | Ethical Considerations and Data Privacy |
| 12 | Recommendations and Next Steps |
| 12.1 | Short-term Actions |
| 12.2 | Long-term Research Directions |
| 13 | Conclusion |
| 14 | Appendix: Code and Additional Visualizations |

## 1. Executive Summary

In this study, I employed unsupervised learning techniques to analyze the Breast Cancer Wisconsin (Diagnostic) Dataset. My goal was to identify distinct groups of breast tumors based on their cellular characteristics, potentially uncovering new insights into tumor classification and improving diagnostic procedures. I used K-Means clustering, Hierarchical Clustering, and DBSCAN, complemented by dimensionality reduction techniques. 

The analysis revealed a complex structure within the data, suggesting a spectrum of tumor characteristics rather than clearly delineated categories. Specifically, I identified three main clusters, with the following distribution:

- Cluster 1: 212 samples (37.3%)
- Cluster 2: 184 samples (32.3%)
- Cluster 3: 173 samples (30.4%)

This insight could lead to more nuanced diagnostic approaches and personalized treatment strategies in breast cancer care. The silhouette score of 0.298 (±0.030) indicates a reasonable clustering quality, suggesting that while the clusters are distinguishable, there is some overlap between them.







------------------------------------------------







## README

This project presents an unsupervised learning analysis of the Breast Cancer Wisconsin Dataset. The main objective is to use clustering techniques to identify distinct groups of breast cancer tumors, potentially enhancing current diagnostic methods and contributing to personalized treatment strategies.

### Key Features:
- Comprehensive exploratory data analysis
- Implementation of multiple clustering algorithms (K-Means, Hierarchical, DBSCAN)
- Detailed interpretation of clustering results
- Implications for breast cancer diagnosis and treatment
- Recommendations for future research and applications

### How to Use:
1. Clone this repository
2. Install required dependencies (list provided in requirements.txt)
3. Run the Jupyter notebook to reproduce the analysis
4. Refer to the markdown document for a detailed report of findings and insights

### Contributors:
This project uses the Breast Cancer Wisconsin (Diagnostic) Data Set from the UCI Machine Learning Repository. The original contributors to this dataset are:

1. Dr. William H. Wolberg, General Surgery Dept., University of Wisconsin, Clinical Sciences Center, Madison, WI 53792
2. W. Nick Street, Computer Sciences Dept., University of Wisconsin, 1210 West Dayton St., Madison, WI 53706
3. Olvi L. Mangasarian, Computer Sciences Dept., University of Wisconsin, 1210 West Dayton St., Madison, WI 53706

For more information about the dataset, visit: https://archive.ics.uci.edu/dataset/17/breast+cancer+wisconsin+diagnostic
https://www.kaggle.com/datasets/uciml/breast-cancer-wisconsin-data

### License:
This project is licensed under the MIT License.

For any questions or feedback, please open an issue in this repository.
