## Student Stress Level Prediction

**Dataset:** Student Stress Monitoring Dataset (Kaggle, 880 observations)  
**Tools:** R (ggplot2, dplyr, FactoMineR, randomForest, xgboost, e1071)

### Objective
Classify student stress into three categories — No Stress, Eustress, and Distress — 
using psychological, physiological, lifestyle, and academic variables.

### Methods
- Exploratory Data Analysis (EDA)
- Phi-K Correlation Analysis for mixed data
- Factor Analysis for Mixed Data (FAMD)
- K-Means Cluster Analysis (silhouette score: 0.35)
- Outlier Detection via Mahalanobis Distance
- Ensemble Feature Selection (11 key features selected)
- Models: Random Forest, XGBoost, LightGBM, Logistic Regression, SVM, Neural Network

### Key Results
- Random Forest achieved the best performance: test accuracy 90.45%, Distress F1-score: 0.89
- Top predictors: blood pressure, sleep quality, teacher-student relationship, academic performance
- Cluster analysis aligned strongly with true stress labels
