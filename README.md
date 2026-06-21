# Oncology Outcome Classifier 🧬📊

## 📌 Project Overview
This project implements a comprehensive machine learning pipeline to predict patient survival outcomes using the **SEER (Surveillance, Epidemiology, and End Results) Breast Cancer dataset**. The system benchmarks 8 different classification algorithms to establish a "Champion Model" for clinical outcome prediction.

## Tech Stack
* **Language:** Python 3.x
* **Data Processing:** Pandas, NumPy, Regex
* **Visualization:** Seaborn, Matplotlib
* **Machine Learning:** Scikit-Learn (KNN, SVM, FFNN, Random Forest), XGBoost, LightGBM, CatBoost

## Methodology & Clinical Engineering
1. **Data Leakage Prevention:** Strictly dropped the `Survival Months` feature prior to training to prevent the model from artificially learning the outcome.
2. **Feature Engineering:** Synthesized a new biological marker, the **Lymph Node Ratio (LNR)**, by computing the ratio of positive nodes to examined nodes.
3. **Regex Sanitization:** Cleaned column nomenclatures to ensure compatibility with LightGBM's strict parsing constraints.
4. **Benchmarking:** Scaled all features and trained 8 distinct architectures, ranging from distance-based (KNN) and structural (Neural Networks) to advanced tree ensembles (Gradient Boosting).

##Key Results
* Extracted and visualized the **Pearson Correlation Matrix** to map biological relationships.
* Identified **XGBoost** as the Champion Model for the clinical deep dive.
* Generated **Feature Importance Profiles**, identifying the most critical variables driving mortality predictions in the dataset.

## 🚀 How to Run
1. Ensure the dataset (`dataset.csv`) is in the root directory.
2. Install dependencies: `pip install -r requirements.txt`
3. Execute the pipeline: `python main.py`