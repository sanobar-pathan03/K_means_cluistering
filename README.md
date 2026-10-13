# K_means_cluistering
🩺 An end-to-end Machine Learning pipeline analyzing the Breast Cancer Coimbra Dataset. Features exploratory data analysis, K-Means clustering for clinical patient cohort identification, multi-model evaluation (Random Forest &amp; KNN), and a deployment-ready Streamlit web application.
# 🩺 Clinical Cohort & Cluster Predictor

This repository contains an end-to-end Machine Learning project that leverages clinical data to group patients into distinct metabolic cohorts and classifies new profiles using classical supervised classifiers. The final model is deployed as an interactive web-based dashboard using Streamlit.

## 📋 Project Overview

Using the **Breast Cancer Coimbra Dataset (datar2.csv)**, which contains physical and metabolic indicators, we perform unsupervised learning to define 3 clinical profiles (cohorts):
1. **Cluster 0 (Healthy-Leaning Profile):** Characterized by healthy clinical baselines and lower inflammatory marker loads.
2. **Cluster 1 (High Metabolic/Glycemic Risk Profile):** Marked by high blood glucose levels, spiked insulin, and elevated HOMA-IR indexes.
3. **Cluster 2 (Obesity & Leptin Profile):** Characterized by elevated BMIs and high systemic Leptin levels.

Following unsupervised grouping, we evaluate supervised classifiers (**Random Forest Classifier** and **K-Nearest Neighbors**) to predict these clinical groups with over **90%+ classification accuracy**.

## 🛠️ Key Features

* **Exploratory Data Analysis (EDA):** Insights on dataset distributions and relationships between features like Glucose, Insulin, and BMI.
* **Clustering Analysis:** K-Means clustering ($k=3$) evaluated using PCA dimensionality reduction and pairwise distributions.
* **Supervised Learning:** Classification of patient cohorts with comparative performance evaluation using Random Forest and KNN (generating confusion matrices).
* **Streamlit Application:** A self-contained clinical dashboard allowing medical professionals or researchers to input clinical values and predict client cluster status instantly.

## 📂 Repository Structure

* `datar2.csv` - The clinical breast cancer patient indicator dataset.
* `app.py` - Streamlit application script containing the interactive UI code.
* `model.pkl` - Trained, serialized production-ready Random Forest model.
* `scaler.pkl` - Standard scaler object matching the feature normalization pipeline.
* `requirements.txt` - Python environment dependencies.
* `project_files.zip` - Complete packaged workspace for easy deployment.

## 🚀 Quick Start / Local Installation

Clone the repository and run the Streamlit app locally using these steps:

1. **Clone this repository:**
   ```bash
   git clone https://github.com/YOUR_USERNAME/your-repo-name.git
   cd your-repo-name
Install dependencies:

pip install -r requirements.txt
Launch the Streamlit app:

streamlit run app.py
📊 Model Evaluation Summary
Random Forest Accuracy: ~94%
KNN Accuracy: ~91%
