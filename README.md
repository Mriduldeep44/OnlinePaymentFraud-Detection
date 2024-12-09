

# 🛡️ Online Payment Fraud Detection

## 🔍 Overview

This repository focuses on detecting **online payment fraud** in transactions involving credit cards, UPI, and other modes. Fraud detection is essential for securing financial transactions and protecting users from cyber threats. In this project, we perform an **Exploratory Data Analysis (EDA)** followed by building a **Machine Learning Model** to predict potential fraudulent activities.

The project is divided into the following main sections:
- **Exploratory Data Analysis (EDA)**: Gaining insights into the transaction patterns and identifying potential fraud indicators.
- **Fraud Detection Model**: Training a machine learning model to detect fraud in real-time transactions.

## 🏗️ Project Features

- 📊 **Comprehensive EDA** to uncover important patterns, correlations, and outliers in the transaction data.
- 🔄 **Data Preprocessing** including handling missing values, encoding categorical features, and feature engineering.
- 🤖 **Machine Learning Models** such as Logistic Regression, Decision Trees, and Random Forest to identify fraudulent transactions.
- 📈 **Evaluation Metrics** including accuracy, precision, recall, F1-score, and ROC-AUC to measure model performance.
- 🎨 **Custom Visualizations** to interpret model results and explore data insights.

## 📂 Project Structure

```bash
online_fraud_detection/
│
├── data/
│   └── transactions.csv                # Transaction data
│
├── notebooks/
│   ├── Online_fraud_detection_EDA.ipynb        # EDA Notebook
│   └── Online_fraud_detection_model.ipynb      # Model Notebook
│
├── images/                             # Visualizations used in the analysis
│
├── README.md                           # This README file
├── requirements.txt                    # Required dependencies
└── LICENSE                             # License file
```

## 🛠️ Installation

### Prerequisites

Ensure the following are installed on your system:

- 🐍 [Python 3.8+](https://www.python.org/downloads/)
- 📒 [Jupyter Notebook](https://jupyter.org/install)
- 🐼 [Pandas](https://pandas.pydata.org/)
- 🧮 [NumPy](https://numpy.org/)
- 📊 [Matplotlib](https://matplotlib.org/)
- 📉 [Scikit-learn](https://scikit-learn.org/)
- 🎨 [Seaborn](https://seaborn.pydata.org/)


### Installation Steps

1. **Clone the repository** to your local machine:
    ```bash
    git clone https://github.com/yourusername/online_fraud_detection.git
    ```
2. **Navigate to the project directory**:
    ```bash
    cd online_fraud_detection
    ```
3. **Install dependencies** listed in `requirements.txt`:
    ```bash
    pip install -r requirements.txt
    ```

## 🚀 Usage

### Running the Exploratory Data Analysis (EDA)
1. Open the `Online_fraud_detection_EDA.ipynb` in Jupyter Notebook.
2. Execute the cells step by step to explore transaction behavior, identify potential fraud indicators, and gain insights through visualizations.

### Running the Fraud Detection Model
1. Open the `Online_fraud_detection_model.ipynb` in Jupyter Notebook.
2. Follow the steps for data preprocessing and training machine learning models.
3. Evaluate the model's performance using metrics like precision, recall, and F1-score.
4. Tune model hyperparameters to optimize fraud detection performance.

## 🧠 Key Insights from EDA

During the EDA, some key insights were uncovered:
- 💡 **Correlation Analysis**: Certain features show a high correlation with fraudulent transactions.
- 🔍 **Anomalies**: Several high-value transactions were identified as potential fraud cases.
- 📊 **Visualization**: Fraud patterns were visualized using scatter plots, histograms, and correlation heatmaps.

## 🔧 Model Pipeline

1. **Data Preprocessing**: 
   - Handle missing data.
   - Encode categorical variables.
   - Standardize/normalize continuous features.
2. **Model Training**:
   - Logistic Regression, Decision Trees, Random Forest.
   - Compare performance using cross-validation.
3. **Evaluation**:
   - Precision, recall, F1-score.
   - ROC-AUC curve analysis.

## 🧪 Model Performance

The models were evaluated based on the following metrics:

- **Accuracy**: 📈 High accuracy indicates that the model correctly identifies the majority of transactions as either fraudulent or legitimate.
- **Precision**: 🎯 Precision reflects the percentage of correctly identified fraudulent transactions among all predicted frauds.
- **Recall**: 🔄 Recall indicates the percentage of actual frauds correctly identified by the model.
- **F1-Score**: ⚖️ F1-Score is the harmonic mean of precision and recall, providing a balance between both.

## 📊 Visualizations

Some important visualizations generated during the project include:
- 📉 **Correlation Heatmap**: Shows relationships between transaction features and fraud occurrence.
- 🔍 **Scatter Plots**: Visualizes the distribution of transaction amounts and fraud probability.
- 📈 **ROC Curve**: Evaluates the model's classification threshold trade-offs.

