# Breast-Cancer-Prediction-Using ML
I have done this project in the internship program offered by Edunet  
This project aims to predict whether a breast cancer tumor is malignant or benign based on various features. Using a machine learning approach, this model is trained on a breast cancer dataset and can provide accurate predictions for unseen data. The primary goal of this project is to provide a predictive tool that can assist medical professionals in diagnosing breast cancer early, ultimately helping to save lives.

 ## Table of Contents

* [Project Overview](#project-overview)
* [Dataset](#dataset)
* [Technologies Used](#technologies-used)
* [Installation Instructions](#installation-instructions)
* [Usage](#usage)
* [Model](#model)
* [Results](#results)
* [Contributing](#contributing)
* [License](#license)

## Project Overview

Breast cancer is one of the most common types of cancer among women worldwide. Early detection plays a crucial role in improving survival rates, and machine learning provides an efficient way to predict the likelihood of cancer being malignant or benign based on various clinical and diagnostic parameters.

In this project, we built multiple models that use machine learning algorithms to predict breast cancer outcomes. The project uses the scikit-learn's built-in Breast Cancer Wisconsin dataset, containing features like radius, texture, smoothness, and other cell nuclei characteristics.

## Dataset

The dataset used in this project is the **Breast Cancer Wisconsin (Diagnostic) dataset, available in `scikit-learn`.

* **Dataset Description**: This dataset consists of 569 instances of breast cancer diagnoses, with 30 features for each instance. The output variable (class) is binary:
  * Malignant
  * Benign

* **Features** include:
  * `radius_mean`
  * `texture_mean`
  * `smoothness_mean`
  * `compactness_mean`
  * `concavity_mean`
  * ... (30 features in total)

## Technologies Used

* **Python**: Core programming language
* **Pandas**: Data manipulation
* **Scikit-learn**: ML algorithms and evaluation
* **Matplotlib** & **Seaborn**: Data visualization
* **Jupyter Notebook / Python Scripts**: For interactive development and modular code execution

## Installation Instructions

To run this project on your local machine:

1. **Clone the repository**:

   ```bash
   git clone https://github.com/your-username/breast-cancer-prediction.git
````

2. **Navigate to the project directory**:

   ```bash
   cd breast-cancer-prediction
   ```

3. **(Optional) Create a virtual environment**:

   ```bash
   python -m venv venv
   source venv/bin/activate   # Mac/Linux
   venv\Scripts\activate      # Windows
   ```

4. **Install required libraries**:

   ```bash
   pip install -r requirements.txt
   ```

## Usage

Run each objective-based script independently:

```bash
python objective1_tumor_classification.py
python objective2_feature_importance.py
python objective3_risk_estimator.py
python objective4_model_comparison.py
```

Each script performs a unique task like tumor classification, feature importance visualization, personalized risk estimation, or model comparison.

## Model

Multiple models were trained and evaluated:

* **Support Vector Machine (SVM)** – for early tumor classification
* **Random Forest Classifier** – to rank the most influential features
* **Logistic Regression** – to estimate individual risk
* **Decision Tree** – for interpretable model comparison

## Results

* **Objective 1**: SVM achieved high classification accuracy. Output includes confusion matrix heatmap.
* **Objective 2**: Random Forest ranked top 10 most important features. Displayed using a bar graph.
* **Objective 3**: Logistic Regression provided patient-level risk with a prediction graph.
* **Objective 4**: Compared models using AUC Score, Confusion Matrix, and graphical AUC comparison.
