# Flu Shot Vaccine Prediction with Logistic Regression

Machine learning project focused on predicting **H1N1** and **seasonal flu vaccine uptake** using survey-based respondent data from the **Flu Shot Learning** dataset.

This repository presents a complete supervised learning workflow, from data preparation and exploratory analysis to baseline modelling, hyperparameter tuning and final evaluation on an unseen test set.

---

## Project Overview

Vaccination uptake prediction can support more targeted public health communication and resource allocation. In this project, I built two separate **Logistic Regression** models to predict whether a respondent received:

- **H1N1 vaccine**
- **Seasonal flu vaccine**

The project includes:

- data loading and merging
- exploratory data analysis
- missing value handling through imputation and amputation
- categorical encoding
- feature scaling
- train / validation / test split
- baseline modelling
- hyperparameter tuning with **GridSearchCV**
- validation and final test evaluation

---

## Objectives

The main goal of this project was to evaluate how well Logistic Regression can classify vaccine uptake for two binary targets and compare:

- a **baseline model**
- a **tuned model** with GridSearchCV

The final objective was to assess whether tuning improved performance and whether the final models generalised well on unseen data.

---

## Dataset

**Source:** [DrivenData - Flu Shot Learning: Predict H1N1 and Seasonal Flu Vaccines](https://www.drivendata.org/competitions/66/flu-shot-learning/data/)

Files used in this project:

- `training_set_features.csv`
- `training_set_labels.csv`

The labels were merged with the feature dataset using `respondent_id`.

---

## Tech Stack

- **Python**
- **Pandas**
- **NumPy**
- **Matplotlib**
- **Seaborn**
- **Scikit-learn**
- **Jupyter Notebook**

---

## Project Workflow

### 1. Data Preparation

- Loaded feature and label datasets
- Merged both files using `respondent_id`
- Created a modelling copy of the dataset
- Dropped highly incomplete categorical fields:
  - `employment_industry`
  - `employment_occupation`
- Removed `respondent_id` from the modelling dataset

### 2. Exploratory Data Analysis

The project includes:

- **5 univariate visualisations**
- **5 grouped / complex visualisations**

These visualisations were used to identify:

- target imbalance
- missing data patterns
- relevant behavioural and demographic relationships
- potential drivers of vaccine uptake

### 3. Preprocessing

- Missing categorical values were handled with **most frequent imputation**
- Missing numerical values were handled with **median imputation**
- Categorical variables were encoded with **OneHotEncoder**
- Numerical variables were scaled with **StandardScaler**
- Data was split into:
  - **60% training**
  - **20% validation**
  - **20% test**
- Stratification was applied using `h1n1_vaccine`

### 4. Modelling

Two separate Logistic Regression models were built for:

- `h1n1_vaccine`
- `seasonal_vaccine`

For each target, I trained:

- a **baseline Logistic Regression**
- a **tuned Logistic Regression** using **GridSearchCV**

### 5. Hyperparameter Tuning

The tuned models were optimised using:

- `C`
- `penalty`
- `solver`

The main optimisation metric used during tuning was **ROC-AUC**.

---

## Why Logistic Regression?

Logistic Regression was selected because it is:

- strong for binary classification tasks
- interpretable
- efficient to train
- appropriate for structured tabular data
- a solid baseline for probability-based classification problems

It also allows probability prediction, which is useful when evaluating classification with **ROC-AUC**.

---

## Evaluation Metrics

The models were evaluated using:

- **ROC-AUC**
- **F1-score**
- **Precision**
- **Recall**
- **Accuracy**
- **Classification Report**

ROC-AUC was especially important because it evaluates the quality of probability ranking, while F1-score helped assess balance between precision and recall.

---

## Final Test Results

| Target | ROC-AUC | F1-score | Precision | Recall | Accuracy |
|--------|--------:|---------:|----------:|-------:|---------:|
| H1N1 Vaccine | 0.9310 | 0.6645 | 0.7075 | 0.6264 | 0.8656 |
| Seasonal Vaccine | 0.9883 | 0.9613 | 0.9290 | 0.9959 | 0.9635 |

---

## Key Findings

- Logistic Regression performed strongly for both targets.
- The **seasonal vaccine** target achieved very high and stable performance.
- The **H1N1** target was more challenging, especially in identifying positive cases, but the final model still showed good generalisation.
- Hyperparameter tuning produced only **marginal improvements** over baseline, but the tuned models remained stable across validation and test data.
- Final results suggest that Logistic Regression is a reliable approach for this structured classification problem.

---

## Repository Structure

~~~bash
vaccine_prediction_ml_project/
│
├── assets/                # exported charts and visual assets
├── data/                  # raw dataset files
├── src/                   # notebook / source files
├── requirements.txt       # project dependencies
├── .gitignore
└── README.md
~~~

---

## How to Run

### 1. Clone the repository

~~~bash
git clone https://github.com/enriquebruno12/vaccine_prediction_ml_project.git
cd vaccine_prediction_ml_project
~~~

### 2. Install dependencies

~~~bash
pip install -r requirements.txt
~~~

### 3. Open the notebook

~~~bash
jupyter notebook
~~~

Then open the project notebook inside the `src/` folder.

---

## Reproducibility

This project uses the student ID as the `random_state` for reproducibility in:

- train / validation / test splitting
- Logistic Regression training
- GridSearchCV model configuration

---

## Limitations

Although the project achieved strong results, some limitations remain:

- only Logistic Regression was implemented
- the tuning grid was relatively narrow
- the classification threshold remained fixed at `0.5`
- no class weighting was applied
- no probability calibration was tested
- no external validation dataset was used

---

## Future Improvements

Possible next steps for improving this project include:

- threshold tuning instead of a fixed `0.5` cutoff
- class weighting for the H1N1 target
- probability calibration
- feature selection
- comparison against other machine learning models
- deployment as a simple API or Streamlit app

---

## What This Project Demonstrates

This repository highlights my ability to:

- work with structured tabular datasets
- perform end-to-end machine learning preprocessing
- build and evaluate classification models
- use GridSearchCV for hyperparameter tuning
- interpret validation and test performance critically
- organise a machine learning workflow in a clear and reproducible way

---

## Author

**Enrique Soares**

- GitHub: [enriquebruno12](https://github.com/enriquebruno12)
- LinkedIn: [enrique-bruno](https://www.linkedin.com/in/enriquebruno/)
