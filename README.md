# AIAP22 Technical Assessment – Phishing Website Detection  
**Author:** Lian Jia Rong Jerome  
**Email:** jeromelian27@gmail.com  

---

## 1. Project Overview  

This repository contains my full submission for the AIAP Batch 22 Technical Assessment.

The assessment requires two components:

1. **Exploratory Data Analysis (EDA)** in a Jupyter notebook (`eda.ipynb`)
2. **An end-to-end Machine Learning Pipeline** implemented in Python scripts under `src/` and executed using `run.sh`

The goal is to build and evaluate machine learning models that classify websites as **phishing (label = 0)** or **legitimate (label = 1)** using features extracted from website metadata such as redirects, iFrames, link structure, domain age, and hosting provider.

The dataset is stored in a SQLite file (`phishing.db`) and accessed via the `data/` directory.  
Per assessment rules, the database is not uploaded to GitHub.

---

## 2. Folder Structure  

.
├── eda.ipynb
├── requirements.txt
├── run.sh
├── data/
│ └── phishing.db (NOT committed to GitHub)
├── src/
│ ├── config.py
│ ├── data_loader.py
│ ├── preprocess.py
│ ├── train.py
│ └── utils.py
└── outputs/
├── models/
├── plots/
├── reports/
└── summary.csv

---

## 3. How to Run the Pipeline  

### **3.1 Environment Setup**

```bash
python -m venv .venv

# Windows
.venv\Scripts\activate

# macOS / Linux
source .venv/bin/activate

pip install -r requirements.txt

Ensure that:
data/phishing.db
exists locally.

3.2 Execute the pipeline
./run.sh
This script:

Loads dataset through SQLite

Preprocesses and performs feature engineering

Trains 3 ML models

Evaluates each model

Saves:

trained models → outputs/models/

ROC curves → outputs/plots/

classification reports → outputs/reports/

summary.csv → outputs/summary.csv

4. Pipeline Architecture & Flow
Step 1 — Data Loading (data_loader.py)

Loads SQLite database using sqlite3

Reads table into pandas DataFrame

Centralizes DB path via config.py

Step 2 — Preprocessing & Feature Engineering (preprocess.py)

Operations include:

Missing value handling

Casting columns to numeric

One-hot encoding categorical features (Industry, HostingProvider, Robots, IsResponsive)

Keeping meaningful numeric features

Splitting into X (features) and y (label)

Step 3 — Model Training (train.py)

Trains three models:

Logistic Regression

Random Forest

Gradient Boosting Classifier

For each model, the script:

Performs train/test split

Fits the model

Computes evaluation metrics:

Accuracy

ROC AUC

Classification report

Saves classifier using joblib

Produces ROC curve

Saves per-model report

Finally, results are compiled into outputs/summary.csv.

5. How EDA Informed Pipeline Design

The EDA (eda.ipynb) provided insights that directly shaped the preprocessing and model pipeline.

Data Quality Observations

Several numeric fields contained missing or extreme values → imputation required

Categorical fields (Industry, HostingProvider) contained many unique categories → one-hot encoding needed

label was not severely imbalanced → standard ML algorithms appropriate

Feature Behavior Insights

Patterns observed:

Phishing sites tend to have fewer self-references and more external references

Legitimate sites tend to have longer domain age

Phishing sites often lack responsiveness and robots.txt

Tree-based models captured nonlinear interactions between redirect counts, iFrames, and linking structure

Pipeline Implications

These observations guided:

Deciding which columns required numeric conversion vs one-hot encoding

Choosing tree-based models (Random Forest, Gradient Boosting) since they handle categorical expansion and nonlinear patterns well

Retaining ROC AUC as a key metric due to ranking/prediction importance in security applications

Feature cleaning, missing value strategies, and model selection

6. Feature Processing Summary
Feature	Type	Processing Applied
LineOfCode	Numeric	Missing filled
LargestLineLength	Numeric	Missing filled
NoOfURLRedirect	Numeric	As-is
NoOfSelfRedirect	Numeric	As-is
NoOfPopup	Numeric	As-is
NoOfiFrame	Numeric	As-is
NoOfSelfRef	Numeric	As-is
NoOfExternalRef	Numeric	As-is
Robots	Categorical	One-hot encoded
IsResponsive	Categorical	One-hot encoded
Industry	Categorical	One-hot encoded
DomainAgeMonths	Numeric	As-is
HostingProvider	Categorical	One-hot encoded
label	Target	0 = phishing, 1 = legitimate
7. Model Selection Rationale
Model	Reason
Logistic Regression	Strong linear baseline, interpretable
Random Forest	Robust, handles outliers and nonlinear interactions, strong performance on tabular data
Gradient Boosting	Often best-in-class for tabular tasks, high ROC AUC, handles complex feature interactions
8. Evaluation Summary

Final performance from running train.py:

Model	Accuracy	ROC AUC
Logistic Regression	~0.7810	~0.8144
Random Forest	~0.8310	~0.8814
Gradient Boosting	~0.8338	~0.8922

Gradient Boosting achieved the best overall performance.

9. Deployment Considerations

Package trained model as .joblib

Wrap inference with FastAPI or Flask for browser extension integration

Validate incoming fields before prediction

Monitor model drift (phishing tactics evolve regularly)

Consider using probability thresholds tuned for risk tolerance

10. Closing Remarks

This project demonstrates a full end-to-end machine learning workflow, including EDA, feature engineering, model experimentation, evaluation, and pipeline structuring.

Thank you for reviewing my submission.
