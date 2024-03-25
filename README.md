# Loan Evaluation analysis and ML model

#### -- Project Status: [Inactive]

## Project Intro/Objective
The purpose of this project is to further develope machine learning skills as part of the Data Science programme curriculum. 
LendingClub was one of the first American peer-to-peer lending companies, headquartered in San Francisco, California. The platform enables borrowers to create unsecured personal loans between 1,000 USD and 40,000 USD for personal loans and up to 500,000 USD for business use. Investors can search and browse the loan listings on Lending Club website and select loans that they want to invest in based on the information supplied about the borrower, amount of loan, loan grade, and loan purpose. Investors make money from interest. Lending Club makes money by charging borrowers an origination fee and investors a service fee.

The big assignment is to automate LendingClub loan decision and evaluation process fully. The task is split into three objectives:

- Create a machine learning model to classify loans into accepted/rejected
- Create a machine learning model to predict the grade of an accepted loan
- Create a machine learning model to predict sub-grade and interest rate of the accepted loan

Once the models are created, tuned and tested deploy to Google Cloud Plaform.

### Technologies
* Python
* Pandas, Jupyter
* Scikit-Learn
* LightGBM
* FastAPI
* SHAP
* Optuna 

## Project Description
For the given Lending Club dataset (https://storage.googleapis.com/335-lending-club/lending-club.zip) preprocessing notebook was developed, EDA performed and three machine learning models developed, tuned and tested:

- Accepted/Rejected loan binary predictor with binary LigthGBM classier (2 classes, Balanced Accuracy score: 0.96)
- Accepted loan grade predictor with multi-class LightGBM classifier (7 classes, Macro F1 score: 0.32 )
- Accepted loan sub-grade predictor with multi-class LightGBM classifier (36 classes, Macro F1 score: 0.23 )
- Accepted loan interest rate predictor with LightGBM regressor (RSME score: 0.42 )

Every model is explained using SHAP analysis diagrams. 

## Needs of this project

- Learning purposes

## Getting Started

1. Clone this repo (for help see this [tutorial](https://github.com/TuringCollegeSubmissions/vbeino-ML.3.5.git)).
2. Pip install requirements.txt
3. Download and unzip the dataset locally 
4. Run notebook 'Part1_loan_decision.ipynb'
5. Run notebooks 'Part2_loan_grade.ipynb' & 'Part3_subgrade_int_rate.ipynb' for further analysis and loan grade, sub-grade and interest rate models

## Author 

**Lead : [Vytas Bein ]**

