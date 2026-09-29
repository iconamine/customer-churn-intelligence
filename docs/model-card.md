# Model card

## Intended use

Educational portfolio demonstration of an explainable binary-classification workflow for customer-retention prioritisation.

## Data

All records are generated synthetically from documented distributions. The dataset contains no real customer data and must not be treated as representative of a specific population or organisation.

## Model

A class-balanced logistic regression is fitted after standardising numeric variables and one-hot encoding categorical variables. It is intentionally used as a transparent baseline.

## Evaluation

The command-line workflow reports ROC AUC, accuracy, precision, recall, and the observed churn rate on a stratified holdout set. Results depend on the random seed and synthetic data-generating process.

## Limitations

- synthetic relationships may not transfer to production data;
- the score is not causal and does not estimate the effect of a retention action;
- thresholds require business costs and capacity constraints;
- fairness, calibration, drift, and privacy reviews are mandatory before real use.

