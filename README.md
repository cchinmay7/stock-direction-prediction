# Stock Direction Prediction

Predicting the weekly direction (Up/Down) of the S&P 500 from lagged returns and trading volume, comparing logistic regression, k-nearest neighbours, and Gaussian Naive Bayes.

## Data

The `Weekly` S&P 500 dataset: 1,089 weekly observations from 1990 to 2010 with nine variables: `Year`, `Lag1`–`Lag5` (percentage returns for the previous five weeks), `Volume`, `Today`, and `Direction`.

## Approach

1. Exploratory analysis: descriptive statistics, correlation matrix, pair plots, and Up/Down counts by year.
2. Logistic regression (`statsmodels`) on all lag and volume predictors, then on `Lag2` alone. Only `Lag1` and `Lag2` were significant at p < 0.05 in the full model.
3. kNN (k = 1, 3, 5; Euclidean and Manhattan distance) and Gaussian Naive Bayes on the same features.
4. Test-set accuracy and confusion matrices for every model.

## Results

| Model | Test accuracy |
| --- | --- |
| Logistic regression, `Lag2` only | **62.50%** |
| Gaussian Naive Bayes | 58.65% |
| kNN, k = 3 | 53.85% |
| Logistic regression, all predictors | 53.67% |
| kNN, k = 1 | 50.00% |

A single well-chosen predictor beats the full model. The extra lags add noise rather than signal, which is consistent with weekly returns being close to a random walk.

## Files

- `ML.ipynb`: full analysis with outputs
- `Predictive Modeling in Finance.pdf`: written report

## Running

```bash
pip install pandas numpy matplotlib seaborn statsmodels scikit-learn jupyter
```

Place `Weekly.csv` (from the *ISLR* package) next to `ML.ipynb` and run the notebook.
