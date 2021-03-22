# BayesianLinearRegressor

This reposity covers two files: **bayesian_linear_regressor.py** and **test_suite.py**. They work with the BanditoAPI (https://github.com/KoyoteScience/BanditoAPI) to test different approaches to performing a linear regression with correct uncertainties:

* Using Linear Algebra Identities (which doesn't scale with the number of training rows)
* Using the statsmodels library (as a ground truth, and which also doesn't scale with the number of training rows)
* Using the Bandito API (online/incremental learning/updating)
* Using the BayesianLinearRegressor (online/incremtal learning/updating)

At Koyote Science, LLC, we use an even more stringent set of integration tests to ensure the accuracy and efficiency of Bandito.

