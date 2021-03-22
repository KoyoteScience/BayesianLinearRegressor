# BayesianLinearRegressor

This reposity covers two files: **bayesian_linear_regressor.py** and **test_suite.py**. They work with the BanditoAPI (https://github.com/KoyoteScience/BanditoAPI) to test different approaches to performing a linear regression with correct uncertainties:

* Using Linear Algebra Identities (which doesn't scale with the number of training rows)
* Using the statsmodels library (as a ground truth, and which also doesn't scale with the number of training rows)
* Using the Bandito API (online/incremental learning/updating)
* Using the BayesianLinearRegressor (online/incremtal learning/updating)

At Koyote Science, LLC, we use an even more stringent set of integration tests to ensure the accuracy and efficiency of Bandito.

Given a set of ![](https://latex.codecogs.com/svg.latex?n_text{obs}) observations with values ![](https://latex.codecogs.com/svg.latex?\mathbf{X}) and ![](https://latex.codecogs.com/svg.latex?\mathbf{y}), we concatenate these arrays to yield ![](https://latex.codecogs.com/svg.latex?\mathbf{X}\oplus\mathbf{y}). Thus after ![](https://latex.codecogs.com/svg.latex?n) observations, and adding in the ![](https://latex.codecogs.com/svg.latex?n_text{obs}) new observations, the update rules are as follows:

* ![](https://latex.codecogs.com/svg.latex?\mathbf{X_{n+n_{\text{obs}}}\oplus\mathbf{y_{n+n_{\text{obs}}}=\mathbf{X_{n}\oplus\mathbf{y_{n}}+(\mathbf{X}\oplus\mathbf{y})^{\text{T}}\mathbf{X}\oplus\mathbf{y})
* ![](https://latex.codecogs.com/svg.latex?R_{n+n_{\text{obs}}}=R_{n}+\mathbf{y}^{\text{T}}\mathbf{y}-\mathbf{\mu}_{n+n_{\text{obs}}}^\text{T}\mathbf{\Sigma}_{n+n_{\text{obs}}}^{-1}\mathbf{\mu}_{n+n_{\text{obs}}}+\mathbf{\mu}_{n}^\text{T}\mathbf{\Sigma}_{n}^{-1}\mathbf{\mu}_{n})

