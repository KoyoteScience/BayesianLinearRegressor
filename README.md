# BayesianLinearRegressor

This reposity covers two files: **bayesian_linear_regressor.py** and **test_suite.py**. They work with the BanditoAPI (https://github.com/KoyoteScience/BanditoAPI) to test different approaches to performing a linear regression with correct uncertainties:

* Using Linear Algebra Identities (which doesn't scale with the number of training rows)
* Using the statsmodels library (as a ground truth, and which also doesn't scale with the number of training rows)
* Using the Bandito API (online/incremental learning/updating)
* Using the BayesianLinearRegressor (online/incremtal learning/updating)

At Koyote Science, LLC, we use an even more stringent set of integration tests to ensure the accuracy and efficiency of Bandito.

Given a set of ![](https://latex.codecogs.com/svg.latex?n_\text{obs}) observations with values ![](https://latex.codecogs.com/svg.latex?\mathbf{X}) and ![](https://latex.codecogs.com/svg.latex?\mathbf{y}), we concatenate these arrays to yield ![](https://latex.codecogs.com/svg.latex?\mathbf{X}\oplus\mathbf{y}). Thus after ![](https://latex.codecogs.com/svg.latex?n) observations, and adding in the ![](https://latex.codecogs.com/svg.latex?n_\text{obs}) new observations, the update rules are as follows:

* <img src="https://latex.codecogs.com/svg.latex?(\mathbf{X}\oplus\mathbf{y})_{n+n_{\text{obs}}}^\text{T}(\mathbf{X}\oplus\mathbf{y})_{n+n_{\text{obs}}}=(\mathbf{X}\oplus\mathbf{y})_{n}^\text{T}(\mathbf{X}\oplus\mathbf{y})_{n}+(\mathbf{X}\oplus\mathbf{y})^{\text{T}}(\mathbf{X}\oplus\mathbf{y})">
 
  * equivalent to <img src="https://latex.codecogs.com/svg.latex?(\mathbf{X}^\text{T}\mathbf{X})_{n+n_{\text{obs}}}=(\mathbf{X}^\text{T}\mathbf{X})_{n}+\mathbf{X}^{\text{T}}\mathbf{X}">
  * equivalent to <img src="https://latex.codecogs.com/svg.latex?\mathbf{y}_{n+n_{\text{obs}}}^\text{T}\mathbf{y}_{n+n_{\text{obs}}}=\mathbf{y}_{n}^\text{T}\mathbf{y}_{n}+\mathbf{y}^{\text{T}}\mathbf{y}">
  * equivalent to <img src="https://latex.codecogs.com/svg.latex?\mathbf{X}_{n+n_{\text{obs}}}^\text{T}\mathbf{y}_{n+n_{\text{obs}}}=\mathbf{X}_{n}^\text{T}\mathbf{y}_{n}+\mathbf{X}^{\text{T}}\mathbf{y}">
* <img src="https://latex.codecogs.com/svg.latex?R_{n+n_{\text{obs}}}=R_{n}+\mathbf{y}^{\text{T}}\mathbf{y}-\mathbf{\mu}_{n+n_{\text{obs}}}^\text{T}\mathbf{\Sigma}_{n+n_{\text{obs}}}^{-1}\mathbf{\mu}_{n+n_{\text{obs}}}+\mathbf{\beta}_{n}^\text{T}\mathbf{\Sigma}_{n}^{-1}\mathbf{\beta}_{n}">
* <img src="https://latex.codecogs.com/svg.latex?\mathbf{\Sigma}_{n}^{-1}=\mathbf{X}_{n}^\text{T}\mathbf{X}_{n}+\lambda\mathbf{I}">
* <img src="https://latex.codecogs.com/svg.latex?\mathbf{\beta}_{n}=\mathbf{\Sigma}_{n}\mathbf{X}_{n}^\text{T}\mathbf{y}">

