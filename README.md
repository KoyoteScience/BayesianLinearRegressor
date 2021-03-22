# BayesianLinearRegressor

This reposity covers two files: **bayesian_linear_regressor.py** and **test_suite.py**. They work with the BanditoAPI (https://github.com/KoyoteScience/BanditoAPI) to test different approaches to performing a linear regression with correct uncertainties:

* Using Linear Algebra Identities (which doesn't scale with the number of training rows)
* Using the statsmodels library (as a ground truth, and which also doesn't scale with the number of training rows)
* Using the Bandito API (online/incremental learning/updating)
* Using the included BayesianLinearRegressor module (online/incremtal learning/updating)

At Koyote Science, LLC, we use an even more stringent set of integration tests to ensure the accuracy and efficiency of Bandito.

Given a set of ![](https://latex.codecogs.com/svg.latex?n_\text{obs}) observations with values ![](https://latex.codecogs.com/svg.latex?\mathbf{X}) and ![](https://latex.codecogs.com/svg.latex?\mathbf{y}), we concatenate these arrays to yield ![](https://latex.codecogs.com/svg.latex?\mathbf{X}\oplus\mathbf{y}). Thus after ![](https://latex.codecogs.com/svg.latex?n) observations, and adding in the ![](https://latex.codecogs.com/svg.latex?n_\text{obs}) new observations, and setting ![](https://latex.codecogs.com/svg.latex?R) as the sum of squared residuals, ![](https://latex.codecogs.com/svg.latex?\mathbf{\hat{\beta}}) as the model coefficient means, ![](https://latex.codecogs.com/svg.latex?\Sigma) as the parameter covariance matrix, and ![](https://latex.codecogs.com/svg.latex?\lambda) as the ridge regularization constant (we generally set it to 1e-6), the update rules are as follows:

* <img src="https://latex.codecogs.com/svg.latex?((\mathbf{X}\oplus\mathbf{y})^\text{T}(\mathbf{X}\oplus\mathbf{y}))_{n+n_{\text{obs}}}=((\mathbf{X}\oplus\mathbf{y})^\text{T}(\mathbf{X}\oplus\mathbf{y}))_{n}+(\mathbf{X}\oplus\mathbf{y})^{\text{T}}(\mathbf{X}\oplus\mathbf{y})"> equivalent to:
  * <img src="https://latex.codecogs.com/svg.latex?(\mathbf{X}^\text{T}\mathbf{X})_{n+n_{\text{obs}}}=(\mathbf{X}^\text{T}\mathbf{X})_{n}+\mathbf{X}^{\text{T}}\mathbf{X}">
  * <img src="https://latex.codecogs.com/svg.latex?(\mathbf{y}^\text{T}\mathbf{y})_{n+n_{\text{obs}}}=(\mathbf{y}^\text{T}\mathbf{y})_{n}+\mathbf{y}^{\text{T}}\mathbf{y}">
  * <img src="https://latex.codecogs.com/svg.latex?(\mathbf{X}^\text{T}\mathbf{y})_{n+n_{\text{obs}}}=(\mathbf{X}^\text{T}\mathbf{y})_{n}+\mathbf{X}^{\text{T}}\mathbf{y}">
* <img src="https://latex.codecogs.com/svg.latex?R_{n+n_{\text{obs}}}=R_{n}+\mathbf{y}^{\text{T}}\mathbf{y}-\mathbf{\hat{\beta}}_{n+n_{\text{obs}}}^\text{T}\mathbf{\Sigma}_{n+n_{\text{obs}}}^{-1}\mathbf{\hat{\beta}}_{n+n_{\text{obs}}}+\mathbf{\hat{\beta}}_{n}^\text{T}\mathbf{\Sigma}_{n}^{-1}\mathbf{\hat{\beta}}_{n}">
* <img src="https://latex.codecogs.com/svg.latex?\mathbf{\Sigma}_{n}^{-1}=\mathbf{X}_{n}^\text{T}\mathbf{X}_{n}+\lambda\mathbf{I}">
* <img src="https://latex.codecogs.com/svg.latex?\mathbf{\hat{\beta}}_{n}=\mathbf{\Sigma}_{n}\mathbf{X}_{n}^\text{T}\mathbf{y}">

When we sample from the distibutions of our model parameters ![](https://latex.codecogs.com/svg.latex?\mathbf{\beta}) using a multivariate normal with means ![](https://latex.codecogs.com/svg.latex?\mathbf{\hat{\beta}}) and covariance matrix ![](https://latex.codecogs.com/svg.latex?\mathbf{\Sigma}R/n_\text{d.o.f.}) where ![](https://latex.codecogs.com/svg.latex?n_{\text{d.o.f.}}) is the number of degrees of freedom, or the number of updates ![](https://latex.codecogs.com/svg.latex?n) minus the number of features. The update rules for the sum of squared residuals arises from linear algebra identities and the definition <img src="https://latex.codecogs.com/svg.latex?R_{n}=(\mathbf{y}-\mathbf{\hat{\beta}}^\text{T}\mathbf{X})^\text{T}_{n}(\mathbf{y}-\mathbf{\hat{\beta}}^\text{T}\mathbf{X})_{n}"> 

Using these

