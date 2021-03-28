# BayesianLinearRegression

## Example Usage:

```python
    bayes_linear_regression = BayesLinearRegressor(n_features)
    
    for input_vector, output_value in zip(list_of_input_vectors, list_of_output_values):
      bayes_linear_regression.partial_fit(input_vector, output_value)
      
    prediction = bayes_linear_regression.predict(prediction_input_vector)
```

## Introduction

Linear regressors are the workhorse of data science and machine learning. Nearly every problem can use a linear regression as a baseline, and they can be used in tandem with other algorithms, such as deep learning neural networks and gradient-boosted trees, to add prediction uncertainties and to provide an ideal balance of computational efficiency and interpretability. This is in large part due to the fact that linear regressions are shallow neural networks -- literally, neural nets with only one layer. This limitation means that their solutions are analytically solvable, making linear regressions ridiculously efficient and easy to work with, and offering us access to exact prediction uncertainties without needing to resort to expensive Bayesian inferencing techniques. Either as a standard candle to confirm other Bayesian engines such as bootstrapping for advanced models, or as the engine and model itself, these properties make the Bayesian linear regressor a critical element in applications for sequential decision processes like games, recommendation systems, and interface optimization.

However, our efforts to find a pre-made reliable package in Python to implement a Bayesian Linear Regressor came up short. Scikit-learn is the most trustworthy and tested, but doesn't offer an estimate of the parameter covariance matrix, meaning that we lose prediction uncertainties which are critical in the bandit setting. Statsmodels offers these uncertainty estimates, but is far slower. Both packages store all the training data in memory. Other repositories on Github that copy the key equations from Christopher Bishop's "Pattern Recognition and Machine Learning" require that the noise parameter (the sum of squared residuals) be added as a tunable hyper parameter because that's what he did, and it breaks the ability to generalize those packages to new data sets.

## Description of Repository

For these reasons, we provide a minimal (<200 lines of code) and complete Python implementation of a Bayesian Linear Regression / Bayesian Linear Regressor, made to mimic the relevant methods in statsmodels and scikit-learn. No parameters need to be set, although the ridge regularization constant is available to change beyond the default value. All internal parameters are learned in an online/streaming/incremental/sequential fashion, including the noise parameter, i.e., the sum of squared residuals divided by the number of degrees of freedom.

This reposity covers two files: **bayesian_linear_regressor.py** and **test_suite.py**. They work with the BanditoAPI (https://github.com/KoyoteScience/BanditoAPI) to test different approaches to performing a linear regression with correct uncertainties:

* Using linear algebra identities (doesn't support online/streaming/incremental/sequential)
* Using the statsmodels library (as a ground truth, and doesn't support online/streaming/incremental/sequential)
* Using the Bandito API (supports online/streaming/incremental/sequential)
* Using the included BayesianLinearRegressor module (supports online/streaming/incremental/sequential)

At Koyote Science, LLC, we used this code as the basis for larger and more stringent integration tests we use to ensure the accuracy and efficiency of Bandito.

## Update Rules for Bayesian Linear Regression
Given a set of ![](https://latex.codecogs.com/svg.latex?n_\text{obs}) observations with values ![](https://latex.codecogs.com/svg.latex?\mathbf{X}) and ![](https://latex.codecogs.com/svg.latex?\mathbf{y}), we concatenate these arrays to yield the moment matrix ![](https://latex.codecogs.com/svg.latex?\mathbf{M}=\mathbf{X}\oplus\mathbf{y}). Thus after ![](https://latex.codecogs.com/svg.latex?n) observations, we set ![](https://latex.codecogs.com/svg.latex?R_{n}) as the sum of squared residuals, ![](https://latex.codecogs.com/svg.latex?\mathbf{\hat{\beta}}_{n}) as the model coefficient means, ![](https://latex.codecogs.com/svg.latex?\Sigma_{n}) as the parameter covariance matrix, and ![](https://latex.codecogs.com/svg.latex?\lambda) as the ridge regularization constant (also known as the Tikhonov regularization constant, and we generally set it to 1e-6 expecting order unity in our input and output value distributions). The update rules are then as follows:

* <img src="https://latex.codecogs.com/svg.latex?\mathbf{M}_{n+n_{\text{obs}}}=\mathbf{M}_{n}+\mathbf{M}^\text{T}\mathbf{M}"> 
* which is equivalent to:
  * <img src="https://latex.codecogs.com/svg.latex?\mathbf{X}_{n+n_{\text{obs}}}^\text{T}\mathbf{X}_{n+n_{\text{obs}}}=\mathbf{X}_{n}^\text{T}\mathbf{X}_{n}+\mathbf{X}^{\text{T}}\mathbf{X}">
  * <img src="https://latex.codecogs.com/svg.latex?\mathbf{y}_{n+n_{\text{obs}}}^\text{T}\mathbf{y}_{n+n_{\text{obs}}}=\mathbf{y}_{n}^\text{T}\mathbf{y}_{n}+\mathbf{y}^{\text{T}}\mathbf{y}">
  * <img src="https://latex.codecogs.com/svg.latex?\mathbf{X}_{n+n_{\text{obs}}}^\text{T}\mathbf{y}_{n+n_{\text{obs}}}=\mathbf{X}_{n}^\text{T}\mathbf{y}_{n}+\mathbf{X}^{\text{T}}\mathbf{y}">
* <img src="https://latex.codecogs.com/svg.latex?R_{n+n_{\text{obs}}}=R_{n}+\mathbf{y}^{\text{T}}\mathbf{y}-\mathbf{\hat{\beta}}_{n+n_{\text{obs}}}^\text{T}\mathbf{\Sigma}_{n+n_{\text{obs}}}^{-1}\mathbf{\hat{\beta}}_{n+n_{\text{obs}}}+\mathbf{\hat{\beta}}_{n}^\text{T}\mathbf{\Sigma}_{n}^{-1}\mathbf{\hat{\beta}}_{n}">
* <img src="https://latex.codecogs.com/svg.latex?\mathbf{\Sigma}_{n}^{-1}=\mathbf{X}_{n}^\text{T}\mathbf{X}_{n}+\lambda\mathbf{I}">
* <img src="https://latex.codecogs.com/svg.latex?\mathbf{\hat{\beta}}_{n}=\mathbf{\Sigma}_{n}\mathbf{X}_{n}^\text{T}\mathbf{y}_n">

Note that we only need to update the moment matrix and the sum of squared residuals. The rest of the update rules follow downstream. We can either predict using the  model parameter means ![](https://latex.codecogs.com/svg.latex?\mathbf{y}=\mathbf{\hat{\beta}}^\text{T}\mathbf{X}), or we can sample from the distibutions of our model parameters ![](https://latex.codecogs.com/svg.latex?\mathbf{\beta}) using a multivariate normal with those means and covariance matrix ![](https://latex.codecogs.com/svg.latex?\mathbf{\Sigma}\times&space;R\div&space;n_\text{d.o.f.}). Here, ![](https://latex.codecogs.com/svg.latex?n_{\text{d.o.f.}}=n-n_\text{features}) is the number of degrees of freedom, or the number of updates ![](https://latex.codecogs.com/svg.latex?n) minus the number of features ![](https://latex.codecogs.com/svg.latex?n_\text{features}). 

The update rules for the sum of squared residuals arises from linear algebra identities and the definition <img src="https://latex.codecogs.com/svg.latex?R_{n}=(\mathbf{y}_{n}-\mathbf{\hat{\beta}_{n}^\text{T}\mathbf{X}_{n})^\text{T}(\mathbf{y}_{n}-\mathbf{\hat{\beta}}_{n}^\text{T}\mathbf{X}_{n})">. For more information, see [this derivation](http://www.biostat.umn.edu/~ph7440/pubh7440/BayesianLinearModelGoryDetails.pdf). Note that this is the most numerically unstable calculation among the update rules, and often drifts off after the fourth decimal place (assuming all other distributions are on order unity variance). Moreover, these update should only be done once we have as many training samples as we have parameters. For small residuals, this value can even become negative. However, in this case, we can take the sum of squared residuals to be effectively zero, so we just take the absolute value when sampling from the multivariate gaussian to obtain our regression parameters in a probabilistic (bandit) setting.

Using these update rules, we can train BayesianLinearRegressor in a streaming/online/incremental/sequential fashion, meaning that at no point do we need to store more than one row of training data in memory, and we achieve the same results as using linear algebra identities or tested code like the scikit-learn and statsmodels modules, and with the full covariance matrix of the parameters (which only statsmodels offers, at a substantial computational cost). These properties make the BayesianLinearRegressor ideal for the bandit setting. Storage and computational requirements for BayesianLinearRegressor are <img src="https://latex.codecogs.com/svg.latex?O(n_\text{features}^2)"> and <img src="https://latex.codecogs.com/svg.latex?O(n\times&space;n_\text{features}^3)"> respectively, which is ideal for most cases where <img src="https://latex.codecogs.com/svg.latex?n_\text{features}<<n"> is reasonable.  Note that standard linear regressors that keep all the training data in memory enjoy lower computational complexity <img src="https://latex.codecogs.com/svg.latex?O(n\times&space;n_\text{features}^3+n_\text{features}^3)"> which is why they don't use sequential methods under the hood.

## "Unlearning" Data
The update rules only have to be slightly tweaked to allow us to "unlearn" data, making it possible to create a trailing-window linear regression, by keeping a log of the trailing input and output data and incrementally unlearning and removing the tail end of the log after a desired expiration period. The changed update rules for *unlearning* are:

* <img src="https://latex.codecogs.com/svg.latex?\mathbf{M}_{n+n_{\text{obs}}}=\mathbf{M}_{n}-\mathbf{M}"> 
* which is equivalent to:
  * <img src="https://latex.codecogs.com/svg.latex?\mathbf{X}_{n+n_{\text{obs}}}^\text{T}\mathbf{X}_{n+n_{\text{obs}}}=\mathbf{X}_{n}^\text{T}\mathbf{X}_{n}-\mathbf{X}^{\text{T}}\mathbf{X}">
  * <img src="https://latex.codecogs.com/svg.latex?\mathbf{y}_{n+n_{\text{obs}}}^\text{T}\mathbf{y}_{n+n_{\text{obs}}}=\mathbf{y}_{n}^\text{T}\mathbf{y}_{n}-\mathbf{y}^{\text{T}}\mathbf{y}">
  * <img src="https://latex.codecogs.com/svg.latex?\mathbf{X}_{n+n_{\text{obs}}}^\text{T}\mathbf{y}_{n+n_{\text{obs}}}=\mathbf{X}_{n}^\text{T}\mathbf{y}_{n}-\mathbf{X}^{\text{T}}\mathbf{y}">
* <img src="https://latex.codecogs.com/svg.latex?R_{n+n_{\text{obs}}}=R_{n}-\mathbf{y}^{\text{T}}\mathbf{y}-\mathbf{\hat{\beta}}_{n+n_{\text{obs}}}^\text{T}\mathbf{\Sigma}_{n+n_{\text{obs}}}^{-1}\mathbf{\hat{\beta}}_{n+n_{\text{obs}}}+\mathbf{\hat{\beta}}_{n}^\text{T}\mathbf{\Sigma}_{n}^{-1}\mathbf{\hat{\beta}}_{n}">
* <img src="https://latex.codecogs.com/svg.latex?(n+n_{\text{obs}})_{\text{d.o.f.}}=n_{\text{d.o.f.}}-n_\text{obs}">

## Weighted Linear Regression / Weighted Least Squares
In the bandit setting, it will also be necessary to implement weights for our training data. This occurs because in the sequential decision process, each action is selected according the outputs of a model, and therefore each action has a different propensity (or likelihood) of being selected. We need to weight the training data by the inverse of this propensity to get an accurate representation of the true data distribution when it isn't being meddled with by the bandit algorithm.

When including data weights to our training procedure, all we have to do is adjust the moment matrix update rules. We use the weight matrix <img src="https://latex.codecogs.com/svg.latex?\mathbf{W}"> which is a diagonal matrix of size <img src="https://latex.codecogs.com/svg.latex?n_\text{obs}"> with the weight for each training on the corresponding diagonal entry. This gives us the update rules:

* <img src="https://latex.codecogs.com/svg.latex?\mathbf{M}_{n+n_{\text{obs}}}=\mathbf{M}_{n}+\mathbf{M}^\text{T}\mathbf{W}\mathbf{M}"> 
* which is equivalent to:
  * <img src="https://latex.codecogs.com/svg.latex?\mathbf{X}_{n+n_{\text{obs}}}^\text{T}\mathbf{X}_{n+n_{\text{obs}}}=\mathbf{X}_{n}^\text{T}\mathbf{X}_{n}+\mathbf{X}^{\text{T}}\mathbf{W}\mathbf{X}">
  * <img src="https://latex.codecogs.com/svg.latex?\mathbf{y}_{n+n_{\text{obs}}}^\text{T}\mathbf{y}_{n+n_{\text{obs}}}=\mathbf{y}_{n}^\text{T}\mathbf{y}_{n}+\mathbf{y}^{\text{T}}\mathbf{W}\mathbf{y}">
  * <img src="https://latex.codecogs.com/svg.latex?\mathbf{X}_{n+n_{\text{obs}}}^\text{T}\mathbf{y}_{n+n_{\text{obs}}}=\mathbf{X}_{n}^\text{T}\mathbf{y}_{n}+\mathbf{X}^{\text{T}}\mathbf{W}\mathbf{y}">
