import numpy as np

class BayesLinearRegressor:
    
    def __init__(self, number_of_features, alpha=1e6, diagonal_covariance=False):
        '''
        :param number_of_features: Integer number of features in the training rows, excluding the intercept and output values 
        :param alpha: Float inverse ridge regularizaiton constant, set to 1e6
        :param diagonal_covariance: Boolean, True means that we assume the covariance matrix is diagonal
          i.e., there is no correlation between our inputs. TODO: Further computational complexity reduction for this case.
        '''
        
        # alpha is our initial guess on the variance, basically, all parameters initialized to 0 with alpha variance
        #   so, you know, just set it super-high. This is the same as L2 regularization, btw!
        # all those weird Bayesian update rules actually amount to very standard linear algebra identities
        #   Once you see that it's just updating the moment matrix and the sum of squared residuals, it's straightforward!
        #   So those are our internal variables that everything else depends upon
        
        self.number_of_features = number_of_features
        self.alpha = alpha
        self.beta_means = np.array([[0] * (number_of_features + 1)],dtype=np.float).T # + 1 for the intercept
        self.number_of_updates = 0
        self.residual_sum_squares = 0
        self.diagonal_covariance = diagonal_covariance
        if diagonal_covariance:
            self.moment_of_X = [0] * (number_of_features + 1) # not a matrix but a vector, + 1 for the intercept
            self.moment_of_X_and_Y = [0] * (number_of_features + 1) # not a matrix but a vector, + 1 for the intercept
        else:
            self.moment_matrix = np.eye(number_of_features + 2) * 0.0 # + 2 for the intercept and the output
        
    def partial_fit(self, X, Y, W=None, reverse=False):
        '''
        The online updating rules
        :param X: Input feature vector(s) as 2D numpy array 
        :param Y: Input output values as 1D numpy array
        :param W: Data weights (relative to unity) as a 1D numpy array
        :param reverse: Boolean, True means that we "unfit" the training rows, otherwise acts as normal
        :return: None
        '''
        
        # clear the frozen parameter sample since we are updating the parameter distributions
        self.frozen_parameter_sample = None 
        
        if self.diagonal_covariance:

            moment_of_X_before = self.moment_matrix[:-1]
            beta_means_before = self.beta_means.copy()
            inverted_variance_vector_before = moment_of_X_before + float(1 / self.alpha)

            X_with_intercept = np.array([[1] + row for row, output in X.tolist()])
            
            if W is None:
                moment_of_X_update_term = np.sum(X_with_intercept ** 2, axis=0)
                moment_of_X_and_Y_update_term = X_with_intercept @ Y
            else:
                moment_of_X_update_term = np.sum(np.array([vec ** 2 * W[i] for i, vec in rank_n_obs_update_matrix]), axis=0)
                moment_of_X_and_Y_update_term = X_with_intercept @ np.diag(W.tolist()) @ Y

            if not reverse:
                self.moment_of_X += moment_of_X_update_term
                self.moment_of_X_and_Y += moment_of_X_and_Y_update_term
                moment_of_Y_update_term = Y.T @ Y
                self.number_of_updates += 1
            else:
                self.moment_of_X -= moment_of_X_update_term
                self.moment_of_X_and_Y -= moment_of_X_and_Y_update_term
                moment_of_Y_update_term = -Y.T @ Y
                self.number_of_updates -= 1

            inverted_variance_vector = self.moment_of_X + float(1 / self.alpha)
            covariance_matrix = inverted_variance_vector ** -1

            # Note that the matrix multiplication operators are now replaced by element-wise multiplication
            self.beta_means = covariance_matrix * moment_of_X_and_Y
            if self.number_of_updates > len(covariance_matrix) - 1:
                self.residual_sum_squares += (
                        moment_of_Y_update_term -
                        self.beta_means ** 2 @ inverted_variance_vector +
                        beta_means_before ** 2 @ inverted_variance_vector_before
                )
        else:
            moment_of_X_before = self.moment_matrix[:-1, :-1]
            beta_means_before = self.beta_means.copy()
            inverted_covariance_matrix_before = moment_of_X_before + np.eye(self.number_of_features + 1) / self.alpha

            # Here we concatenate the intercept input value (constant 1), the input vector, and the output value:
            rank_n_obs_update_matrix = np.array([[1] + row + output for row, output in zip(X.tolist(), Y.tolist())])
            
            if W is None:
                moment_matrix_update_term = rank_n_obs_update_matrix.T @ rank_n_obs_update_matrix
            else:
                moment_matrix_update_term = rank_n_obs_update_matrix.T @ np.diag(W.tolist()) @ rank_n_obs_update_matrix
                
            if not reverse:
                self.moment_matrix += moment_matrix_update_term
                moment_of_Y_update_term = Y.T @ Y
                self.number_of_updates += 1
            else:
                self.moment_matrix -= moment_matrix_update_term
                moment_of_Y_update_term = -Y.T @ Y
                self.number_of_updates -= 1
            
            moment_of_X = self.moment_matrix[:-1, :-1]
            moment_of_X_and_Y = self.moment_matrix[:-1, -1]
            inverted_covariance_matrix = moment_of_X + np.eye(self.number_of_features + 1) / self.alpha
            covariance_matrix = np.linalg.inv(inverted_covariance_matrix)
            
            self.beta_means = covariance_matrix @ moment_of_X_and_Y
            if self.number_of_updates > len(covariance_matrix) - 1:
                self.residual_sum_squares += (
                        moment_of_Y_update_term -
                        self.beta_means.T @ inverted_covariance_matrix @ self.beta_means +
                        beta_means_before.T @ inverted_covariance_matrix_before @ beta_means_before
                )
        
    def partial_unfit(self, X, Y):
        return self.partial_fit(X, Y, reverse=True)
        
    def predict(self, X, use_means=False, freeze_parameter_sample=False):
        '''
        :param X: Input feature vector excluding the intercept constant as a 2D numpy array 
        :param use_means: Boolean where True means we just provide the prediciton at the mean of the coefficients
            (sometimes referred to as deterministic prediction); otherwise sample parameters from the multivariate norm
            and incorporate the uncertainty of the parameters in your prediction
        :param freeze_parameter_sample: Boolean. When set to True, we sample from the parameters only once for each prediction
        :return: 
        '''

        scale_multiplier = 1.0 / max(1.0, (self.number_of_updates - self.number_of_features - 1))
        
        if use_means:
            return self.beta_means.T @ X.T
        else:
            if freeze_parameter_sample:
                if self.frozen_parameter_sample is not None:
                    self.frozen_parameter_sample =  np.random.multivariate_normal(
                        self.beta_means.T[0], 
                        self.residual_sum_squares * scale_multiplier * self.cov
                    )
                beta = self.frozen_parameter_sample
            else:
                beta = np.random.multivariate_normal(
                    self.beta_means.T[0], 
                    self.residual_sum_squares * scale_multiplier * self.cov_params
                )
            return beta.T @ X.T
        
    @property
    def coef_(self):
        return self.beta_means[1:].T
    
    @property
    def intercept_(self):
        return self.beta_means[0]

    @property
    def cov_params(self):
        if self.diagonal_covariance:
            return np.array((self.moment_of_X + float(1 / self.alpha))) ** -1
        else:    
            moment_of_X = self.moment_matrix[:-1,:-1]
            inverted_covariance_matrix = moment_of_X + np.eye(self.number_of_features + 1) / self.alpha
            return np.linalg.inv(inverted_covariance_matrix)
        
