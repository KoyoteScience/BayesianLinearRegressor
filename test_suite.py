api_key = '<CONTACT INFO@KOYOTESCIENCE.COM FOR A DEMONSTRATION API KEY>'
opt_run_bandit = True
opt_print = True
trailing_test = False
opt_big_test = False

if trailing_test:
    if opt_big_test:
        moving_window_size = 24 # same as len(list(itertools.permutations([1, 2, 3, 4])))
    else:
        moving_window_size = 6  # same as len(list(itertools.permutations([1, 2, 3])))
    model_type = 'TrailingBayesianLinearRegression'
else:
    moving_window_size = 1e6  # large number we won't encounter
    model_type = 'BayesianLinearRegression'

#########################################

from collections import namedtuple
import time
from src.bandito import *
from src.bayes_linear_regressor import *
import random
import numpy as np
import statsmodels.api as sm
import itertools


def safe_matrix_inversion(m, tikhonov=1e-6):
    m += np.eye(len(m)) * tikhonov
    a, b = m.shape
    if a != b:
        raise ValueError("Only square matrices are invertible.")
    i = np.eye(a, a)
    return np.linalg.lstsq(m, i)[0]


class ModelType(Enum):
    def __str__(self):
        return self.value

    SGDRegressor = 'SGDRegressor'
    BayesianLinearRegression = 'BayesianLinearRegression'
    TrailingBayesianLinearRegression = 'TrailingBayesianLinearRegression'
    AverageCategoryMembership = 'AverageCategoryMembership'


# from https://www.kite.com/python/answers/how-to-convert-a-dictionary-into-an-object-in-python
def convertDictToObj(tmp_dict):
    return namedtuple("ObjectName", tmp_dict.keys())(*tmp_dict.values())


if opt_big_test:
    n_features = 4
    list_of_input_vectors = list(list(x) for x in itertools.permutations([1, -2, 3, -4]))
    random.shuffle(list_of_input_vectors)
    list_of_output_values = [input_vector[0] * 1 +
                             input_vector[1] * -2 +
                             input_vector[2] * 3 +
                             input_vector[3] * -4 for input_vector in list_of_input_vectors]
else:
    n_features = 3
    list_of_input_vectors = list(list(x) for x in itertools.permutations([1, -2, 3]))
    random.shuffle(list_of_input_vectors)
    list_of_output_values = [input_vector[0] * 1 +
                         input_vector[1] * -2 +
                         input_vector[2] * 3 for input_vector in list_of_input_vectors]
list_of_input_vectors *= 2
list_of_output_values *= 2
sequence_of_input_vectors_and_output_values = []

for i in range(len(list_of_input_vectors)):
    for j in range(len(list_of_input_vectors[0])):
        list_of_input_vectors[i][j] += (np.random.random() - 0.5) * 0.1
    list_of_output_values[i] += (np.random.random() - 0.5) * 0.1

for i in range(len(list_of_input_vectors)):
    sequence_of_input_vectors_and_output_values.append({
        'input_vector': list_of_input_vectors[i],
        'output_value': list_of_output_values[i]
    })

example_pull_payload = {
    'api_key': api_key,
    'model_id': 'app_id=internal_testing&test_name=variance',
    # this setting doesn't matter for the test, since we are checking the pull responses, not the train responses
    'model_type': 'SGDRegressor',
    'predict_on_all_models': False,
    'feature_metadata': [{
        'name': f'doesnt_matter_{i}',
        'categorical_or_continuous': 'continuous'
    } for i in range(n_features)],
    'trailing_list_length': moving_window_size
}

if opt_run_bandit:
    bandit = banditoAPI(**example_pull_payload)
    bandit

    restart_response = bandit.restart()
    restart_response = convertDictToObj(restart_response)
    restart_response

train_in_batch = False
if train_in_batch:
    train_response = bandit.train(
        [tmp_dict['input_vector'] for tmp_dict in sequence_of_input_vectors_and_output_values],
        [tmp_dict['output_value'] for tmp_dict in sequence_of_input_vectors_and_output_values]
    )
    train_response = convertDictToObj(train_response)
    try:
        print(f'success: progress: {train_response.progress}')
    except:
        print(f'ERROR:\n{train_response.stackTrace}')
else:

    bayes_linear_regression = BayesLinearRegressor(n_features)
    indices_in_moving_window = []

    for i, tmp_dict in enumerate(sequence_of_input_vectors_and_output_values[:]):
        input_vector = tmp_dict['input_vector']
        output_value = tmp_dict['output_value']

        if opt_print:
            print(f'\n========== Input vector index #{i}============\n')

        if opt_run_bandit:
            train_response = bandit.train(input_vector, output_value)
            train_response = convertDictToObj(train_response)
            try:
                print(f'success: {i} progress: {train_response.progress}')
                print(f'input feature vector: {sequence_of_input_vectors_and_output_values[i]}')
            except:
                print(train_response.stackTrace)

        X = np.array([val['input_vector'] for val in
                      sequence_of_input_vectors_and_output_values[max(0, i - moving_window_size + 1):i + 1]])
        y = np.array([val['output_value'] for val in
                      sequence_of_input_vectors_and_output_values[max(0, i - moving_window_size + 1):i + 1]])
        X_with_intercept = np.array([[1] + list(row) for row in X])

        model = sm.OLS(y, X_with_intercept).fit()
        predictions = model.predict(X_with_intercept)

        indices_in_moving_window.append(i)
        bayes_linear_regression.partial_fit(
            np.array([tmp_dict['input_vector']]),
            np.array([[tmp_dict['output_value']]])
        )

        if i - moving_window_size >= 0:
            indices_in_moving_window.remove(i - moving_window_size)
            unfit_tmp_dict = sequence_of_input_vectors_and_output_values[i - moving_window_size]
            bayes_linear_regression.partial_unfit(
                np.array([unfit_tmp_dict['input_vector']]),
                np.array([[unfit_tmp_dict['output_value']]])
            )

        if opt_run_bandit:
            pull_response = bandit.pull(
                [tmp_dict['input_vector'] for tmp_dict in sequence_of_input_vectors_and_output_values],
                model_type=ModelType[model_type],
                deterministic=True
            )
            pull_response = convertDictToObj(pull_response)

        if opt_print:
            print('X:')
            print(X)

            print('BayesLinearRegressor coeffs:')
            coeffs = np.array([bayes_linear_regression.intercept_] + bayes_linear_regression.coef_.tolist())
            print(coeffs)
            print(
                f'  giving prediction residual_sum_squares: {sum(x ** 2 for x in (coeffs @ X_with_intercept.T - y).tolist())}')

            if opt_run_bandit:
                print('Bandit coeffs:')
                coeffs = [pull_response.deterministic_model_intercept] + pull_response.deterministic_model_coefficients
                print(coeffs)
                print(
                    f'  giving prediction residual_sum_squares: {sum(x ** 2 for x in coeffs @ X_with_intercept.T - y)}')

            print('Linear Algebra coeffs w/o covariance matrices:')
            coeffs = safe_matrix_inversion(
                np.array(X_with_intercept.T @ X_with_intercept, dtype=np.float)) @ X_with_intercept.T @ y
            print(coeffs)
            print(f'  giving prediction residual_sum_squares: {sum(x ** 2 for x in coeffs @ X_with_intercept.T - y)}')

            print('Linear Algebra coeffs (2):')
            try:
                coeffs = (safe_matrix_inversion(np.cov(X.T)) @ np.cov(X.T, y)[:-1, -1]).tolist()
                residuals = np.array(coeffs) @ X.T - y
                intercept = -np.mean(residuals)
                print([np.nan] + coeffs)
                print(f'  giving prediction residual_sum_squares: {sum(x ** 2 for x in residuals + intercept)}')
            except:
                print('whatevs')

            print('Linear Algebra coeffs (3, stabler):')
            try:
                # NB: doesn't matter if you use X or X_with_intercept
                coeffs = (np.linalg.lstsq(np.cov(X.T), np.cov(X.T, y)[:-1, -1])[0]).tolist()
                orig_residuals = np.array(coeffs) @ X.T - y
                intercept = -np.mean(orig_residuals)
                residuals = orig_residuals + intercept
                print([np.nan] + coeffs)
                print(f'  giving prediction residual_sum_squares: {sum(x ** 2 for x in residuals)}')
            except:
                print('whatevs')

            print('Linear Algebra coeffs from bandit covariance matrix:')
            try:
                coeffs = \
                    safe_matrix_inversion(
                        np.array(pull_response.covariance_matrix, dtype=np.float)[:-1, :-1] / float(i)
                    ) @ \
                    np.array(pull_response.covariance_matrix, dtype=np.float)[:-1, -1] / float(i)
                residuals = np.array(coeffs) @ X.T - y
                intercept = -np.mean(residuals)
                print(coeffs)
                print(f'  giving prediction residual_sum_squares: {sum(x ** 2 for x in residuals + intercept)}')
            except:
                print('whatevs')

            print('Statsmodels coeffs:')
            coeffs = model.params
            print(coeffs)
            print(f'  standard errors: {model.bse}')
            print(f'  giving prediction residual_sum_squares: {sum(x ** 2 for x in coeffs @ X_with_intercept.T - y)}')

            print('BayesianLinearRegression moment matrix:')
            print(bayes_linear_regression.moment_of_X)

            if opt_run_bandit:
                print('Bandit moment matrix:')
                print(np.array(pull_response.moment_matrix, dtype=np.float)[:-1, :-1])

            print('Linear Algebra moment matrix:')
            print(X_with_intercept.T @ X_with_intercept)

            if opt_run_bandit:
                print('Bandit covariance matrix:')
                print(np.array(pull_response.covariance_matrix, dtype=np.float)[:-1, :-1] / float(i))

            print('Linear Algebra covariance matrix:')
            print(np.cov(X.T))

            print('BayesLinearRegresso parameter covariance matrix:')
            residual_sum_squares = bayes_linear_regression.residual_sum_squares
            scale_multiplier = 1.0 / max(1.0, (len(X) - len(X[0]) - 1))
            print(bayes_linear_regression.cov_params * residual_sum_squares * scale_multiplier)

            if opt_run_bandit:
                print('Bandit parameter covariance matrix:')
                try:
                    residual_sum_squares = float(pull_response.residual_sum_squares)
                    scale_multiplier = 1.0 / max(1.0, (len(X) - len(X[0]) - 1))
                    moment_of_X = np.array(pull_response.moment_matrix, dtype=np.float)[:-1, :-1]
                    print(safe_matrix_inversion(moment_of_X) * residual_sum_squares * scale_multiplier)
                except Exception as e:
                    print(f'Error!: {e}')

            print('Linear Algebra parameter covariance matrix:')
            try:
                coeffs = safe_matrix_inversion(
                    np.array(X_with_intercept.T @ X_with_intercept, dtype=np.float)) @ X_with_intercept.T @ y
                residuals = np.array(coeffs) @ X_with_intercept.T - y
                residual_sum_squares = float(sum([x ** 2 for x in residuals]))
                scale_multiplier = 1.0 / max(1.0, (len(X) - len(X[0]) - 1))
                covariance_matirx = safe_matrix_inversion(np.array(X_with_intercept.T @ X_with_intercept,
                                                                   dtype=np.float))
                print(covariance_matirx * residual_sum_squares * scale_multiplier)
                # print(safe_matrix_inversion(np.array(np.cov(X.T), dtype=np.float)) * sigma_squared_hat)
            except Exception as e:
                print(f'Error!: {e}')

            print('Statsmodels parameter covariance matrix')
            print(model.cov_params())

            print(
                f'Bayes Linear Regression residual_sum_squares approx: {float(bayes_linear_regression.residual_sum_squares / len(residuals))}')
            if opt_run_bandit:
                print(
                    f'Bandit residual_sum_squares approx: {float(pull_response.residual_sum_squares / len(residuals))}')
            print(f'Linear Algebra residual_sum_squares: {float(sum([x ** 2 for x in residuals]) / len(residuals))}')
