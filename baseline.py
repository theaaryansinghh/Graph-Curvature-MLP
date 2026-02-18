"""
Why this baseline exists:

Before using a neural network, we need to check whether the problem
is already well explained by a linear model.

If a simple linear regression achieves high R², then the relationship
between structural graph features and Gromov delta is mostly linear,
and a neural network would add unnecessary complexity.

If the linear model performs moderately or poorly, it suggests that
nonlinear interactions between features are important.

So this baseline justifies the use of a nonlinear model like an MLP.

In our case, the linear model achieves~0.45 test R^2,
which means there is structure, but it is not purely linear.
"""


import numpy as np
X_train = np.load("data/X_train.npy")
y_train = np.load("data/y_train.npy")

X_test = np.load("data/X_test.npy")
y_test = np.load("data/y_test.npy")

def add_bias(X):
    ones = np.ones((X.shape[0], 1))
    return np.hstack((ones, X))

X_train_bias = add_bias(X_train)
X_test_bias = add_bias(X_test)

# w = (X^T X)^(-1) X^T y

XtX = X_train_bias.T @ X_train_bias
XtX_inv = np.linalg.inv(XtX)
Xt_y = X_train_bias.T @ y_train

w = XtX_inv @ Xt_y

y_pred_train = X_train_bias @ w
y_pred_test = X_test_bias @ w

def mse(y_true, y_pred):
    return np.mean((y_true - y_pred) ** 2)

def r2_score(y_true, y_pred):
    ss_res = np.sum((y_true - y_pred) ** 2)
    ss_tot = np.sum((y_true - np.mean(y_true)) ** 2)
    return 1 - (ss_res / ss_tot)

train_mse = mse(y_train, y_pred_train)
test_mse = mse(y_test, y_pred_test)

train_r2 = r2_score(y_train, y_pred_train)
test_r2 = r2_score(y_test, y_pred_test)

print("Linear Regression Baseline Results")
print("Train MSE:", train_mse)
print("Test MSE:", test_mse)
print("Train R2:", train_r2)
print("Test R2:", test_r2)

# Linear Regression Baseline Results
# Train MSE: 0.40121404812275957
# Test MSE: 0.23922864489096354
# Train R2: 0.43934814811174383
# Test R2: 0.4552668427530243
