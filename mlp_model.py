import numpy as np

X_train = np.load("data/X_train.npy")
y_train = np.load("data/y_train.npy")

X_test = np.load("data/X_test.npy")
y_test = np.load("data/y_test.npy")

#this is ReLU which is Rectified Linear Unit
#it returns 0 if the input is -ve, and returns the original value if the input is +ve
#old school method was sigmoid function but this how now we do it
def relu(Z):
    return np.maximum(0, Z)

#returns 1 is positive and 0 is negative, basically neuron is active if gradient flows
#if it dont then gradient is zero
#so a list of true or false is made
def relu_derivative(Z):
    return (Z > 0).astype(float)

def initialize_parameters(input_dim, hidden1, hidden2, output_dim):

    np.random.seed(42)
    #first layer
    W1 = np.random.randn(input_dim, hidden1) * 0.1
    b1 = np.zeros((1, hidden1))
    #secoind layer
    W2 = np.random.randn(hidden1, hidden2) * 0.1
    b2 = np.zeros((1, hidden2))
    #third layer
    W3 = np.random.randn(hidden2, output_dim) * 0.1
    b3 = np.zeros((1, output_dim))

    return W1, b1, W2, b2, W3, b3


def forward(X, W1, b1, W2, b2, W3, b3):

    Z1 = X @ W1 + b1
    A1 = relu(Z1)

    Z2 = A1 @ W2 + b2
    A2 = relu(Z2)

    Z3 = A2 @ W3 + b3
    output = Z3  # regression, no activation

    return Z1, A1, Z2, A2, Z3, output

#loss calc. summation((diff)^2)
def mse(y_true, y_pred):
    return np.mean((y_true - y_pred) ** 2)

#backpropagation
def backward(X, y, Z1, A1, Z2, A2, Z3, output, W2, W3):

    m = X.shape[0]

    dZ3 = (output - y) * (2 / m)
    dW3 = A2.T @ dZ3
    db3 = np.sum(dZ3, axis=0, keepdims=True)

    dA2 = dZ3 @ W3.T
    dZ2 = dA2 * relu_derivative(Z2)
    dW2 = A1.T @ dZ2
    db2 = np.sum(dZ2, axis=0, keepdims=True)

    dA1 = dZ2 @ W2.T
    dZ1 = dA1 * relu_derivative(Z1)
    dW1 = X.T @ dZ1
    db1 = np.sum(dZ1, axis=0, keepdims=True)

    return dW1, db1, dW2, db2, dW3, db3


def r2_score(y_true, y_pred):
    ss_res = np.sum((y_true - y_pred) ** 2)
    ss_tot = np.sum((y_true - np.mean(y_true)) ** 2)
    return 1 - (ss_res / ss_tot)

#training
input_dim = X_train.shape[1]
hidden1 = 32
hidden2 = 16
output_dim = 1

W1, b1, W2, b2, W3, b3 = initialize_parameters(
    input_dim, hidden1, hidden2, output_dim
)

learning_rate = 0.01
epochs = 2000

for epoch in range(epochs):

    Z1, A1, Z2, A2, Z3, output = forward(
        X_train, W1, b1, W2, b2, W3, b3
    )

    loss = mse(y_train, output)

    dW1, db1, dW2, db2, dW3, db3 = backward(
        X_train, y_train, Z1, A1, Z2, A2, Z3, output, W2, W3
    )

    W1 -= learning_rate * dW1
    b1 -= learning_rate * db1

    W2 -= learning_rate * dW2
    b2 -= learning_rate * db2

    W3 -= learning_rate * dW3
    b3 -= learning_rate * db3

    if epoch % 200 == 0:
        print(f"Epoch {epoch}, Loss: {loss}")


#yeah this shit is crazy but i wrote it, belive me ;)
_, _, _, _, _, train_pred = forward(
    X_train, W1, b1, W2, b2, W3, b3
)

_, _, _, _, _, test_pred = forward(
    X_test, W1, b1, W2, b2, W3, b3
)

print("\nMLP Results")
print("Train MSE:", mse(y_train, train_pred))
print("Test MSE:", mse(y_test, test_pred))
print("Train R2:", r2_score(y_train, train_pred))
print("Test R2:", r2_score(y_test, test_pred))


# Epoch 0, Loss: 3.308819905929494
# Epoch 200, Loss: 0.6128110168340444
# Epoch 400, Loss: 0.48408384936479354
# Epoch 600, Loss: 0.4366741996878659
# Epoch 800, Loss: 0.4098742978092883
# Epoch 1000, Loss: 0.3891734723185518
# Epoch 1200, Loss: 0.368841692971834
# Epoch 1400, Loss: 0.3500475024310718
# Epoch 1600, Loss: 0.3347781042073911
# Epoch 1800, Loss: 0.32240068054338106
#
# MLP Results
# Train MSE: 0.31248382361105626
# Test MSE: 0.0913850919723404
# Train R2: 0.5633387334955491
# Test R2: 0.7919125040478018
