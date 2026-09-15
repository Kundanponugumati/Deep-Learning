import numpy as np

# ============================================================
# 1. DATASET
# ============================================================

X = np.array([
    [1, 2],
    [2, 1],
    [2, 3],
    [3, 2],
    [4, 5],
    [5, 4],
    [5, 6],
    [6, 5]
])

y = np.array([
    0,
    0,
    0,
    0,
    1,
    1,
    1,
    1
])
# before y shape is (8,)
# Make y shape (8, 1)
y = y.reshape(-1, 1)


# ============================================================
# 2. INITIALIZE PARAMETERS
# ============================================================

# Hidden layer
# Input = 2 features
# Hidden layer = 4 neurons

W1 = np.random.randn(2, 4)
b1 = np.zeros((1, 4))

# Output layer
# Hidden layer = 4 neurons
# Output = 1 neuron

W2 = np.random.randn(4, 1)
b2 = np.zeros((1, 1))


# ============================================================
# 3. ACTIVATION FUNCTIONS
# ============================================================

def relu(z):
    return np.maximum(0, z)


def relu_derivative(z):
    return (z > 0).astype(float)


def sigmoid(z):
    return 1 / (1 + np.exp(-z))


# ============================================================
# 4. TRAINING
# ============================================================

learning_rate = 0.1
epochs = 1000

N = len(X)


for epoch in range(epochs):

    # --------------------------------------------------------
    # FORWARD PROPAGATION
    # --------------------------------------------------------

    # Hidden layer
    Z1 = X @ W1 + b1
    A1 = relu(Z1)

    # Output layer
    Z2 = A1 @ W2 + b2
    A2 = sigmoid(Z2)


    # --------------------------------------------------------
    # LOSS
    # --------------------------------------------------------

    loss = -np.mean(
        y * np.log(A2 + 1e-8) +
        (1 - y) * np.log(1 - A2 + 1e-8)
    )


    # --------------------------------------------------------
    # BACKPROPAGATION
    # --------------------------------------------------------

    # Output layer

    dZ2 = A2 - y

    dW2 = (A1.T @ dZ2) / N

    db2 = np.sum(dZ2, axis=0, keepdims=True) / N


    # Send gradient backwards into hidden layer

    dA1 = dZ2 @ W2.T


    # ReLU derivative

    dZ1 = dA1 * relu_derivative(Z1)

    dW1 = (X.T @ dZ1) / N

    db1 = np.sum(dZ1, axis=0, keepdims=True) / N


    # --------------------------------------------------------
    # UPDATE PARAMETERS
    # --------------------------------------------------------

    W2 = W2 - learning_rate * dW2
    b2 = b2 - learning_rate * db2

    W1 = W1 - learning_rate * dW1
    b1 = b1 - learning_rate * db1


    # --------------------------------------------------------
    # PRINT LOSS
    # --------------------------------------------------------

    if epoch % 100 == 0:
        print(f"Epoch {epoch}, Loss: {loss}")


# ============================================================
# 5. FINAL FORWARD PASS
# ============================================================

Z1 = X @ W1 + b1
A1 = relu(Z1)

Z2 = A1 @ W2 + b2
A2 = sigmoid(Z2)


# ============================================================
# 6. PREDICTIONS
# ============================================================

predictions = (A2 >= 0.5).astype(int)


# ============================================================
# 7. RESULTS
# ============================================================

print("\n--------------------------------")
print("FINAL RESULTS")
print("--------------------------------")

print("\nProbabilities:")
print(A2.flatten())

print("\nPredictions:")
print(predictions.flatten())

print("\nActual:")
print(y.flatten())

print("\nFinal W1:")
print(W1)

print("\nFinal b1:")
print(b1)

print("\nFinal W2:")
print(W2)

print("\nFinal b2:")
print(b2)