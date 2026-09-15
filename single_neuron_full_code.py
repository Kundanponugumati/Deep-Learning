import numpy as np
import math 


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


# initiate weights with arbitary values

W = np.array([0.5,0.8])
b = 0.1
learning_rate = 0.1

def sigmoid(z):
    return 1/(1+np.exp(-z))

epochs = 10000

for epoch in range(epochs):
        z = X @ W + b
        a = sigmoid(z)

        # loss function
        loss = -np.mean(y*np.log(a)+(1-y)*np.log(1-a))

        # backpropagation
        dz = a - y
        dW = (X.T @ dz) / len(X)
        db = np.mean(dz)

        #update
        W =W - learning_rate*dW
        b = b - learning_rate*db

        # Print progress
        if epoch % 100 == 0:
            print(f"Epoch {epoch}, Loss: {loss}")

# final W and b 

print("\nFinal W:", W)
print("Final b:", b)

# Final forward pass
z = X @ W + b
a = sigmoid(z)

# Convert probabilities to classes
predictions = (a >= 0.5).astype(int)

print("\nProbabilities:")
print(a)

print("\nPredictions:")
print(predictions)

print("\nActual:")
print(y)