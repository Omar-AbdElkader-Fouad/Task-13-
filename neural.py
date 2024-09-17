import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import fetch_openml
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split

# Load MNIST dataset
mnist = fetch_openml('mnist_784')
X, y = mnist['data'], mnist['target'].astype(int)

# Preprocess data
scaler = StandardScaler()
X = scaler.fit_transform(X)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Neural Network parameters
input_size = 784  # 28x28 images
hidden_size = 64
output_size = 10  # digits 0-9
learning_rate = 0.01
epochs = 10

# Initialize weights
def init_weights(input_size, hidden_size, output_size):
    W1 = np.random.randn(input_size, hidden_size) * 0.01
    b1 = np.zeros((1, hidden_size))
    W2 = np.random.randn(hidden_size, output_size) * 0.01
    b2 = np.zeros((1, output_size))
    return W1, b1, W2, b2

# Forward pass
def forward(X, W1, b1, W2, b2):
    z1 = np.dot(X, W1) + b1
    a1 = np.maximum(0, z1)  # ReLU activation
    z2 = np.dot(a1, W2) + b2
    exp_scores = np.exp(z2 - np.max(z2, axis=1, keepdims=True))
    probs = exp_scores / np.sum(exp_scores, axis=1, keepdims=True)
    return probs, a1

# Loss function
def compute_loss(probs, y):
    correct_logprobs = -np.log(probs[np.arange(len(y)), y])
    loss = np.sum(correct_logprobs) / len(y)
    return loss

# Backward pass
def backward(X, y, probs, a1, W1, W2):
    delta3 = probs
    delta3[np.arange(len(y)), y] -= 1
    dW2 = np.dot(a1.T, delta3) / len(y)
    db2 = np.sum(delta3, axis=0, keepdims=True) / len(y)
    delta2 = np.dot(delta3, W2.T)
    delta2[a1 <= 0] = 0
    dW1 = np.dot(X.T, delta2) / len(y)
    db1 = np.sum(delta2, axis=0, keepdims=True) / len(y)
    return dW1, db1, dW2, db2

# Training
def train(X_train, y_train):
    W1, b1, W2, b2 = init_weights(input_size, hidden_size, output_size)
    for epoch in range(epochs):
        probs, a1 = forward(X_train, W1, b1, W2, b2)
        loss = compute_loss(probs, y_train)
        dW1, db1, dW2, db2 = backward(X_train, y_train, probs, a1, W1, W2)
        W1 -= learning_rate * dW1
        b1 -= learning_rate * db1
        W2 -= learning_rate * dW2
        b2 -= learning_rate * db2
        if epoch % 1 == 0:
            print(f'Epoch {epoch}, Loss: {loss}')
    return W1, b1, W2, b2

# Test accuracy
def predict(X, W1, b1, W2, b2):
    probs, _ = forward(X, W1, b1, W2, b2)
    return np.argmax(probs, axis=1)

W1, b1, W2, b2 = train(X_train, y_train)
y_pred = predict(X_test, W1, b1, W2, b2)
accuracy = np.mean(y_pred == y_test)
print(f'Test accuracy: {accuracy}')
