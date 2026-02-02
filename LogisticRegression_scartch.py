# ======================================
# Logistic Regression From Scratch
# Dataset: sklearn.make_classification
# ======================================

import numpy as np
from sklearn.datasets import make_classification

# -------------------------------
# 1. Generate Synthetic Dataset
# -------------------------------

X, y = make_classification(
    n_samples=500,
    n_features=5,
    n_informative=2,
    n_redundant=0,
    n_classes=2,
    random_state=42
)

y = y.reshape(-1, 1)

n_samples, n_features = X.shape

# -------------------------------
# 2. Sigmoid Function
# -------------------------------

def sigmoid(z):
    return 1 / (1 + np.exp(-z))

# -------------------------------
# 3. Binary Cross-Entropy Loss
# -------------------------------

def binary_cross_entropy(y_true, y_pred):
    epsilon = 1e-9
    y_pred = np.clip(y_pred, epsilon, 1 - epsilon)
    loss = -np.mean(
        y_true * np.log(y_pred) + (1 - y_true) * np.log(1 - y_pred)
    )
    return loss

# -------------------------------
# 4. Initialize Parameters
# -------------------------------

weights = np.zeros((n_features, 1))
bias = 0.0

# -------------------------------
# 5. Gradient Descent Training
# -------------------------------

learning_rate = 0.1
epochs = 1000

for epoch in range(epochs):

    # Forward propagation
    z = np.dot(X, weights) + bias
    y_pred = sigmoid(z)

    # Compute loss
    loss = binary_cross_entropy(y, y_pred)

    # Gradients
    dw = (1 / n_samples) * np.dot(X.T, (y_pred - y))
    db = (1 / n_samples) * np.sum(y_pred - y)

    # Update parameters
    weights -= learning_rate * dw
    bias -= learning_rate * db

    if epoch % 100 == 0:
        print(f"Epoch {epoch} | Loss: {loss:.4f}")

# -------------------------------
# 6. Prediction Function
# -------------------------------

def predict(X, weights, bias):
    z = np.dot(X, weights) + bias
    probs = sigmoid(z)
    return (probs >= 0.5).astype(int)

# -------------------------------
# 7. Model Predictions
# -------------------------------

y_pred = predict(X, weights, bias)

# -------------------------------
# 8. Accuracy Calculation
# -------------------------------

accuracy = np.mean(y_pred == y)
print("\nAccuracy:", accuracy)

# -------------------------------
# 9. Confusion Matrix (Custom)
# -------------------------------

def confusion_matrix(y_true, y_pred):
    TP = np.sum((y_true == 1) & (y_pred == 1))
    TN = np.sum((y_true == 0) & (y_pred == 0))
    FP = np.sum((y_true == 0) & (y_pred == 1))
    FN = np.sum((y_true == 1) & (y_pred == 0))

    return np.array([[TN, FP],
                     [FN, TP]])

cm = confusion_matrix(y, y_pred)

print("\nConfusion Matrix:")
print("TN FP")
print("FN TP")
print(cm)

# -------------------------------
# 10. True vs Predicted Comparison
# -------------------------------

comparison = np.hstack((y, y_pred))
print("\nTrue vs Predicted Labels (First 10 Samples)")
print("True  Pred")
print(comparison[:10])