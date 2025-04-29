import numpy as np

# Logistic Regression

# Sigmoid function
def sigmoid(z):
    return 1 / (1 + np.exp(-z))

# Compute cost function
def compute_cost(X, y, w, b):
    m = X.shape[0]
    f_wb = sigmoid(np.dot(X, w) + b)
    cost = (-1 / m) * np.sum(y * np.log(f_wb + 1e-15) + (1 - y) * np.log(1 - f_wb + 1e-15))
    return cost

# Prediction function
def predict(X, w, b):
    f_wb = sigmoid(np.dot(X, w) + b)
    preds = np.where(f_wb >= 0.5, 1, 0)
    return preds

# Compute gradients
def compute_gradient(X, y, w, b):
    m = X.shape[0]
    f_wb = sigmoid(np.dot(X, w) + b)
    error = f_wb - y
    dj_dw = (1 / m) * np.dot(X.T, error)
    dj_db = (1 / m) * np.sum(error)
    return dj_db, dj_dw

# Gradient Descent Algorithm
def gradient_descent(X, y, w_in, b_in, alpha, iterations):
    J_history = []
    w = w_in.copy()
    b = b_in
    for i in range(iterations):
        dj_db, dj_dw = compute_gradient(X, y, w, b)

        w -= alpha * dj_dw
        b -= alpha * dj_db

        if i % (iterations // 100) == 0 or i == iterations-1:
            cost = compute_cost(X, y, w, b)
            J_history.append(cost)

    return w, b, J_history
