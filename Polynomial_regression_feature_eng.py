import numpy as np
import matplotlib.pyplot as plt
import math

def zscore_normalize_features(X):
    mu = np.mean(X, axis=0)
    sigma = np.std(X, axis=0)
    X_norm = (X - mu) / sigma
    return X_norm

def compute_cost(x, y, w, b):
    m = x.shape[0]
    cost = 0.0
    for i in range(m):
        cost += (np.dot(x[i], w) + b - y[i])**2
    cost /= 2*m
    return cost

def compute_gradient(x, y, w, b):
    m, n = x.shape
    dj_w = np.zeros((n,))
    dj_b = 0.0
    for i in range(m):
        base = (np.dot(x[i], w) + b) - y[i]
        for j in range(n):
            dj_w[j] += base * x[i][j]
        dj_b += base
    dj_w /= m
    dj_b /= m
    return dj_b, dj_w

def run_gradient(x, y, iterations, alpha):
    w = np.zeros(x.shape[1])
    b = 0.0
    cost_history = []

    plt.ion()  # Włącz tryb interaktywny
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12,5))

    for iter in range(iterations):
        dj_b, dj_w = compute_gradient(x, y, w, b)
        w -= alpha * dj_w
        b -= alpha * dj_b

        # Liczymy i zapisujemy koszt
        cost = compute_cost(x, y, w, b)
        cost_history.append(cost)

        if iter % 100 == 0 or iter == iterations-1:  # Rysuj co kilka iteracji, aby było płynniej
            ax1.clear()
            ax2.clear()

            # Wykres funkcji przybliżonej
            y_pred = np.dot(x, w) + b
            ax1.plot(np.arange(len(y)), y, label="Prawdziwe y", marker='o')
            ax1.plot(np.arange(len(y_pred)), y_pred, label="Przybliżone y", marker='x')
            ax1.set_title("Przybliżenie funkcji")
            ax1.legend()
            ax1.grid(True)

            # Wykres kosztu
            ax2.plot(cost_history, label="Koszt")
            ax2.set_title("Zmiana kosztu")
            ax2.set_xlabel("Iteracja")
            ax2.set_ylabel("Koszt")
            ax2.grid(True) 

            plt.pause(0.1)
        if iter%1000 == 0: print(cost, iter)
    plt.ioff()  # Wyłącz tryb interaktywny
    plt.show()

# Dane
x = np.arange(0, 20, 0.5)
y = np.cos(x/2)
X = np.c_[x,x**2, x**3,x**4, x**5, x**6, x**7, x**8, x**9, x**10, x**11, x**12, x**13, x**14, x**15, x**16]
X = zscore_normalize_features(X)
run_gradient(X, y, iterations=10000000, alpha=0.05)
