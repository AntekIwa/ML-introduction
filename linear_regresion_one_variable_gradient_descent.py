import random
import matplotlib.pyplot as plt

n = 50

true_a = random.uniform(-5, 5)
true_b = random.uniform(-10, 10)

points_x = [random.uniform(0, 10) for _ in range(n)]
points_y = [true_a * x + true_b + random.uniform(-5, 5) for x in points_x]

def show_pts():
    plt.scatter(points_x, points_y)
    plt.title(f"Random {n} points")
    plt.xlabel("X")
    plt.ylabel("Y")
    plt.grid(True)
    plt.show()

def show_f(a, b):
    plt.clf()  
    plt.scatter(points_x, points_y, label="Points")
    x_min, x_max = min(points_x), max(points_x)
    x_linia = [x_min, x_max]
    y_linia = [a * x + b for x in x_linia]
    plt.plot(x_linia, y_linia, color='red', label=f"y = {a}x + {b:.2f}")
    plt.title(f"Random {n} points")
    plt.xlabel("X")
    plt.ylabel("Y")
    plt.legend()
    plt.grid(True)
    plt.pause(0.1) 

def detrW(w, b):
    suma = 0
    for i in range(n):
        x = points_x[i]
        y = points_y[i]
        suma += (w * x + b - y) * x
    return suma / n

def detrB(w, b):
    suma = 0
    for i in range(n):
        x = points_x[i]
        y = points_y[i]
        suma += (w * x + b - y)
    return suma / n

def linear_regresion_gradient_descent():
    plt.ion()  
    fig = plt.figure()  
    w = 0
    b = 0
    alpha = 0.01
    for _ in range(1000):  
        dW = detrW(w, b)
        dB = detrB(w, b)
        if abs(dW) < 0.001 and abs(dB) < 0.001:
            break
        w = w - alpha * dW
        b = b - alpha * dB
        show_f(w, b)
    plt.ioff()  
    show_f(w, b)  

linear_regresion_gradient_descent()
