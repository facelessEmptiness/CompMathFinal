import numpy as np
import matplotlib.pyplot as plt

# Task 7: First Derivative Using Newton’s Forward Difference Formula
x = np.array([0, 1, 2])  # Given x values
y = np.array([1, 8, 27])  # Given y values

# Step size
h = x[1] - x[0]  # Assuming uniform spacing

# First-order forward difference formula: f'(x) ≈ (f(x+h) - f(x)) / h
dy_dx_at_x1 = (y[1] - y[0]) / h

dy_dx_at_x2 = (y[2] - y[1]) / h

# Newton’s forward difference approximation at x=1 (midpoint)
dy_dx_at_x1_newton = dy_dx_at_x1 + ((dy_dx_at_x2 - dy_dx_at_x1) / 2)

# Print the result
print(f"Estimated dy/dx at x=1 using Newton's Forward Difference: {dy_dx_at_x1_newton}")

# Visualization
plt.plot(x, y, 'bo-', label='Data points')
plt.xlabel("x")
plt.ylabel("y")
plt.title("First Derivative Estimation")
plt.legend()
plt.grid()
plt.show()

# Task 8: Trapezoidal Rule Integration
def f(x):
    return x**2 + x  # Given function

# Integration limits
a, b = 0, 1
n = 4  # Number of subintervals

# Step size
h = (b - a) / n

# Trapezoidal Rule Formula
x_values = np.linspace(a, b, n+1)
y_values = f(x_values)

integral_approx = (h / 2) * (y_values[0] + 2 * sum(y_values[1:-1]) + y_values[-1])

# Exact integral calculation
from sympy import symbols, integrate
x_sym = symbols('x')
exact_integral = integrate(x_sym**2 + x_sym, (x_sym, a, b)).evalf()

# Print results
print(f"Approximated Integral using Trapezoidal Rule: {integral_approx}")
print(f"Exact Integral Value: {exact_integral}")
print(f"Error: {abs(exact_integral - integral_approx)}")

# Visualization of integration
x_curve = np.linspace(a, b, 100)
y_curve = f(x_curve)
plt.plot(x_curve, y_curve, 'r-', label='Function x^2 + x')
plt.fill_between(x_values, y_values, alpha=0.3, label='Trapezoidal Approximation')
plt.xlabel("x")
plt.ylabel("f(x)")
plt.title("Trapezoidal Rule Approximation")
plt.legend()
plt.grid()
plt.show()