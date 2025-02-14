import numpy as np
from scipy import stats


def iterative_matrix_inverse(A, max_iter=100, tolerance=1e-10):
    """
    Compute matrix inverse using iterative method based on trace
    """
    # Convert to numpy array
    A = np.array(A, dtype=float)
    n = len(A)

    # Initial guess based on trace
    trace = np.trace(A)
    X = (1 / trace) * np.eye(n)

    for i in range(max_iter):
        prev_X = X.copy()
        X = 2 * X - X @ A @ X

        # Check convergence
        if np.all(np.abs(X - prev_X) < tolerance):
            break

    return X


def linear_curve_fitting(x, y):
    """
    Fit a line to data points using least squares
    """
    slope, intercept, r_value, p_value, std_err = stats.linregress(x, y)
    return slope, intercept


def newtons_interpolation(x_data, y_data, x_target):
    """
    Newton's forward interpolation formula
    """
    n = len(x_data)
    F = np.zeros((n, n))
    F[:, 0] = y_data

    # Calculate divided differences
    for j in range(1, n):
        for i in range(n - j):
            F[i, j] = (F[i + 1, j - 1] - F[i, j - 1]) / (x_data[i + j] - x_data[i])

    # Compute interpolation
    result = F[0, 0]
    xterm = 1
    for j in range(1, n):
        xterm *= (x_target - x_data[j - 1])
        result += F[0, j] * xterm

    return result


# Task 4: Matrix Inversion
A = np.array([[4, -2, 1],
              [-2, 4, -2],
              [1, -2, 4]])
inverse = iterative_matrix_inverse(A)
print("Task 4 - Matrix Inverse:")
print(inverse)
print("\nVerification (A * A^-1 should be close to identity):")
print(np.round(A @ inverse, decimals=6))

# Task 5: Linear Curve Fitting
x = np.array([1, 2, 3, 4, 5])
y = np.array([2, 3, 5, 7, 11])
slope, intercept = linear_curve_fitting(x, y)
print("\nTask 5 - Linear Curve Fitting:")
print(f"Fitted line: y = {slope:.4f}x + {intercept:.4f}")

# Task 6: Newton's Interpolation
x_data = np.array([0, 1, 2, 3])
y_data = np.array([1, 4, 9, 16])
x_target = 1.5
result = newtons_interpolation(x_data, y_data, x_target)
print("\nTask 6 - Newton's Interpolation:")
print(f"f(1.5) ≈ {result:.4f}")