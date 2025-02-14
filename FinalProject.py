import tkinter as tk
from tkinter import ttk, messagebox
import numpy as np
from scipy import stats


class MathApp:
    def __init__(self, root):
        self.root = root
        self.root.title("Mathematical Computations")

        # Create tabs
        self.notebook = ttk.Notebook(root)
        self.notebook.pack(pady=10, expand=True)

        # Matrix inversion tab
        self.matrix_frame = ttk.Frame(self.notebook)
        self.notebook.add(self.matrix_frame, text="Matrix Inversion")
        self.setup_matrix_tab()

        # Linear regression tab
        self.regression_frame = ttk.Frame(self.notebook)
        self.notebook.add(self.regression_frame, text="Linear Regression")
        self.setup_regression_tab()

        # Interpolation tab
        self.interpolation_frame = ttk.Frame(self.notebook)
        self.notebook.add(self.interpolation_frame, text="Interpolation")
        self.setup_interpolation_tab()

    def setup_matrix_tab(self):
        # Create input fields for 3x3 matrix
        ttk.Label(self.matrix_frame, text="Enter 3x3 matrix elements:").pack(pady=5)

        self.matrix_entries = []
        matrix_frame = ttk.Frame(self.matrix_frame)
        matrix_frame.pack(pady=10)

        for i in range(3):
            row = []
            for j in range(3):
                entry = ttk.Entry(matrix_frame, width=8)
                entry.grid(row=i, column=j, padx=5, pady=5)
                entry.insert(0, "0")
                row.append(entry)
            self.matrix_entries.append(row)

        ttk.Button(self.matrix_frame, text="Calculate",
                   command=self.calculate_inverse).pack(pady=10)

        self.matrix_result = tk.Text(self.matrix_frame, height=6, width=40)
        self.matrix_result.pack(pady=10)

    def setup_regression_tab(self):
        ttk.Label(self.regression_frame,
                  text="Enter points (x,y) separated by commas\nExample: 1,2,3,4,5 for x\n2,3,5,7,11 for y").pack(
            pady=5)

        self.x_points = ttk.Entry(self.regression_frame, width=40)
        self.x_points.pack(pady=5)
        self.x_points.insert(0, "1,2,3,4,5")

        self.y_points = ttk.Entry(self.regression_frame, width=40)
        self.y_points.pack(pady=5)
        self.y_points.insert(0, "2,3,5,7,11")

        ttk.Button(self.regression_frame, text="Calculate",
                   command=self.calculate_regression).pack(pady=10)

        self.regression_result = tk.Text(self.regression_frame, height=4, width=40)
        self.regression_result.pack(pady=10)

    def setup_interpolation_tab(self):
        ttk.Label(self.interpolation_frame,
                  text="Enter x points separated by commas:").pack(pady=5)
        self.x_interp = ttk.Entry(self.interpolation_frame, width=40)
        self.x_interp.pack(pady=5)
        self.x_interp.insert(0, "0,1,2,3")

        ttk.Label(self.interpolation_frame,
                  text="Enter y values separated by commas:").pack(pady=5)
        self.y_interp = ttk.Entry(self.interpolation_frame, width=40)
        self.y_interp.pack(pady=5)
        self.y_interp.insert(0, "1,4,9,16")

        ttk.Label(self.interpolation_frame,
                  text="Enter interpolation point:").pack(pady=5)
        self.x_target = ttk.Entry(self.interpolation_frame, width=10)
        self.x_target.pack(pady=5)
        self.x_target.insert(0, "1.5")

        ttk.Button(self.interpolation_frame, text="Calculate",
                   command=self.calculate_interpolation).pack(pady=10)

        self.interpolation_result = tk.Text(self.interpolation_frame, height=4, width=40)
        self.interpolation_result.pack(pady=10)

    def iterative_matrix_inverse(self, A, max_iter=100, tolerance=1e-10):
        A = np.array(A, dtype=float)
        n = len(A)
        trace = np.trace(A)
        X = (1 / trace) * np.eye(n)

        for i in range(max_iter):
            prev_X = X.copy()
            X = 2 * X - X @ A @ X
            if np.all(np.abs(X - prev_X) < tolerance):
                break
        return X

    def calculate_inverse(self):
        try:
            # Get matrix from input fields
            matrix = []
            for row in self.matrix_entries:
                matrix.append([float(entry.get()) for entry in row])

            # Calculate inverse matrix
            inverse = self.iterative_matrix_inverse(matrix)

            # Verify result
            original = np.array(matrix)
            verification = np.round(original @ inverse, decimals=6)

            # Display result
            result_str = "Inverse matrix:\n"
            for row in inverse:
                result_str += " ".join(f"{x:8.4f}" for x in row) + "\n"
            result_str += "\nVerification (A * A^-1):\n"
            for row in verification:
                result_str += " ".join(f"{x:8.4f}" for x in row) + "\n"

            self.matrix_result.delete(1.0, tk.END)
            self.matrix_result.insert(1.0, result_str)

        except Exception as e:
            messagebox.showerror("Error", str(e))

    def calculate_regression(self):
        try:
            # Get points from input fields
            x = np.array([float(i) for i in self.x_points.get().split(",")])
            y = np.array([float(i) for i in self.y_points.get().split(",")])

            # Calculate regression
            slope, intercept, r_value, p_value, std_err = stats.linregress(x, y)

            # Display result
            result_str = f"Line equation: y = {slope:.4f}x + {intercept:.4f}\n"
            result_str += f"R-squared: {r_value ** 2:.4f}\n"
            result_str += f"Standard error: {std_err:.4f}"

            self.regression_result.delete(1.0, tk.END)
            self.regression_result.insert(1.0, result_str)

        except Exception as e:
            messagebox.showerror("Error", str(e))

    def newtons_interpolation(self, x_data, y_data, x_target):
        n = len(x_data)
        F = np.zeros((n, n))
        F[:, 0] = y_data

        for j in range(1, n):
            for i in range(n - j):
                F[i, j] = (F[i + 1, j - 1] - F[i, j - 1]) / (x_data[i + j] - x_data[i])

        result = F[0, 0]
        xterm = 1
        for j in range(1, n):
            xterm *= (x_target - x_data[j - 1])
            result += F[0, j] * xterm

        return result

    def calculate_interpolation(self):
        try:
            # Get data from input fields
            x_data = np.array([float(i) for i in self.x_interp.get().split(",")])
            y_data = np.array([float(i) for i in self.y_interp.get().split(",")])
            x_target = float(self.x_target.get())

            # Calculate interpolation
            result = self.newtons_interpolation(x_data, y_data, x_target)

            # Display result
            result_str = f"Interpolated value:\n"
            result_str += f"f({x_target}) ≈ {result:.4f}"

            self.interpolation_result.delete(1.0, tk.END)
            self.interpolation_result.insert(1.0, result_str)

        except Exception as e:
            messagebox.showerror("Error", str(e))


if __name__ == "__main__":
    root = tk.Tk()
    app = MathApp(root)
    root.mainloop()