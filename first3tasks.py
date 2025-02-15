import numpy as np
import matplotlib.pyplot as plt
import tkinter as tk
from tkinter import messagebox
from scipy.optimize import bisect


def graph_method():
    try:
        func_str = entry_func_graph.get()
        a = float(entry_a_graph.get())
        b = float(entry_b_graph.get())
        approx_root = float(entry_approx_root.get())

        f = lambda x: eval(func_str, {"x": x, "np": np})

        if f(a) * f(b) >= 0:
            messagebox.showerror("Error", "Function must have opposite signs at a and b (f(a) * f(b) < 0).")
            return

        x = np.linspace(a, b, 400)
        y = np.array([f(val) for val in x])

        plt.figure(figsize=(10, 6))
        plt.plot(x, y, label=f'f(x) = {func_str}')
        plt.axhline(0, color='black', linewidth=0.5, linestyle='--')
        plt.axvline(0, color='black', linewidth=0.5, linestyle='--')
        plt.grid(True)

        numerical_root = bisect(f, a, b)
        absolute_error = abs(numerical_root - approx_root)

        plt.scatter(approx_root, f(approx_root), color='red', label=f'Approx Root: {approx_root:.4f}')
        plt.scatter(numerical_root, 0, color='green', label=f'Bisect Root: {numerical_root:.4f}')

        plt.legend()
        plt.show()

        messagebox.showinfo("Graph Method",
                            f"Numerical Root: {numerical_root:.4f}\nAbsolute Error: {absolute_error:.4f}")
    except Exception as e:
        messagebox.showerror("Error", str(e))


def run_bisection_secant():
    try:
        func_str = entry_func_bisec_secant.get()
        a = float(entry_a_bisec_secant.get())
        b = float(entry_b_bisec_secant.get())
        tol = float(entry_tol_bisec_secant.get())

        f = lambda x: eval(func_str, {"x": x, "np": np})

        if f(a) * f(b) >= 0:
            messagebox.showerror("Error", "Function must have opposite signs at a and b (f(a) * f(b) < 0).")
            return

        def bisection(a, b, tol):
            iterations = 0
            while (b - a) / 2 > tol:
                c = (a + b) / 2
                if f(c) == 0 or (b - a) / 2 < tol:
                    return c, iterations
                elif f(a) * f(c) < 0:
                    b = c
                else:
                    a = c
                iterations += 1
            return (a + b) / 2, iterations

        def secant(x0, x1, tol, max_iter=100):
            iterations = 0
            for _ in range(max_iter):
                if abs(x1 - x0) < tol:
                    return x1, iterations
                if f(x1) - f(x0) == 0:
                    return None, iterations
                x2 = x1 - f(x1) * (x1 - x0) / (f(x1) - f(x0))
                x0, x1 = x1, x2
                iterations += 1
            return None, iterations

        root_bisection, iter_bisection = bisection(a, b, tol)
        root_secant, iter_secant = secant(a, b, tol)

        if root_secant is None:
            secant_msg = "Failed to converge"
            relative_error_secant = "N/A"
        else:
            secant_msg = f"{root_secant:.6f}"
            relative_error_secant = abs(root_secant - root_bisection) / abs(root_bisection)

        relative_error_bisection = abs(root_bisection - root_secant) / abs(root_secant) if root_secant else "N/A"

        messagebox.showinfo("Bisection & Secant",
                            f"Bisection Root: {root_bisection:.6f}\nIterations: {iter_bisection}\nRelative Error: {relative_error_bisection}\n\n"
                            f"Secant Root: {secant_msg}\nIterations: {iter_secant}\nRelative Error: {relative_error_secant}")
    except Exception as e:
        messagebox.showerror("Error", str(e))


def run_jacobi():
    try:
        A = np.array(eval(entry_matrix_A.get()))
        b = np.array(eval(entry_matrix_b.get()))
        x0 = np.array(eval(entry_matrix_x0.get()))
        tol = float(entry_tol_jacobi.get())

        def jacobi(A, b, x0, tol, max_iter=100):
            n = len(A)
            x = np.array(x0, dtype=float)
            x_new = np.zeros_like(x)

            for iteration in range(max_iter):
                for i in range(n):
                    sum_Ax = sum(A[i][j] * x[j] for j in range(n) if j != i)
                    x_new[i] = (b[i] - sum_Ax) / A[i][i]
                if np.linalg.norm(x_new - x, ord=np.inf) < tol:
                    return x_new, iteration + 1
                x = x_new.copy()
            return x, max_iter

        solution, iterations = jacobi(A, b, x0, tol)

        messagebox.showinfo("Jacobi Method", f"Solution: {solution}\nIterations: {iterations}")
    except Exception as e:
        messagebox.showerror("Error", str(e))


root = tk.Tk()
root.title("Root Finding & Jacobi Method")
root.geometry("600x700")

tk.Label(root, text="Graphical Method").pack()
entry_func_graph = tk.Entry(root)
entry_func_graph.insert(0, "x**3 - 4*x + 1")
entry_func_graph.pack()
tk.Label(root, text="a").pack()
entry_a_graph = tk.Entry(root)
entry_a_graph.insert(0, "0")
entry_a_graph.pack()
tk.Label(root, text="b").pack()
entry_b_graph = tk.Entry(root)
entry_b_graph.insert(0, "3")
entry_b_graph.pack()
tk.Label(root, text="Approximate Root").pack()
entry_approx_root = tk.Entry(root)
entry_approx_root.insert(0, "0.25")
entry_approx_root.pack()
tk.Button(root, text="Run Graph Method", command=graph_method).pack()

tk.Label(root, text="Bisection & Secant Methods").pack()
entry_func_bisec_secant = tk.Entry(root)
entry_func_bisec_secant.insert(0, "x**2 - 5")
entry_func_bisec_secant.pack()
tk.Label(root, text="a").pack()
entry_a_bisec_secant = tk.Entry(root)
entry_a_bisec_secant.insert(0, "2")
entry_a_bisec_secant.pack()
tk.Label(root, text="b").pack()
entry_b_bisec_secant = tk.Entry(root)
entry_b_bisec_secant.insert(0, "3")
entry_b_bisec_secant.pack()
tk.Label(root, text="Tolerance").pack()
entry_tol_bisec_secant = tk.Entry(root)
entry_tol_bisec_secant.insert(0, "1e-6")
entry_tol_bisec_secant.pack()
tk.Button(root, text="Run Bisection & Secant", command=run_bisection_secant).pack()

tk.Label(root, text="Jacobi Method").pack()
entry_matrix_A = tk.Entry(root)
entry_matrix_A.insert(0, "[[1,1,1],[0,2,5],[2,3,1]]")
entry_matrix_A.pack()
tk.Label(root, text="b").pack()
entry_matrix_b = tk.Entry(root)
entry_matrix_b.insert(0, "[6,-4,27]")
entry_matrix_b.pack()
tk.Label(root, text="Initial x0").pack()
entry_matrix_x0 = tk.Entry(root)
entry_matrix_x0.insert(0, "[0,0,0]")
entry_matrix_x0.pack()
tk.Label(root, text="Tolerance").pack()
entry_tol_jacobi = tk.Entry(root)
entry_tol_jacobi.insert(0, "1e-6")
entry_tol_jacobi.pack()
tk.Button(root, text="Run Jacobi", command=run_jacobi).pack()

root.mainloop()