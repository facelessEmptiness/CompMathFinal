import numpy as np
import tkinter as tk
from tkinter import ttk
from sympy import symbols, integrate

class Application(tk.Tk):
    def __init__(self):
        super().__init__()
        self.title("Last 2 tasks")
        self.geometry("400x400")

        notebook = ttk.Notebook(self)
        notebook.pack(expand=True, fill="both")

        self.newtons_tab(notebook)
        self.trapezoidal_tab(notebook)

    def newtons_tab(self, notebook):
        frame_newton = ttk.Frame(notebook)
        notebook.add(frame_newton, text="Newton's Forward Difference")

        ttk.Label(frame_newton, text="x values (comma separated):").pack()
        self.entry_x = ttk.Entry(frame_newton)
        self.entry_x.insert(0, "0,1,2")
        self.entry_x.pack()

        ttk.Label(frame_newton, text="y values (comma separated):").pack()
        self.entry_y = ttk.Entry(frame_newton)
        self.entry_y.insert(0, "1,8,27")
        self.entry_y.pack()

        self.label_newton_result = ttk.Label(frame_newton, text="Result will be displayed here", font=("Arial", 12, "bold"), foreground="red")
        self.label_newton_result.pack(pady=5)

        ttk.Button(frame_newton, text="Calculate", command=self.calculate_newton).pack(pady=5)

    def trapezoidal_tab(self, notebook):
        frame_trapezoidal = ttk.Frame(notebook)
        notebook.add(frame_trapezoidal, text="Trapezoidal Rule")

        ttk.Label(frame_trapezoidal, text="Function: x^2 + x").pack()
        ttk.Label(frame_trapezoidal, text="Lower limit (a):").pack()

        self.entry_a = ttk.Entry(frame_trapezoidal)
        self.entry_a.insert(0, "0")
        self.entry_a.pack()

        ttk.Label(frame_trapezoidal, text="Upper limit (b):").pack()
        self.entry_b = ttk.Entry(frame_trapezoidal)
        self.entry_b.insert(0, "1")
        self.entry_b.pack()

        ttk.Label(frame_trapezoidal, text="Number of subintervals (n):").pack()
        self.entry_n = ttk.Entry(frame_trapezoidal)
        self.entry_n.insert(0, "4")
        self.entry_n.pack()

        self.label_trap_result = ttk.Label(frame_trapezoidal, text="Result will be displayed here", font=("Arial", 12, "bold"), foreground="red")
        self.label_trap_result.pack(pady=5)

        ttk.Button(frame_trapezoidal, text="Calculate", command=self.calculate_trapezoidal).pack(pady=5)

    def calculate_newton(self):
        x = np.array([float(i) for i in self.entry_x.get().split(",")])
        y = np.array([float(i) for i in self.entry_y.get().split(",")])

        h = x[1] - x[0]
        dy_dx_at_x1 = (y[1] - y[0]) / h
        dy_dx_at_x2 = (y[2] - y[1]) / h
        dy_dx_at_x1_newton = dy_dx_at_x1 + ((dy_dx_at_x2 - dy_dx_at_x1) / 2)

        self.label_newton_result.config(text=f"Estimated dy/dx at x=1: {dy_dx_at_x1_newton:.4f}")

    def calculate_trapezoidal(self):
        def f(x):
            return x**2 + x

        a = float(self.entry_a.get())
        b = float(self.entry_b.get())
        n = int(self.entry_n.get())

        h = (b - a) / n
        x_values = np.linspace(a, b, n+1)
        y_values = f(x_values)
        integral_approx = (h / 2) * (y_values[0] + 2 * sum(y_values[1:-1]) + y_values[-1])

        x_sym = symbols('x')
        exact_integral = integrate(x_sym**2 + x_sym, (x_sym, a, b)).evalf()

        self.label_trap_result.config(text=f"Approx: {integral_approx:.4f}, Exact: {exact_integral:.4f}, Error: {abs(exact_integral - integral_approx):.4f}")

if __name__ == "__main__":
    app = Application()
    app.mainloop()