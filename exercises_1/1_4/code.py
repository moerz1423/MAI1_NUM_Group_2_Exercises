import numpy as np

def f(x, y):
    """Objective function f(x, y) = x^2 + y^2."""
    return x**2 + y**2

def grad_f(x, y):
    """Gradient of f: f(x, y) = (2x, 2y)."""
    return np.array([2*x, 2*y], dtype=float)

def gradient_descent(x0, y0, gamma, n_steps):
    """Run n_steps of gradient descent for f starting at (x0, y0)."""
    x, y = float(x0), float(y0)
    trajectory = [(x, y)]
    for k in range(n_steps):
        g = grad_f(x, y)
        x -= gamma * g[0]
        y -= gamma * g[1]
        trajectory.append((x, y))
    return trajectory

# Parameters
x0, y0 = 1.0, -1.0   # initial point
gamma_good = 0.25    # step size (convergent without oscillation)
gamma_overshoot = 1.0  # step size that leads to oscillation

# Run gradient descent with a good step size
traj_good = gradient_descent(x0, y0, gamma_good, n_steps=10)
print("Gradient descent with gamma =", gamma_good)
for k, (xk, yk) in enumerate(traj_good):
    print(f"k={k:2d}: (x, y) = ({xk: .6f}, {yk: .6f}), f = {f(xk, yk): .6e}")

# Run gradient descent with gamma = 1 (overshooting)
traj_over = gradient_descent(x0, y0, gamma_overshoot, n_steps=6)
print("\nGradient descent with gamma =", gamma_overshoot, "(overshoot)")
for k, (xk, yk) in enumerate(traj_over):
    print(f"k={k:2d}: (x, y) = ({xk: .6f}, {yk: .6f}), f = {f(xk, yk): .6e}")

