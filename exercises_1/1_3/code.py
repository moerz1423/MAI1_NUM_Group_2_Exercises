import sympy as sp
import numpy as np

# ----- Define symbols for the direct (symbolic) approach -----
x_sym, a_sym, b_sym, c_sym = sp.symbols("x a b c", real=True)

def f_sym(expr):
    return a_sym*expr**2 + b_sym + c_sym**2

# F(x) = f(f(f(x+a) + b) + c)
z0_sym = x_sym + a_sym
z1_sym = f_sym(z0_sym)
z2_sym = z1_sym + b_sym
z3_sym = f_sym(z2_sym)
z4_sym = z3_sym + c_sym
F_sym = f_sym(z4_sym)

Fa_sym = sp.diff(F_sym, a_sym)
Fb_sym = sp.diff(F_sym, b_sym)
Fc_sym = sp.diff(F_sym, c_sym)

F_direct_func = sp.lambdify((x_sym, a_sym, b_sym, c_sym),
                            (F_sym, Fa_sym, Fb_sym, Fc_sym),
                            "numpy")

# ----- Scalar function f and its partials -----
def f_scalar(x, a, b, c):
    return a*x**2 + b + c**2

def f_partials(x, a, b, c):
    fx = 2*a*x
    fa = x**2
    fb = 1.0
    fc = 2*c
    return fx, fa, fb, fc

# ----- Backpropagation for F -----
def F_backprop(x, a, b, c):
    # Forward pass
    z0 = x + a
    z1 = f_scalar(z0, a, b, c)
    z2 = z1 + b
    z3 = f_scalar(z2, a, b, c)
    z4 = z3 + c
    F_val = f_scalar(z4, a, b, c)

    # Backward pass
    # Initialize parameter gradients
    Fa = 0.0
    Fb = 0.0
    Fc = 0.0

    # Node F = f(z4)
    fx4, fa4, fb4, fc4 = f_partials(z4, a, b, c)
    g4 = fx4
    Fa += fa4
    Fb += fb4
    Fc += fc4

    # Node z4 = z3 + c
    g3 = g4
    Fc += g4  # derivative wrt c

    # Node z3 = f(z2)
    fx2, fa2, fb2, fc2 = f_partials(z2, a, b, c)
    g2 = g3 * fx2
    Fa += g3 * fa2
    Fb += g3 * fb2
    Fc += g3 * fc2

    # Node z2 = z1 + b
    g1 = g2
    Fb += g2  # derivative wrt b

    # Node z1 = f(z0)
    fx0, fa0, fb0, fc0 = f_partials(z0, a, b, c)
    g0 = g1 * fx0
    Fa += g1 * fa0
    Fb += g1 * fb0
    Fc += g1 * fc0

    # Node z0 = x + a
    Fa += g0  # derivative wrt a

    return F_val, Fa, Fb, Fc

# ----- Numerical test -----
x_val = 0.7
a_val = 1.3
b_val = -0.5
c_val = 0.9

F_dir, Fa_dir, Fb_dir, Fc_dir = F_direct_func(x_val, a_val, b_val, c_val)
F_bp,  Fa_bp,  Fb_bp,  Fc_bp  = F_backprop(x_val, a_val, b_val, c_val)

print("Direct    F, Fa, Fb, Fc:")
print(F_dir, Fa_dir, Fb_dir, Fc_dir)
print("Backprop  F, Fa, Fb, Fc:")
print(F_bp,  Fa_bp,  Fb_bp,  Fc_bp)

print("\nAbsolute differences:")
print(" |Fa_dir - Fa_bp| =", abs(Fa_dir - Fa_bp))
print(" |Fb_dir - Fb_bp| =", abs(Fb_dir - Fb_bp))
print(" |Fc_dir - Fc_bp| =", abs(Fc_dir - Fc_bp))

