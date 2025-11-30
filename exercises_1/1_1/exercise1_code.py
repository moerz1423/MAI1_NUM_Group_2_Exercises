import numpy as np
import matplotlib.pyplot as plt

# Taylor approximation of exp(x) - 1 around x = 0:
# exp(x) - 1 = x + x^2/2! + ... + x^n/n!
def taylor_exp_minus_one(x, n_terms=10):
    x = np.array(x, dtype=np.float64)
    s = np.zeros_like(x)
    term = x.copy()  # first term: x^1/1!

    for k in range(1, n_terms + 1):
        if k == 1:
            term = x                   # x^1 / 1!
        else:
            term = term * x / k        # x^k / k! from previous term
        s += term

    return s

def main():
    # x = 10^-k, k = 1, 2, ..., 15
    k_vals = np.arange(1, 16)
    x = 10.0 ** (-k_vals)

    # Direct evaluation using double precision (np.float64)
    f_direct = np.exp(x) - 1.0

    # "Almost exact" value via Taylor polynomial up to degree 10
    f_taylor = taylor_exp_minus_one(x, n_terms=10)

    # Relative error between the two
    rel_err = np.abs(f_direct - f_taylor) / np.abs(f_taylor)

    # Optional: print a small table
    print("  k       x         direct         taylor         rel. error")
    for k, xv, fd, ft, err in zip(k_vals, x, f_direct, f_taylor, rel_err):
        print(f"{k:3d}  {xv:8.1e}  {fd: .3e}  {ft: .3e}  {err: .3e}")

    # Plot relative error on a logarithmic scale
    plt.figure()
    plt.loglog(x, rel_err, "o-")
    plt.gca().invert_xaxis()  # so smaller x (10^-k) appear to the right if you like
    plt.xlabel(r"$x = 10^{-k}$")
    plt.ylabel(r"relative error $|f_{\mathrm{direct}} - f_{\mathrm{Taylor}}| / |f_{\mathrm{Taylor}}|$")
    plt.title(r"Relative error of $\exp(x)-1$ vs Taylor approximation")
    plt.grid(True, which="both", ls="--")
    plt.show()

if __name__ == "__main__":
    main()

