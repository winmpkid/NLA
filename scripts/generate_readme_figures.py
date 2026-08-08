"""Generate the stability figures displayed near the top of README.md."""

from pathlib import Path
import time

import matplotlib.pyplot as plt
import numpy as np


ROOT = Path(__file__).resolve().parents[1]
OUTPUT_DIR = ROOT / "docs" / "figures"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

BLUE = "#2563EB"
ORANGE = "#EA580C"
GREEN = "#16A34A"
PURPLE = "#7C3AED"
SLATE = "#475569"
RED = "#DC2626"
EPS = np.finfo(float).eps

plt.rcParams.update(
    {
        "font.size": 11,
        "axes.titlesize": 15,
        "axes.titleweight": "bold",
        "axes.labelsize": 11,
        "axes.spines.top": False,
        "axes.spines.right": False,
        "axes.grid": True,
        "grid.alpha": 0.22,
        "grid.linestyle": "--",
        "legend.frameon": False,
        "figure.facecolor": "white",
        "axes.facecolor": "white",
        "savefig.facecolor": "white",
    }
)


def save_figure(fig, filename):
    fig.tight_layout()
    fig.savefig(OUTPUT_DIR / filename, dpi=180, bbox_inches="tight")
    plt.close(fig)


def forward_substitution(L, b):
    y = np.empty_like(b, dtype=float)
    for i in range(L.shape[0]):
        y[i] = (b[i] - L[i, :i] @ y[:i]) / L[i, i]
    return y


def backward_substitution(U, y):
    x = np.empty_like(y, dtype=float)
    for i in range(U.shape[0] - 1, -1, -1):
        x[i] = (y[i] - U[i, i + 1 :] @ x[i + 1 :]) / U[i, i]
    return x


def lu_no_pivot(A):
    work = np.asarray(A, dtype=float).copy()
    n = work.shape[0]
    L = np.eye(n)
    U = np.zeros_like(work)
    for i in range(n):
        U[i, i:] = work[i, i:]
        if i < n - 1:
            L[i + 1 :, i] = work[i + 1 :, i] / work[i, i]
            work[i + 1 :, i + 1 :] -= np.outer(L[i + 1 :, i], U[i, i + 1 :])
    return L, U


def lu_partial_pivot(A):
    work = np.asarray(A, dtype=float).copy()
    n = work.shape[0]
    L = np.eye(n)
    P = np.eye(n)
    U = np.zeros_like(work)
    for i in range(n):
        pivot_row = i + np.argmax(np.abs(work[i:, i]))
        if pivot_row != i:
            work[[i, pivot_row], :] = work[[pivot_row, i], :]
            P[[i, pivot_row], :] = P[[pivot_row, i], :]
            L[[i, pivot_row], :i] = L[[pivot_row, i], :i]
        U[i, i:] = work[i, i:]
        if i < n - 1:
            L[i + 1 :, i] = work[i + 1 :, i] / work[i, i]
            work[i + 1 :, i + 1 :] -= np.outer(L[i + 1 :, i], U[i, i + 1 :])
    return P, L, U


def solve_lu(P, L, U, b):
    return backward_substitution(U, forward_substitution(L, P @ b))


def plot_lu_pivoting():
    pivot_sizes = np.logspace(-16, -1, 31)
    true_x = np.array([1.0, 1.0])
    no_pivot_errors = []
    pivot_errors = []

    for pivot_size in pivot_sizes:
        A = np.array([[pivot_size, 1.0], [1.0, 1.0]])
        b = A @ true_x

        L, U = lu_no_pivot(A)
        x_no_pivot = solve_lu(np.eye(2), L, U, b)
        P, L, U = lu_partial_pivot(A)
        x_pivot = solve_lu(P, L, U, b)

        no_pivot_errors.append(np.linalg.norm(x_no_pivot - true_x) / np.linalg.norm(true_x))
        pivot_errors.append(np.linalg.norm(x_pivot - true_x) / np.linalg.norm(true_x))

    fig, ax = plt.subplots(figsize=(8.2, 5.0))
    ax.loglog(pivot_sizes, np.maximum(no_pivot_errors, EPS), "o-", color=RED, label="LU without pivoting")
    ax.loglog(pivot_sizes, np.maximum(pivot_errors, EPS), "s-", color=BLUE, label="LU with partial pivoting")
    ax.axhline(EPS, color=SLATE, linestyle=":", linewidth=1.4, label="Machine precision")
    ax.set(
        title="LU Stability: Partial Pivoting Prevents Error Amplification",
        xlabel=r"Leading pivot magnitude $\epsilon$",
        ylabel="Relative forward error",
    )
    ax.legend(loc="best")
    save_figure(fig, "lu-pivoting-stability.png")


def cholesky_decomposition(A):
    A = np.asarray(A, dtype=float)
    n = A.shape[0]
    L = np.zeros_like(A)
    for j in range(n):
        diagonal = A[j, j] - L[j, :j] @ L[j, :j]
        L[j, j] = np.sqrt(diagonal)
        if j < n - 1:
            L[j + 1 :, j] = (A[j + 1 :, j] - L[j + 1 :, :j] @ L[j, :j]) / L[j, j]
    return L


def median_runtime(function, A, repeats=5):
    samples = []
    for _ in range(repeats):
        start = time.perf_counter()
        function(A)
        samples.append(time.perf_counter() - start)
    return np.median(samples)


def plot_factorization_runtime():
    rng = np.random.default_rng(10)
    sizes = np.array([30, 60, 100, 160, 240, 340])
    lu_times = []
    cholesky_times = []

    for n in sizes:
        M = rng.normal(size=(n, n))
        A = M.T @ M + 0.5 * np.eye(n)
        lu_times.append(median_runtime(lambda matrix: lu_partial_pivot(matrix), A))
        cholesky_times.append(median_runtime(cholesky_decomposition, A))

    lu_times = np.asarray(lu_times)
    cholesky_times = np.asarray(cholesky_times)

    fig, ax = plt.subplots(figsize=(8.2, 5.0))
    ax.loglog(sizes, lu_times * 1e3, "o-", color=ORANGE, label="LU with partial pivoting")
    ax.loglog(sizes, cholesky_times * 1e3, "s-", color=GREEN, label="Cholesky")
    ax.loglog(
        sizes,
        cholesky_times[0] * (sizes / sizes[0]) ** 3 * 1e3,
        "--",
        color=SLATE,
        linewidth=1.5,
        label=r"$O(n^3)$ reference",
    )
    ax.set(
        title="Exploiting SPD Structure Reduces Factorization Cost",
        xlabel="Matrix dimension n",
        ylabel="Median runtime (ms)",
    )
    ax.legend(loc="best")
    save_figure(fig, "cholesky-lu-runtime.png")


def qr_classical(A):
    A = np.asarray(A, dtype=float)
    m, n = A.shape
    Q = np.zeros((m, n))
    R = np.zeros((n, n))
    for j in range(n):
        R[:j, j] = Q[:, :j].T @ A[:, j]
        residual = A[:, j] - Q[:, :j] @ R[:j, j]
        R[j, j] = np.linalg.norm(residual)
        Q[:, j] = residual / R[j, j]
    return Q, R


def qr_modified(A):
    A = np.asarray(A, dtype=float)
    m, n = A.shape
    Q = np.zeros((m, n))
    R = np.zeros((n, n))
    V = A.copy()
    for i in range(n):
        R[i, i] = np.linalg.norm(V[:, i])
        Q[:, i] = V[:, i] / R[i, i]
        for j in range(i + 1, n):
            R[i, j] = Q[:, i] @ V[:, j]
            V[:, j] -= R[i, j] * Q[:, i]
    return Q, R


def plot_qr_orthogonality():
    rng = np.random.default_rng(35)
    m, n = 40, 8
    left, _ = np.linalg.qr(rng.normal(size=(m, n)), mode="reduced")
    right, _ = np.linalg.qr(rng.normal(size=(n, n)))
    condition_numbers = np.logspace(0, 12, 25)
    errors = {"Classical Gram–Schmidt": [], "Modified Gram–Schmidt": [], "Householder": []}

    for condition_number in condition_numbers:
        singular_values = np.geomspace(1.0, 1.0 / condition_number, n)
        A = left @ np.diag(singular_values) @ right.T
        decompositions = {
            "Classical Gram–Schmidt": qr_classical(A),
            "Modified Gram–Schmidt": qr_modified(A),
            "Householder": np.linalg.qr(A, mode="reduced"),
        }
        for name, (Q, _) in decompositions.items():
            errors[name].append(np.linalg.norm(np.eye(n) - Q.T @ Q, ord="fro"))

    fig, ax = plt.subplots(figsize=(8.2, 5.0))
    for name, color, marker in [
        ("Classical Gram–Schmidt", RED, "o"),
        ("Modified Gram–Schmidt", ORANGE, "s"),
        ("Householder", BLUE, "^"),
    ]:
        ax.loglog(condition_numbers, np.maximum(errors[name], EPS), marker + "-", color=color, label=name)
    ax.set(
        title="QR Stability: Orthogonalization Method Matters",
        xlabel="Matrix condition number",
        ylabel=r"Orthogonality error $\|I-Q^TQ\|_F$",
    )
    ax.legend(loc="upper left")
    save_figure(fig, "qr-orthogonality-stability.png")


def plot_svd_conditioning():
    rng = np.random.default_rng(42)
    m, n = 50, 15
    left, _ = np.linalg.qr(rng.normal(size=(m, n)), mode="reduced")
    right, _ = np.linalg.qr(rng.normal(size=(n, n)))
    condition_numbers = np.logspace(0, 14, 29)
    eigen_errors = []
    direct_errors = []

    for condition_number in condition_numbers:
        true_s = np.geomspace(1.0, 1.0 / condition_number, n)
        A = left @ np.diag(true_s) @ right.T

        eigenvalues = np.linalg.eigvalsh(A.T @ A)
        s_from_eigenvalues = np.sqrt(np.clip(eigenvalues, 0.0, None))[::-1]
        s_direct = np.linalg.svd(A, compute_uv=False)

        true_smallest = true_s[-1]
        eigen_errors.append(abs(s_from_eigenvalues[-1] - true_smallest) / true_smallest)
        direct_errors.append(abs(s_direct[-1] - true_smallest) / true_smallest)

    fig, ax = plt.subplots(figsize=(8.2, 5.0))
    ax.loglog(condition_numbers, np.maximum(eigen_errors, EPS), "o-", color=PURPLE, label=r"Eigen route: $A^TA$")
    ax.loglog(condition_numbers, np.maximum(direct_errors, EPS), "s-", color=BLUE, label="Direct SVD")
    threshold = 1.0 / np.sqrt(EPS)
    ax.axvline(threshold, color=SLATE, linestyle="--", linewidth=1.5, label=r"$1/\sqrt{\epsilon_{mach}}$")
    ax.set(
        title=r"SVD Stability: Forming $A^TA$ Squares the Condition Number",
        xlabel="Matrix condition number",
        ylabel="Relative error in smallest singular value",
    )
    ax.legend(loc="upper left")
    save_figure(fig, "svd-conditioning-stability.png")


def main():
    plot_lu_pivoting()
    plot_factorization_runtime()
    plot_qr_orthogonality()
    plot_svd_conditioning()
    print(f"Generated four README figures in {OUTPUT_DIR}")


if __name__ == "__main__":
    main()
