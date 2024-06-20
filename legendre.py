import math
import numpy as np
from dotmap import DotMap
from utils import unique_with_tolerance, find_zeros


def get_collocation(ps_N: int, n_t_ps_grid=200) -> DotMap:
    L_N = legendre_polynomial(n=ps_N)
    L_N_dot = legendre_polynomial_dot(n=ps_N)

    t_guess_array = np.linspace(
        start=-1, stop=1, num=100 * ps_N, endpoint=False
    )  # Increased mumber of points to make sure all zeros are found
    tm_calc = np.zeros(len(t_guess_array))

    # Collocation points are time instants in [-1, 1] where the
    # derivatives of Legendre polynomials become zero. These points
    # are found numerically.
    t_ps_grid = [-1]
    tm_idx_grid = [0]
    LGL_colloc = DotMap()
    # Calculate collocation points: Where Ldot is zeros
    tk = unique_with_tolerance(find_zeros(L_N_dot, t_guess_array))
    tm = np.concatenate([[-1], tk, [1]])  # Add endpoints of interval
    assert len(tm) == ps_N + 1
    L_N_tm = L_N(tm)
    assert len(L_N_tm) == ps_N + 1
    LGL_colloc.tk, LGL_colloc.tm = tk, tm
    LGL_colloc.L_N_tm = (
        L_N_tm  # Legendre polynomials evaluated at the collocation points
    )

    dt_ps = (tm[1] - tm[0]) / n_t_ps_grid

    # Create time grid that contains all collocation points
    for ell in range(ps_N):
        t_range = np.arange(tm[ell] + dt_ps, tm[ell + 1], dt_ps)
        t_ps_grid.extend(t_range)
        t_ps_grid.append(tm[ell + 1])
        # Append the current length of t_ps_grid to tm_idx_grid
        tm_idx_grid.append(len(t_ps_grid) - 1)

    # Convert t_ps_grid and tm_idx_grid to numpy arrays if needed
    t_ps_grid = np.array(t_ps_grid)
    assert t_ps_grid[-1] == 1
    assert t_ps_grid[0] == -1
    tm_idx_grid = np.array(tm_idx_grid)  # Indices of collocation in this grid
    # print(tm_idx_grid)
    assert len(tm_idx_grid) == ps_N + 1
    assert np.all(np.isin(tm, t_ps_grid)), "Not all elements of tm are in t_ps_grid"
    assert [t_ps_grid[tm_idx_grid][i] == tm[i] for i in range(len(tm))]

    # Create differentiation matrix
    ps_D = np.zeros((ps_N + 1, ps_N + 1))
    for m in range(ps_N + 1):
        for l in range(ps_N + 1):
            if m == l:
                ps_D[m, l] = 0
            else:
                ps_D[m, l] = L_N(tm[m]) / L_N(tm[l]) * (1 / (tm[m] - tm[l]))

    ps_D[0, 0] = -ps_N * (ps_N + 1) / 4
    ps_D[ps_N, ps_N] = ps_N * (ps_N + 1) / 4
    LGL_colloc.D = ps_D
    LGL_colloc.t_grid = t_ps_grid
    LGL_colloc.tm_idx = tm_idx_grid

    # print([tm_idx_grid)

    # 	%----- Lagrange polynomials
    # 	% These are needed to reconstruct the state and input after the
    # 	% transcribed optimization problem is solved
    phi_l = np.zeros(
        (len(t_ps_grid), ps_N + 1)
    )  # Evaluate lagrange polynomials at the discretized time points
    for ell in range(ps_N + 1):
        phi = lagrange_polynomial(ps_N, ell, tm)
        phi_l[:, ell] = phi(t_ps_grid)
        phi_l[tm_idx_grid, ell] = 0  # Zero at all collocation points
        phi_l[tm_idx_grid[ell], ell] = 1  # Except for k = l
    LGL_colloc.phi_l = phi_l
    return LGL_colloc


from scipy.special import legendre


def legendre_polynomial(n: int) -> callable:
    """Returns the Legendre polynomial of degree n."""
    return legendre(n)


def legendre_polynomial_dot(n: int) -> callable:
    """Returns the derivative of the Legendre polynomial of degree n."""
    Pn = legendre(n)
    Pn_dot = np.polyder(Pn)
    return np.poly1d(Pn_dot)


# def legendre_polynomial(ps_N: int) -> callable:
#     ps_K = math.floor(ps_N / 2)
#     L_N = lambda t: sum(
#         ((-1) ** k1 * math.factorial(2 * ps_N - 2 * k1))
#         / (
#             2**ps_N
#             * math.factorial(k1)
#             * math.factorial(ps_N - k1)
#             * math.factorial(ps_N - 2 * k1)
#         )
#         * t ** (ps_N - 2 * k1)
#         for k1 in range(ps_K + 1)
#     )
#     return L_N


# def legendre_polynomial_dot(ps_N):
#     ps_K = math.floor(ps_N / 2)
#     if ps_N % 2 == 0:
#         return lambda t: sum(
#             (
#                 ((-1) ** k * math.factorial(2 * ps_N - 2 * k) * (ps_N - 2 * k))
#                 / (
#                     2**ps_N
#                     * math.factorial(k)
#                     * math.factorial(ps_N - k)
#                     * math.factorial(ps_N - 2 * k)
#                 )
#                 * t ** (ps_N - 2 * k - 1)
#                 if t != 0.0
#                 else 0.0
#             )
#             for k in range(ps_K + 1)
#         )
#     else:
#         return (
#             lambda t: sum(
#                 ((-1) ** k * math.factorial(2 * ps_N - 2 * k) * (ps_N - 2 * k))
#                 / (
#                     2**ps_N
#                     * math.factorial(k)
#                     * math.factorial(ps_N - k)
#                     * math.factorial(ps_N - 2 * k)
#                 )
#                 * t ** (ps_N - 2 * k - 1)
#                 for k in range(ps_K + 1)
#             )
#             + (
#                 (-1) ** ps_K
#                 * math.factorial(ps_N)
#                 / (2**ps_N * math.factorial(ps_K) * math.factorial(ps_N - ps_K))
#             )
#             * t**ps_K
#         )


# TODO: SHould handle zero case better
def lagrange_polynomial(ps_N: int, k: int, tk: np.ndarray) -> callable:
    LN = legendre_polynomial(ps_N)
    LN_dot = legendre_polynomial_dot(ps_N)
    phi_k = lambda t: (1 / (ps_N * (ps_N + 1) * LN(tk[k]))) * (
        ((t**2 - 1) * LN_dot(t)) / (t - tk[k])
    )
    return phi_k


# def lagrange_polynomial(ps_N: int, k: int, tk: np.ndarray) -> callable:
#     """
#     Returns the k-th Lagrange polynomial for a given order ps_N and nodes tk.
#     """
#     LN = legendre_polynomial(ps_N)
#     LN_dot = legendre_polynomial_dot(ps_N)

#     def phi_k(t):
#         if np.isclose(t, tk[k]):
#             return (
#                 (1 / (ps_N * (ps_N + 1) * LN(tk[k]))) * (tk[k] ** 2 - 1) * LN_dot(tk[k])
#             )
#         else:
#             return (1 / (ps_N * (ps_N + 1) * LN(tk[k]))) * (
#                 ((t**2 - 1) * LN_dot(t)) / (t - tk[k])
#             )

#     return phi_k
