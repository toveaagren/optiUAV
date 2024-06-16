import numpy as np
import scipy.optimize as optimize


def find_zeros(
    root_function: callable, initial_guess_points: np.ndarray, tolerance=1e-12
) -> np.ndarray:
    zeros = []
    for t0 in initial_guess_points:
        results = optimize.fsolve(root_function, t0, full_output=True)
        status = results[-2]
        root = results[0]
        if status == 1 and np.abs(root_function(root)) < tolerance:
            zeros.append(root[0])
    return np.array(zeros)


def unique_with_tolerance(arr, tol=1e-8):
    # Sort array to make differences calculation valid
    sorted_arr = np.sort(arr)
    # Calculate differences between consecutive elements
    diffs = np.diff(sorted_arr)
    # Find indices where differences are greater than tolerance
    indices = np.where(diffs > tol)[0]
    # Include the last element as well
    unique_indices = np.concatenate(([0], indices + 1))
    return sorted_arr[unique_indices]


def unpack_coefficients(
    q: np.ndarray, n_state: int, n_input: int, n_colloc: int
) -> tuple:
    a_ = np.zeros(
        (n_state, n_colloc)
    )  # These are the state coefficients a_k, i.e., the k_th column is a the k_th column is a_k
    b_ = np.zeros((n_input, n_colloc))
    end_indx = n_state * n_colloc
    for k in range(n_colloc):
        a_[:, k] = q[n_state * k : n_state * (k + 1)]
    for k in range(n_colloc):
        b_[:, k] = q[
            (end_indx + n_input * k) : (end_indx + n_input * (k + 1))
        ]  # %Similarly, these are the input coefficients b_k
    return a_, b_
