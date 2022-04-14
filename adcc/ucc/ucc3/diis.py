"""
Ugly DIIS Helper class
"""
import numpy as np


class Diis:
    def __init__(self, max_vec=10):
        self.residuals = []
        self.solutions = []
        self.max_vec = max_vec

    def pop(self):
        if len(self.solutions) > self.max_vec:
            self.solutions.pop(0)
            self.residuals.pop(0)

    def add_vectors(self, solution, residual):
        self.solutions.append(solution)
        self.residuals.append(residual)
        self.pop()

    def get_optimal_linear_combination(self):
        diis_size = len(self.solutions) + 1
        diis_mat = np.zeros((diis_size, diis_size))
        diis_mat[:, 0] = -1.0
        diis_mat[0, :] = -1.0
        for k, r1 in enumerate(self.residuals, 1):
            for ll, r2 in enumerate(self.residuals, 1):
                diis_mat[k, ll] = r1.dot(r2)
                diis_mat[ll, k] = diis_mat[k, ll]
        diis_rhs = np.zeros(diis_size)
        diis_rhs[0] = -1.0
        weights = np.linalg.solve(diis_mat, diis_rhs)[1:]
        solution = 0
        for ii, s in enumerate(self.solutions):
            solution += s * weights[ii]
        return solution.evaluate()
