# linear_systems.py

from typing import List, Tuple, Optional, Union
import numpy as np
from dataclasses import dataclass
from numpy.typing import ArrayLike

@dataclass
class SolverConfig:
    """Configuration for iterative solvers."""
    max_iterations: int = 10000
    tolerance: float = 0.001

class LinearSystemSolver:
    """
    A class for solving systems of linear equations using various methods.

    Methods implemented:
    - Gaussian elimination
    - Gauss-Seidel iteration
    - Jacobi iteration
    - LU decomposition
    """

    def __init__(self, config: Optional[SolverConfig] = None):
        self.config = config or SolverConfig()
        self._iteration_count = 0

    @staticmethod
    def _matrix_product(a: List[List[float]], b: List[List[float]]) -> List[List[float]]:
        """Compute matrix product of two matrices."""
        product = []
        for i in range(len(a)):
            product.append([])
            for j in range(len(b[0])):
                product[i].append(0)
                for k in range(len(a[0])):
                    product[i][j] += a[i][k] * b[k][j]
        return product

    @staticmethod
    def _vector_error(a: List[float], b: List[float]) -> float:
        """
        Compute relative error between two vectors.

        Parameters
        ----------
        a : List[float]
            First vector
        b : List[float]
            Second vector

        Returns
        -------
        float
            Relative error
        """
        diff = [x - y for x, y in zip(b, a)]
        return max(map(abs, diff)) / max(map(abs, b))

    @staticmethod
    def _inner_product(a: List[float], b: List[float]) -> float:
        """Compute inner product of two vectors."""
        return sum(x * y for x, y in zip(a, b))

    @staticmethod
    def _euclidean_norm(vector: List[float]) -> float:
        """Compute Euclidean norm of a vector."""
        return np.sqrt(sum(x * x for x in vector))

    @staticmethod
    def _create_zero_vector(size: int) -> List[float]:
        """Create a zero vector of given size."""
        return [0.0] * size

    def _compute_residual(
        self,
        coefficients: List[List[float]],
        constants: List[float],
        solution: List[float]
    ) -> List[float]:
        """
        Compute residual vector for a given solution.

        Parameters
        ----------
        coefficients : List[List[float]]
            Coefficient matrix
        constants : List[float]
            Constants vector
        solution : List[float]
            Current solution vector

        Returns
        -------
        List[float]
            Residual vector
        """
        residual = []
        for i in range(len(coefficients)):
            residual.append(
                constants[i] - self._inner_product(coefficients[i], solution)
            )
        return residual

    def solve_gaussian(
        self,
        coefficients: List[List[float]],
        constants: List[float]
    ) -> List[float]:
        """
        Solve system using Gaussian elimination.

        Parameters
        ----------
        coefficients : List[List[float]]
            Coefficient matrix
        constants : List[float]
            Constants vector

        Returns
        -------
        List[float]
            Solution vector
        """
        n = len(coefficients)
        # Create augmented matrix
        matrix = [row[:] for row in coefficients]
        for i in range(n):
            matrix[i].append(constants[i])

        # Forward elimination
        for i in range(n - 1):
            pivot = matrix[i][i]
            for j in range(i + 1, n):
                factor = matrix[j][i] / pivot
                for k in range(i, n + 1):
                    matrix[j][k] -= factor * matrix[i][k]

        # Back substitution
        solution = self._create_zero_vector(n)
        for i in range(n - 1, -1, -1):
            sum_val = sum(matrix[i][j] * solution[j] for j in range(i + 1, n))
            solution[i] = (matrix[i][n] - sum_val) / matrix[i][i]

        return solution

    def solve_gauss_seidel(
        self,
        coefficients: List[List[float]],
        constants: List[float],
        initial_guess: Optional[List[float]] = None
    ) -> List[float]:
        """
        Solve system using Gauss-Seidel iteration.

        Parameters
        ----------
        coefficients : List[List[float]]
            Coefficient matrix
        constants : List[float]
            Constants vector
        initial_guess : Optional[List[float]]
            Initial solution guess

        Returns
        -------
        List[float]
            Solution vector
        """
        n = len(coefficients)
        solution = initial_guess if initial_guess else self._create_zero_vector(n)
        iteration_matrix = []
        constants_vector = []

        # Compute iteration matrix and constants vector
        for i in range(n):
            iteration_matrix.append([])
            constants_vector.append(constants[i] / coefficients[i][i])
            for j in range(n):
                if i == j:
                    iteration_matrix[i].append(0)
                else:
                    iteration_matrix[i].append(
                        -coefficients[i][j] / coefficients[i][i]
                    )

        # Iterate until convergence
        self._iteration_count = 0
        while self._iteration_count < self.config.max_iterations:
            old_solution = solution[:]
            for i in range(n):
                sum_val = sum(
                    iteration_matrix[i][j] * solution[j]
                    for j in range(n)
                )
                solution[i] = sum_val + constants_vector[i]

            if self._vector_error(old_solution, solution) < self.config.tolerance:
                break

            self._iteration_count += 1

        return solution

    def solve_jacobi(
        self,
        coefficients: List[List[float]],
        constants: List[float],
        initial_guess: Optional[List[float]] = None
    ) -> List[float]:
        """
        Solve system using Jacobi iteration.

        Parameters
        ----------
        coefficients : List[List[float]]
            Coefficient matrix
        constants : List[float]
            Constants vector
        initial_guess : Optional[List[float]]
            Initial solution guess

        Returns
        -------
        List[float]
            Solution vector
        """
        n = len(coefficients)
        solution = initial_guess if initial_guess else self._create_zero_vector(n)
        iteration_matrix = []
        constants_vector = []

        # Compute iteration matrix and constants vector
        for i in range(n):
            iteration_matrix.append([])
            constants_vector.append(constants[i] / coefficients[i][i])
            for j in range(n):
                if i == j:
                    iteration_matrix[i].append(0)
                else:
                    iteration_matrix[i].append(
                        -coefficients[i][j] / coefficients[i][i]
                    )

        # Iterate until convergence
        self._iteration_count = 0
        while self._iteration_count < self.config.max_iterations:
            old_solution = solution[:]
            new_solution = self._create_zero_vector(n)

            for i in range(n):
                sum_val = sum(
                    iteration_matrix[i][j] * old_solution[j]
                    for j in range(n)
                )
                new_solution[i] = sum_val + constants_vector[i]

            if self._vector_error(solution, new_solution) < self.config.tolerance:
                solution = new_solution
                break

            solution = new_solution
            self._iteration_count += 1

        return solution

    def decompose_lu(
        self,
        coefficients: List[List[float]]
    ) -> Tuple[List[List[float]], List[List[float]]]:
        """
        Perform LU decomposition of a matrix.

        Parameters
        ----------
        coefficients : List[List[float]]
            Input matrix

        Returns
        -------
        Tuple[List[List[float]], List[List[float]]]
            Tuple containing (L, U) matrices
        """
        n = len(coefficients)
        L = [[1 if i == j else 0 for j in range(n)] for i in range(n)]
        U = [[0 for _ in range(n)] for _ in range(n)]

        for i in range(n):
            # Upper triangular
            for j in range(i, n):
                sum_val = sum(L[i][k] * U[k][j] for k in range(i))
                U[i][j] = coefficients[i][j] - sum_val

            # Lower triangular
            for j in range(i + 1, n):
                sum_val = sum(L[j][k] * U[k][i] for k in range(i))
                L[j][i] = (coefficients[j][i] - sum_val) / U[i][i]

        return L, U

    def solve_lu(
        self,
        coefficients: List[List[float]],
        constants: List[float]
    ) -> List[float]:
        """
        Solve system using LU decomposition.

        Parameters
        ----------
        coefficients : List[List[float]]
            Coefficient matrix
        constants : List[float]
            Constants vector

        Returns
        -------
        List[float]
            Solution vector
        """
        n = len(coefficients)
        L, U = self.decompose_lu(coefficients)

        # Solve Ly = b
        y = self._create_zero_vector(n)
        for i in range(n):
            sum_val = sum(L[i][j] * y[j] for j in range(i))
            y[i] = constants[i] - sum_val

        # Solve Ux = y
        x = self._create_zero_vector(n)
        for i in range(n - 1, -1, -1):
            sum_val = sum(U[i][j] * x[j] for j in range(i + 1, n))
            x[i] = (y[i] - sum_val) / U[i][i]

        return x

    @property
    def iterations(self) -> int:
        """Get number of iterations from last solve."""
        return self._iteration_count

if __name__ == "__main__":
    # Example usage
    coefficients = [
        [4, 1, -1],
        [2, 7, 1],
        [1, -3, 12]
    ]
    constants = [3, 19, 31]

    solver = LinearSystemSolver()

    # Solve using different methods
    gaussian_solution = solver.solve_gaussian(coefficients, constants)
    gauss_seidel_solution = solver.solve_gauss_seidel(coefficients, constants)
    jacobi_solution = solver.solve_jacobi(coefficients, constants)
    lu_solution = solver.solve_lu(coefficients, constants)

    print("Solutions:")
    print(f"Gaussian Elimination: {gaussian_solution}")
    print(f"Gauss-Seidel ({solver.iterations} iterations): {gauss_seidel_solution}")
    print(f"Jacobi ({solver.iterations} iterations): {jacobi_solution}")
    print(f"LU Decomposition: {lu_solution}")
