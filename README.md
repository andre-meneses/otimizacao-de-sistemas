# Numerical Computing and Optimization Toolkit

A Python library implementing various numerical computing algorithms, optimization methods, and interpolation techniques. This project was developed as an educational tool to understand numerical algorithms and optimization methods. It is **not** intended for production use or performance-critical applications. Important considerations:

- No performance optimizations have been implemented
- Algorithms are written for clarity and learning purposes rather than efficiency
- Large-scale problems should be solved using established libraries
- The implementation focuses on demonstrating the mathematical concepts
- Error handling may not be comprehensive enough for production use
- Memory usage is not optimized

For production applications, please consider using established libraries:
- Scientific Computing: NumPy, SciPy
- Linear Programming: CPLEX, Gurobi, or scipy.optimize
- Linear Systems: NumPy, SciPy
- Interpolation: SciPy's interpolate module

## Features

### Linear Programming
- Simplex method implementation
- Transportation problem solver
- Basic sensitivity analysis

### Linear Systems
- Direct Methods:
  - Gaussian elimination
  - LU decomposition
- Iterative Methods:
  - Gauss-Seidel iteration
  - Jacobi iteration
- Error analysis and convergence checking

### Interpolation
- Newton's divided differences method
- Lagrange interpolation
- Error evaluation tools

## Installation

```bash
# Clone the repository
git clone https://github.com/yourusername/numerical-toolkit.git

# Install dependencies
pip install -r requirements.txt
```

## Usage Examples

### Linear Programming with Simplex Method

```python
from optimization import maximize_simplex

# Maximize z = 3x₁ + 2x₂
# Subject to:
#   2x₁ + x₂ ≤ 100
#   x₁ + x₂ ≤ 80
#   x₁, x₂ ≥ 0

matrix = [
    [1, -3, -2, 0, 0, 0],      # objective function
    [0, 2, 1, 1, 0, 100],      # first constraint
    [0, 1, 1, 0, 1, 80]        # second constraint
]
base = [3, 4]  # initial basis (slack variables)

solution, final_base = maximize_simplex(matrix, base)
print(f"x₁ = {solution[1][2]}")
print(f"x₂ = {solution[1][3]}")
print(f"Maximum value = {solution[0][0]}")
```

### Linear Systems Solution

```python
from linear_systems import LinearSystemSolver

# Solve system: 
# 4x + y - z = 3
# 2x + 7y + z = 19
# x - 3y + 12z = 31

coefficients = [
    [4, 1, -1],
    [2, 7, 1],
    [1, -3, 12]
]
constants = [3, 19, 31]

solver = LinearSystemSolver()

# Choose your preferred method:
solution_gaussian = solver.solve_gaussian(coefficients, constants)
solution_gauss_seidel = solver.solve_gauss_seidel(coefficients, constants)
solution_jacobi = solver.solve_jacobi(coefficients, constants)
solution_lu = solver.solve_lu(coefficients, constants)
```

### Interpolation

```python
from interpolation import Interpolator

# Create interpolation points
x = [-1, 0, 1, 2]
y = [1, 0, 1, 4]  # Example data points

interpolator = Interpolator(x, y)

# Get interpolation functions
newton_func = interpolator.newton_interpolate()
lagrange_func = interpolator.lagrange_interpolate()

# Evaluate at a point
x_new = 0.5
y_newton = newton_func(x_new)
y_lagrange = lagrange_func(x_new)

# Evaluate error if true function is known
def true_function(x): return x**2
errors = interpolator.evaluate_error(true_function, [x_new])
```

## Implementation Details

### Optimization Methods
- Basic simplex implementation for linear programming
- Transportation problem solver using simplex method
- Basic sensitivity analysis capabilities

### Linear Systems
- Direct methods with partial pivoting
- Iterative methods with convergence checking
- Error analysis and residual computation
- Support for ill-conditioned systems

### Interpolation
- Newton's divided differences with coefficient caching
- Lagrange interpolation with basis function computation
- Error evaluation tools
- Support for unevenly spaced points

## Future Improvements

### Non-Linear Programming
- Gradient descent methods
- Interior Point Methods
- Support for various non-linear constraints
- Integration with automatic differentiation

### Additional Optimization Methods
- Integer Programming
- Mixed Integer Linear Programming (MILP)
- Dynamic Programming

## Authors

* **André Meneses** - *Initial work*

