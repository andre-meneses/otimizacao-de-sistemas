# interpolation.py

from typing import List, Callable, Tuple, Union, Optional
import numpy as np
from numpy.typing import ArrayLike
from dataclasses import dataclass

@dataclass
class InterpolationPoint:
    """Represents a point for interpolation."""
    x: float
    y: float

class InterpolationError(Exception):
    """Custom exception for interpolation-related errors."""
    pass

class Interpolator:
    """
    A class implementing various interpolation methods.
    
    Methods:
    - Newton's divided differences interpolation
    - Lagrange interpolation
    
    Examples
    --------
    >>> points = [(1, 1), (2, 4), (3, 9)]  # x², f(x) = x²
    >>> interpolator = Interpolator.from_points(points)
    >>> interpolator.newton_interpolate()(2.5)  # ≈ 6.25
    """
    
    def __init__(self, x: List[float], y: List[float]):
        """
        Initialize interpolator with x and y coordinates.
        
        Parameters
        ----------
        x : List[float]
            List of x coordinates
        y : List[float]
            List of y coordinates
            
        Raises
        ------
        InterpolationError
            If input lists have different lengths or are empty
        """
        if len(x) != len(y):
            raise InterpolationError("x and y lists must have the same length")
        if len(x) == 0:
            raise InterpolationError("Cannot interpolate with empty lists")
        
        self.x = x
        self.y = y
        self._newton_coefficients: Optional[List[float]] = None
    
    @classmethod
    def from_points(cls, points: List[Tuple[float, float]]) -> 'Interpolator':
        """
        Create interpolator from list of (x, y) tuples.
        
        Parameters
        ----------
        points : List[Tuple[float, float]]
            List of (x, y) coordinate pairs
            
        Returns
        -------
        Interpolator
            New interpolator instance
        """
        x = [p[0] for p in points]
        y = [p[1] for p in points]
        return cls(x, y)

    def _compute_divided_differences(self) -> List[float]:
        """
        Compute divided differences table for Newton interpolation.
        
        Returns
        -------
        List[float]
            Coefficients for Newton's interpolation formula
        """
        n = len(self.x)
        coef = self.y[:]  # Create a copy of y values
        
        for j in range(1, n):
            for i in range(n - 1, j - 1, -1):
                coef[i] = (coef[i] - coef[i-1]) / (self.x[i] - self.x[i-j])
        
        return coef

    def _get_newton_coefficients(self) -> List[float]:
        """
        Get or compute Newton interpolation coefficients.
        
        Returns
        -------
        List[float]
            Coefficients for Newton's interpolation formula
        """
        if self._newton_coefficients is None:
            self._newton_coefficients = self._compute_divided_differences()
        return self._newton_coefficients

    def newton_interpolate(self) -> Callable[[float], float]:
        """
        Create Newton interpolation function.
        
        Returns
        -------
        Callable[[float], float]
            Function that computes interpolated value for any x
        
        Examples
        --------
        >>> interp = Interpolator([0, 1, 2], [1, 2, 4])
        >>> f = interp.newton_interpolate()
        >>> f(1.5)  # Returns interpolated value at x=1.5
        """
        coef = self._get_newton_coefficients()
        
        def interpolant(x: float) -> float:
            """Evaluate Newton interpolation polynomial at x."""
            n = len(self.x)
            result = coef[n-1]
            
            for i in range(n-2, -1, -1):
                result = result * (x - self.x[i]) + coef[i]
            
            return result
        
        return interpolant

    def _lagrange_basis(self, i: int) -> Callable[[float], float]:
        """
        Compute i-th Lagrange basis polynomial.
        
        Parameters
        ----------
        i : int
            Index of the basis polynomial
            
        Returns
        -------
        Callable[[float], float]
            i-th Lagrange basis polynomial
        """
        def numerator(x: float) -> float:
            points = list(range(i)) + list(range(i + 1, len(self.x)))
            return np.product([(x - self.x[j]) for j in points])
        
        def denominator() -> float:
            points = list(range(i)) + list(range(i + 1, len(self.x)))
            return np.product([(self.x[i] - self.x[j]) for j in points])
        
        return lambda x: numerator(x) / denominator()

    def lagrange_interpolate(self) -> Callable[[float], float]:
        """
        Create Lagrange interpolation function.
        
        Returns
        -------
        Callable[[float], float]
            Function that computes interpolated value for any x
        
        Examples
        --------
        >>> interp = Interpolator([0, 1, 2], [1, 2, 4])
        >>> f = interp.lagrange_interpolate()
        >>> f(1.5)  # Returns interpolated value at x=1.5
        """
        basis_polynomials = [self._lagrange_basis(i) for i in range(len(self.x))]
        
        def interpolant(x: float) -> float:
            """Evaluate Lagrange interpolation polynomial at x."""
            return sum(self.y[i] * basis_polynomials[i](x) for i in range(len(self.x)))
        
        return interpolant

    def evaluate_error(self, true_function: Callable[[float], float], points: List[float]) -> List[float]:
        """
        Compute interpolation error at given points.
        
        Parameters
        ----------
        true_function : Callable[[float], float]
            The actual function being interpolated
        points : List[float]
            Points at which to evaluate the error
            
        Returns
        -------
        List[float]
            List of absolute errors at each point
        """
        interpolant = self.newton_interpolate()  # Could also use Lagrange
        return [abs(true_function(x) - interpolant(x)) for x in points]

def demo_interpolation():
    """Demonstrate usage of the Interpolator class."""
    # Example: Interpolate f(x) = x²
    x_points = [-1, 0, 1, 2]
    y_points = [x**2 for x in x_points]
    
    # Create interpolator
    interpolator = Interpolator(x_points, y_points)
    
    # Create interpolation functions
    newton_func = interpolator.newton_interpolate()
    lagrange_func = interpolator.lagrange_interpolate()
    
    # Evaluate at some test points
    test_points = [-0.5, 0.5, 1.5]
    print("\nInterpolation Results:")
    print("x\tTrue\tNewton\tLagrange")
    print("-" * 40)
    
    for x in test_points:
        true_val = x**2
        newton_val = newton_func(x)
        lagrange_val = lagrange_func(x)
        print(f"{x:.1f}\t{true_val:.4f}\t{newton_val:.4f}\t{lagrange_val:.4f}")
    
    # Compute and display errors
    true_func = lambda x: x**2
    errors = interpolator.evaluate_error(true_func, test_points)
    print("\nAbsolute Errors:")
    for x, err in zip(test_points, errors):
        print(f"x = {x:.1f}: {err:.2e}")

if __name__ == "__main__":
    demo_interpolation()
