import numpy as np
import pandas as pd
from typing import List, Tuple, Union, Optional

def maximize_simplex(
    matrix: List[List[float]], 
    base: List[int], 
    objective: str = "max"
) -> Tuple[List[List[float]], List[int]]:
    """
    Maximizes or minimizes a function using the simplex method.
    
    Parameters
    ----------
    matrix : List[List[float]]
        A matrix where the first row corresponds to the objective function,
        and the remaining rows correspond to the problem's constraints.
    base : List[int]
        A list of integers representing the initial basis of the problem.
    objective : str, optional
        Optimization direction ("max" or "min"), by default "max"
        
    Returns
    -------
    Tuple[List[List[float]], List[int]]
        A tuple containing the solution matrix and the final basis.
    
    Examples
    --------
    >>> matrix = [[1, -2, -3, 0, 0, 0], [1, 1, 0, 1, 0, 4], [0, 0, 1, 0, 1, 6]]
    >>> base = [3, 4]
    >>> solution, final_base = maximize_simplex(matrix, base)
    """
    if objective == "min":
        matrix[0] = [-x for x in matrix[0]]
    
    return _simplex_maximize(matrix, base)

def _simplex_maximize(
    matrix: List[List[float]], 
    base: List[int]
) -> Tuple[List[List[float]], List[int]]:
    """
    Internal function that implements the simplex maximization algorithm.
    
    Parameters
    ----------
    matrix : List[List[float]]
        The simplex tableau
    base : List[int]
        Current basis variables
        
    Returns
    -------
    Tuple[List[List[float]], List[int]]
        Updated tableau and basis
    """
    while True:
        obj_func = np.array(matrix[0][:-1])
        new_base = np.argmin(obj_func)
        
        if obj_func[new_base] >= 0:
            return matrix, base
            
        limiting_factors = [
            b / matrix[i][new_base] if matrix[i][new_base] > 0 else np.inf 
            for i, b in enumerate(matrix[1:])
        ]
        
        idx_old_base = limiting_factors.index(min(limiting_factors))
        matrix = _pivotal_elimination(matrix, idx_old_base + 1, new_base)
        base[idx_old_base] = new_base
        
        print(_generate_table(matrix, base))

def _pivotal_elimination(
    matrix: List[List[float]], 
    row_idx: int, 
    col_idx: int
) -> List[List[float]]:
    """
    Performs pivotal elimination to update the simplex tableau.
    
    Parameters
    ----------
    matrix : List[List[float]]
        Current tableau
    row_idx : int
        Pivot row index
    col_idx : int
        Pivot column index
        
    Returns
    -------
    List[List[float]]
        Updated tableau after pivotal elimination
    """
    pivot = matrix[row_idx][col_idx]
    matrix[row_idx] = [x / pivot for x in matrix[row_idx]]
    
    for i in range(len(matrix)):
        if i != row_idx and matrix[i][col_idx] != 0:
            factor = matrix[i][col_idx]
            matrix[i] = [
                x - factor * y 
                for x, y in zip(matrix[i], matrix[row_idx])
            ]
    
    return matrix

def _generate_table(
    matrix: List[List[float]], 
    base: List[int]
) -> pd.DataFrame:
    """
    Generates a formatted table for the current state of the simplex method.
    
    Parameters
    ----------
    matrix : List[List[float]]
        Current tableau
    base : List[int]
        Current basis variables
        
    Returns
    -------
    pd.DataFrame
        Formatted simplex tableau
    """
    table = {
        "Base": [f"x{j}" for j in base] + 
               ["---" for _ in range(len(matrix) - len(base))]
    }
    
    for i in range(len(matrix[0])):
        col_name = ("z" if i == 0 else 
                   "b" if i == len(matrix[0]) - 1 else 
                   f"x{i}")
        table[col_name] = [matrix[j][i] for j in range(len(matrix))]
    
    return pd.DataFrame(table)

