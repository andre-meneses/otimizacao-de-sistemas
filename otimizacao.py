import numpy as np
import pandas as pd

def maximize_simplex(matrix, base, objective="max"):
    """
    Maximizes or minimizes a function using the simplex method.

    Args:
        matrix (list of lists): A matrix where the first row corresponds to the objective function,
            and the remaining rows correspond to the problem's constraints.
        base (list of integers): A list of integers representing the initial basis of the problem.
        objective (str): A string indicating whether the optimization is for maximization or minimization.

    Returns:
        tuple: A tuple containing the solution matrix and the basis.
    """
    if objective == "min":
        matrix[0] = [-x for x in matrix[0]]

    return simplex_maximize(matrix, base)

def simplex_maximize(matrix, base):
    """
    Maximizes a function using the simplex method.

    Args:
        matrix (list of lists): A matrix where the first row corresponds to the objective function,
            and the remaining rows correspond to the problem's constraints.
        base (list of integers): A list of integers representing the initial basis of the problem.

    Returns:
        tuple: A tuple containing the solution matrix and the basis.
    """

    while True:
        objective_function = np.array(matrix[0][:-1])
        new_base = np.argmin(objective_function)

        if objective_function[new_base] >= 0:
            return matrix, base

        limiting_factors = [b / matrix[i][new_base] if matrix[i][new_base] > 0 else np.inf for i, b in enumerate(matrix[1:])]

        index_old_base = limiting_factors.index(min(limiting_factors))
        old_base = base[index_old_base]

        matrix = pivotal_elimination(matrix, index_old_base + 1, new_base)
        base[index_old_base] = new_base
        print(generate_table(matrix, base))

def pivotal_elimination(matrix, old_base_index, new_base):
    """
    Performs pivotal elimination to update the matrix during the simplex method.

    Args:
        matrix (list of lists): The current matrix.
        old_base_index (int): Index of the row corresponding to the old basis.
        new_base (int): Index of the column corresponding to the new basis.

    Returns:
        list of lists: The updated matrix.
    """
    pivot = matrix[old_base_index][new_base]
    for i in range(len(matrix)):
        if i != old_base_index and matrix[i][new_base] != 0:
            m = matrix[i][new_base] / pivot
            matrix[i] = [x - m * matrix[old_base_index][x] for x in range(len(matrix[0]))]

    return matrix

def generate_table(matrix, base):
    """
    Generates a table for the current state of the simplex method.

    Args:
        matrix (list of lists): The current matrix.
        base (list of integers): The current basis.

    Returns:
        pd.DataFrame: A Pandas DataFrame representing the table.
    """
    table = {"Base": [f"x{j}" for j in base] + ["---" for _ in range(len(matrix) - len(base))]}

    for i in range(len(matrix[0])):
        column_name = "z" if i == 0 else "b" if i == len(matrix[0]) - 1 else f"x{i}"
        table[column_name] = [matrix[j][i] for j in range(len(matrix))]

    return pd.DataFrame(table)

def transportation_matrix(cost_matrix, supply, demand):
    """
    Constructs the transportation matrix for the transportation problem.

    Args:
        cost_matrix (list of lists): A matrix representing the costs of transporting from suppliers to demanders.
        supply (list of integers): A list representing the supply capacities of suppliers.
        demand (list of integers): A list representing the demand quantities of demanders.

    Returns:
        list of lists: The transportation matrix.
    """
    supply_indices = np.repeat(range(len(supply)), len(demand))
    demand_indices = np.tile(range(len(demand)), len(supply))
    matrix = [[cost_matrix[s][d] for s, d in zip(supply_indices, demand_indices)]]

    # Objective function
    matrix.append([1] + [0] * (len(supply) * len(demand)) + [0])

    # Demand equations
    for i in range(len(demand)):
        matrix.append([0] if i == 0 else [0] * (i * len(supply)) + [1] + [0] * ((len(demand) - i - 1) * len(supply)) + [0, demand[i]])

    # Supply equations
    for i in range(len(supply)):
        matrix.append([0] if i == 0 else [0] * i + [1] + [0] * (len(supply) - i - 1) + [0, supply[i]])

    return matrix

def canonical_form(matrix, pivots):
    """
    Converts the matrix to canonical form using specified pivotal elements.

    Args:
        matrix (list of lists): The current matrix.
        pivots (list of tuples): A list of (row, column) tuples specifying pivotal elements.

    Returns:
        list of lists: The matrix in canonical form.
    """
    for i, j in pivots:
        pivot = matrix[i][j]
        matrix[i] = [x / pivot for x in matrix[i]]
        for k in range(len(matrix)):
            if k != i:
                factor = matrix[k][j]
                matrix[k] = [x - factor * y for x, y in zip(matrix[k], matrix[i])]

    return matrix

if __name__ == '__main__':
    # a = [[1,-12,-15,0,0,0,0,0],[0,1,0,1,0,0,0,3],[0,0,1,0,1,0,0,4],[0,1,1,0,0,1,0,6],[0,1,3,0,0,0,1,13]]
    #c = simplexMax(a,[3,4,5,6])

    # b = [[1, -2, -3, 0, 0, 0, 0],[0, 1, 0, 1, 0, 0, 3],[0, 0, 1, 0, 1, 0, 4],[0, 1, 3, 0, 0, 1, 12]]
    # d = simplexMax(b,[3,4,5])


    # print(gerarTabela(d[0],d[1])) 

    demanda = [50,60,70,80]
    oferta = [60,90,110]
    quadroCusto = [[25,20,15,25],[15,20,500,10],[10,15,20,25]]
    base = [1,4,5,8,9,12]

    m = matrizTransporte(quadroCusto,oferta,demanda)
    m = np.delete(m,4,axis=0)

    pivos = [(1,1),(2,4),(5,5),(3,8),(6,9),(4,12)]
    matrizSimplex = formaCanonica(m,pivos)

    #print(gerarTabela(m,[1,4,5,8,9,12]))

    res = simplexMax(matrizSimplex,base)
    print(gerarTabela(res[0],res[1]))

