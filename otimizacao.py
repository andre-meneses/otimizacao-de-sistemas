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
        matrix[0] = [-1 * matrix[0][j] for j in range(len(matrix[0]))]

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

    objective_function = np.array(matrix[0][:-1])
    new_base = np.argmin(objective_function)

    step_matrix = []

    if objective_function[new_base] < 0:

        limiting_factors = limiting_factor(matrix, new_base)

        index_old_base = np.argmin(limiting_factors)
        old_base = base[index_old_base]

        step_matrix = pivotal_elimination(matrix, index_old_base + 1, new_base)

        base[index_old_base] = new_base
        print(generate_table(step_matrix, base))

        return simplex_maximize(step_matrix, base)
    else:
        return matrix, base

def pivotal_elimination(matrix, old_base_index, new_base):
    new_matrix = []

    pivot = matrix[old_base_index][new_base]

    for i in range(len(matrix)):
        if i != old_base_index and matrix[i][new_base] != 0:
            m = matrix[i][new_base] / pivot
            new_matrix.append([matrix[i][x] - m * matrix[old_base_index][x] for x in range(len(matrix[0]))])
        else:
            new_matrix.append(matrix[i])

    return new_matrix

def limiting_factor(matrix, base):
    q = []

    for i in range(1, len(matrix)):

        b = matrix[i][len(matrix[0]) - 1]
        x = matrix[i][base]

        if x != 0 and b >= 0 and x > 0:
            q.append(b / x)
        else:
            q.append(np.inf)

    return q

def generate_table(matrix, base):
    table = {}
    table["Base"] = []

    for i in range(len(base)):
        table["Base"].append("x{j}".format(j=base[i]))

    for i in range(len(matrix) - len(base)):
        table["Base"].insert(0, "---")

    for i in range(len(matrix[0])):
        if i == 0:
            column_name = "z"
        elif i == len(matrix[0]) - 1:
            column_name = "b"
        else:
            column_name = "x{j}".format(j=i)

        table[column_name] = [matrix[j][i] for j in range(len(matrix))]

    return pd.DataFrame(table)

def transportation_matrix(cost_matrix, supply, demand):
    matrix = []

    # Objective function
    matrix.append([1] + [cost_matrix[j][i] for i in range(len(demand)) for j in range(len(supply))] + [0])

    # Demand equations
    for i in range(len(demand)):
        mini = i * len(supply)
        maxi = mini + len(supply)
        matrix.append([0] + [1 if mini <= j < maxi else 0 for j in range(len(matrix[0]) - 2)] + [demand[i]])

    # Supply equations
    for i in range(len(supply)):
        matrix.append([0] + [1 if (j % len(supply)) == i else 0 for j in range(len(matrix[0]) - 2)] + [supply[i]])

    return matrix

def canonical_form(matrix, pivots):
    new_matrix = matrix

    for i, j in pivots:
        new_matrix = pivotal_elimination(new_matrix, i, j)

    return new_matrix

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

