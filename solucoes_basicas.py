import numpy as np
import pandas as pd
import sistemas_lineares

from itertools import combinations

def solucoesBasicas(matriz):
    """
    Retorna todas as soluções básicas compatíveis de um sistema com mais variáveis 
    que equações.

    @param matriz: Pandas dataframe que representa a matriz do sistema
    @returns: Pandas dataframe das soluções básicas
    """ 
    variaveisLivres =  len(matriz.columns) - len(matriz)
    index = list(matriz.columns)
    combinacoes = list(combinations(index,variaveisLivres))

    termos_indepentes = matriz['y']

    solucoes = {}


if __name__=='__main__':
    data = {"x1":[1,2],"x2":[8,7],"y":[11,12]}
    print(data)
