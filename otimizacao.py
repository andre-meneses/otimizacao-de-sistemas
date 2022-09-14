import numpy as np
import pandas as pd

def simplex(matriz,base,objetivo="max"):
    """
    Maximiza ou minimiza uma função a partir do método simplex. 

    @param matriz: Matriz cuja primeira linha corresponde à função objetivo, e cujas linhas restantes
    correspondem às restrições do problema. 
    @param base: Lista de inteiros, representam a base inicial do problema. 
    @param objetivo: string, indica que se a otimização é de maximização ou minimização

    @returns matriz da solução e a base

    """
    if objetivo == "min":
        matriz[0] = [-1 * matriz[0][j] for j in range(len(matriz[0]))] 

    return simplexMax(matriz,base)

def simplexMax(matriz, base):
    """
    Maximiza uma função a partir do método simplex.

    @param matriz: Matriz cuja primeira linha corresponde à função objetivo, e cujas linhas restantes
    correspondem às restrições do problema. 
    @param base: Lista de inteiros, representam a base inicial do problema. 

    @returns vetor de soluções. 
    """

    funcaoObjetivo = np.array(matriz[0])
    novaBase = np.argmin(funcaoObjetivo)

    matrizPasso = []

    if funcaoObjetivo[novaBase] < 0:

        fatoresLimitantes = fatorLimitante(matriz,novaBase)

        indiceAntigaBase = np.argmin(fatoresLimitantes)
        antigaBase = base[indiceAntigaBase]

        matrizPasso = eliminacaoPivotal(matriz,indiceAntigaBase + 1,novaBase)

        base[indiceAntigaBase] = novaBase

        return simplexMax(matrizPasso,base)
    else:
        return matriz,base


def eliminacaoPivotal(matriz,indiceAntigaBase,novaBase):
    novaMatriz = []

    pivo = matriz[indiceAntigaBase][novaBase]

    for i in range(len(matriz)):
        if i != indiceAntigaBase and matriz[i][novaBase] != 0:
            m = matriz[i][novaBase]/pivo
            novaMatriz.append([matriz[i][x] - m*matriz[indiceAntigaBase][x] for x in range(len(matriz[0]))])
        else:
            novaMatriz.append(matriz[i])

    return novaMatriz

def fatorLimitante(matriz,base):
    q = []

    for i in range(1,len(matriz)):

        b = matriz[i][len(matriz[0])-1]
        x = matriz[i][base]

        if x != 0 and b > 0 and x > 0:
            q.append(b/x)
        else:
            q.append(np.inf)

    return q

def gerarTabela(matriz,base):
    tabela = {}
    tabela["Base"] = []

    for i in range(len(base)):
        tabela["Base"].append("x{j}".format(j=base[i])) 

    for i in range(len(matriz) - len(base)):
        tabela["Base"].insert(0,"---")

    for i in range(len(matriz[0])):
        if i == 0:
            nomeColuna = "z"
        elif i == len(matriz[0])-1:
            nomeColuna = "b"
        else:
            nomeColuna = "x{j}".format(j=i)
        
        tabela[nomeColuna] = [matriz[j][i] for j in range(len(matriz))]

    return pd.DataFrame(tabela)

if __name__ == '__main__':
    a = [[1,-12,-15,0,0,0,0,0],[0,1,0,1,0,0,0,3],[0,0,1,0,1,0,0,4],[0,1,1,0,0,1,0,6],[0,1,3,0,0,0,1,13]]
    c = simplexMax(a,[3,4,5,6])

    print(gerarTabela(c[0],c[1])) 

