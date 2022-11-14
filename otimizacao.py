import numpy as np
import pandas as pd

def simplex(matriz,base,objetivo="max"):
    """
    Maximiza ou minimiza uma função a partir do método simplex. 

    @param matriz: Matriz cuja primeira linha corresponde à função objetivo, e cujas linhas restantes
    correspondem às restrições do problema. 
    @param base: Lista de inteiros, representam a base inicial do problema. 
    @param objetivo: string, indica se a otimização é de maximização ou minimização

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

    @returns matriz da solução e base
    """

    funcaoObjetivo = np.array(matriz[0][:-1])
    novaBase = np.argmin(funcaoObjetivo)

    matrizPasso = []

    if funcaoObjetivo[novaBase] < 0:

        fatoresLimitantes = fatorLimitante(matriz,novaBase)

        indiceAntigaBase = np.argmin(fatoresLimitantes)
        antigaBase = base[indiceAntigaBase]

        matrizPasso = eliminacaoPivotal(matriz,indiceAntigaBase + 1,novaBase)

        base[indiceAntigaBase] = novaBase
        print(gerarTabela(matrizPasso,base))

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

        if x != 0 and b >= 0 and x > 0:
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

def matrizTransporte(quadroCusto,oferta,demanda):
    matriz = []

    # Função objetivo
    matriz.append([1] + [quadroCusto[j][i] for i in range(len(demanda)) for j in range(len(oferta))] + [0])

    #Equações de demanda
    for i in range(len(demanda)):
        mini = i*len(oferta)
        maxi = mini + len(oferta)
        matriz.append([0] + [1 if mini <= j < maxi else 0 for j in range(len(matriz[0])-2)] + [demanda[i]])        

    # Equações de oferta
    for i in range(len(oferta)):
        matriz.append([0] + [1 if (j % len(oferta)) == i else 0 for j in range(len(matriz[0])-2)] + [oferta[i]])
    
    return matriz

def formaCanonica(matriz,pivos):
    novaMatriz = matriz

    for i,j in pivos:
        novaMatriz = eliminacaoPivotal(novaMatriz,i,j)

    return novaMatriz

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

