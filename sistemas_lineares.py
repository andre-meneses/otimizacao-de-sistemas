import numpy as np
import matplotlib.pyplot as plt
from copy import deepcopy

nmax = 10000
tolerancia = 0.001
cont = 1

#Funções auxiliares -------------------------------------------

def produto_matricial(a,b):
    produto = []
    for i in range(len(a)):
        produto.append([])
        for j in range(len(b[0])):
            produto[i].append(0)
            for k in range(len(a[0])):
                produto[i][j] += a[i][k] * b[k][j]
    return produto

def erro_vetor(a,b):
    c = operacao_linha(b,a,lambda x,y: x - y)
    return norma_linha(c)/norma_linha(b) < tolerancia

def produto_interno(a,b):
    produto = 0
    for i in range(len(a)):
        produto += a[i] * b[i]
    return produto

def norma_euclidiana(vetor):
    return np.sqrt(produto_interno(vetor, vetor))

def norma_linha(vetor):
    vetor = map(lambda x: abs(x),vetor)
    return max(vetor)

def vetor_nulo(n):
    vetor = []
    for i in range(n):
        vetor.append(0)
    return vetor

def matriz_nula(n,m):
    matriz = []
    for i in range(n):
        matriz.append(vetor_nulo(m))

    return matriz

def operacao_linha(a,b,operacao):
    resultado = []
    for i in range(len(a)):
        resultado.append(operacao(a[i],b[i]))
    return resultado

def residuo(coeficientes, termos_independentes, solucao):
    res = []
    for i in range(len(coeficientes)):
        res.append(termos_independentes[i] - produto_interno(coeficientes[i],solucao))
    
    return res
#---------------------------------------------------------------------------------------

def resolucao_retroativa(coeficientes, termos_independentes):
    """
    A partir de uma matriz triangular superior, retornar a solução do sistema por meio da resolução retroativa. 

    @param coeficientes: Matriz triangular superior dos coeficientes
    @param termos_independentes: Vetor de termoos independentes
    @Returns Vetor de soluções do sistema
    """
    solucao = vetor_nulo(len(coeficientes)) 
    ultima_linha = len(coeficientes) - 1

    for i in range(ultima_linha,-1 , -1):
        soma = 0
        for j in range(len(coeficientes[i])):
            soma += coeficientes[i][j]*solucao[j]
        solucao[i] = (termos_independentes[i] - soma)/coeficientes[i][i]

    return solucao 

def matriz_triangular(matriz):
    """Transforma uma matriz qualquer em uma matriz triangular superior

    @param Matriz de entrada
    """
    for i in range(len(matriz) - 1):
        pivo = matriz[i][i]

        for j in range(i + 1, len(matriz)):
            m = matriz[j][i]/pivo

            def operacao(x,y):
                return x - m*y

            matriz[j] = operacao_linha(matriz[j],matriz[i],operacao)
    
def metodo_gauss(coeficientes, termos_independentes):
    """
    Resolve um sistema linear a partir do método de gauss

    @param coeficientes: Matriz de coeficientes
    @param termos_independentes: Vetor de termos termos_independentes
    @Returns vetor de soluções do sistema
    """
    b = []
    for i in range(len(coeficientes)):
        coeficientes[i].append(termos_independentes[i])

    matriz_triangular(coeficientes)

    for i in range(len(coeficientes)):
        b.append(coeficientes[i].pop())

    return resolucao_retroativa(coeficientes,b)

def metodo_iterativo_aux(coeficientes, termos_independentes, b,d):
    """
    Método auxiliar a ser usado para os métodos iterativos de resolução de sistemas lineares

    @param coeficientes: Matriz de coeficientes
    @param termos_independentes: Vetor de termos independentes

    Os parametros 'b' e 'd' significam alguma coisa nos métodos de Gauss Seidel e de Jacobi, mas já não lembro o que.
    Por isso que as variáveis devem ter nomes significativo!
    """
    #Implementação da matriz 'b' e do vetor 'd'
    for i in range(len(coeficientes)):
        b.append([])
        d.append(termos_independentes[i]/coeficientes[i][i])
        for j in range(len(coeficientes[0])):
            if j == i:
                b[i].append(0)
            else:
                b[i].append(-coeficientes[i][j]/coeficientes[i][i])
    
    
def metodo_gauss_seidel(coeficientes,termos_independentes,solucao):
    """
    Método de Gauss-Seidel para a resolução de um sistema de equações lineares
    
    @param coeficientes: Matriz de coeficientes
    @param termos_independentes: Matriz de termos independentes.
    @param solucao: Solução inicial.
    @Returns: Vetor de soluções do sistema
    """
    b = []
    d = []
    nova_solucao = []
    dummy = [] 
    global cont 
    
    for i in range(len(solucao)):
        dummy.append(solucao[i])

    metodo_iterativo_aux(coeficientes,termos_independentes,b,d)
    
    #Cálculo do novo vetor solução
    for i in range(len(coeficientes)):
        nova_solucao.append(produto_interno(b[i],dummy) + d[i])
        dummy[i] = nova_solucao[i] 
    
    #A nova solução é satisfatória?
    if erro_vetor(solucao,nova_solucao) or cont == nmax:
        print("Iterações Gauss-Seidel: ", cont)
        return nova_solucao
    else:
        cont += 1
        return metodo_gauss_seidel(coeficientes, termos_independentes, nova_solucao)

def jacobi(coeficientes,termos_independentes,solucao):
    """
    Método de Jacobi para a resolução de um sistema de equações lineares
    
    @param coeficientes: Matriz de coeficientes
    @param termos_independentes: Matriz de termos independentes.
    @param solucao: Solução inicial.
    @Returns: Vetor de soluções do sistema
    """
    b = []
    d = []
    nova_solucao = []
    global cont 

    metodo_iterativo_aux(coeficientes,termos_independentes,b,d)

    #Cálculo do novo vetor solução
    for i in range(len(coeficientes)):
        nova_solucao.append(produto_interno(b[i],solucao) + d[i])
    
    #A nova solução é satisfatória?
    if erro_vetor(solucao,nova_solucao) or cont == nmax:
        print("Iterações Jacobi: ", cont)
        return nova_solucao
    else:
        cont += 1
        return jacobi(coeficientes, termos_independentes, nova_solucao)
    
def decomposicao(coeficientes):
    """
    Decomposicão LU de uma matriz

    @param coeficientes: Matriz de coeficientes
    @return: Matriz trianguglar inferior
    """
    triangular_inferior = matriz_nula(len(coeficientes), len(coeficientes[0])) 
    
    #Construção das matrizes upper e lower

    for i in range(len(coeficientes)):
        for j in range(len(coeficientes[0])):
            if i == j:
                triangular_inferior[i][j] = 1
            elif j > i:
                triangular_inferior[i][j] = 0

    for i in range(len(coeficientes) - 1):
            pivo = coeficientes[i][i]

            for j in range(i + 1, len(coeficientes)):
                m = coeficientes[j][i]/pivo
                triangular_inferior[j][i] = m 

                def operacao(x,y):
                    return x - m*y

                coeficientes[j] = operacao_linha(coeficientes[j],coeficientes[i],operacao)

    return triangular_inferior 

#------------------------------------------------------------------------------------------


