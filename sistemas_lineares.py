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

# Resolução retroativa: 
#entrada - Matriz de coeficientes triangular superior A e vetor de termos independentes b.
#saída - Solução x do sistema

def resolucao_retroativa(coeficientes, termos_independentes):

    solucao = vetor_nulo(len(coeficientes)) 
    ultima_linha = len(coeficientes) - 1

    for i in range(ultima_linha,-1 , -1):
        soma = 0
        for j in range(len(coeficientes[i])):
            soma += coeficientes[i][j]*solucao[j]
        solucao[i] = (termos_independentes[i] - soma)/coeficientes[i][i]

    return solucao 

#Matriz triangular: Transforma uma matriz qualquer em matriz triangular superior

def matriz_triangular(matriz):
        for i in range(len(matriz) - 1):
            pivo = matriz[i][i]

            for j in range(i + 1, len(matriz)):
                m = matriz[j][i]/pivo

                def operacao(x,y):
                    return x - m*y

                matriz[j] = operacao_linha(matriz[j],matriz[i],operacao)
        
#Método de Gauss: Gera a matriz ampliada, chama a função matriz_triangular,
#em seguida a função resolução_retroativa.
#Entrada: Matriz de coeficientes e matriz de termos independentes 
#Saída: Solução x do sistema

def metodo_gauss(coeficientes, termos_independentes):
   
    b = []
    for i in range(len(coeficientes)):
        coeficientes[i].append(termos_independentes[i])

    matriz_triangular(coeficientes)

    for i in range(len(coeficientes)):
        b.append(coeficientes[i].pop())

    return resolucao_retroativa(coeficientes,b)

#Método auxiliar a ser usado na implementação dos algoritmos de Jacobi e Gauss-Seidel

def metodo_iterativo_aux(coeficientes, termos_independentes, b,d):
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
    
#Método de jacobi:
#Entrada: Matriz de coeficientes, termos independents, e uma solução inicial
#saída: Resolução x do sistema 

def jacobi(coeficientes,termos_independentes,solucao):
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
    
#Decomposição LU
def decomposicao(coeficientes):
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


