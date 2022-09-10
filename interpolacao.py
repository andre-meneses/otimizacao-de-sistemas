import numpy as np

def diferencas_divididas(x,y):

    """Para uma lista xn e yn, retorna a diferença dividida de ordem n

       @param x: lista de x
       @param y: lista de y
       @returns: Diferença dividida

    """
    if len(x) == 1:
        return y[0]

    a = diferencas_divididas(x[1:],y[1:])
    b = diferencas_divididas(x[:len(x) - 1],y[:len(y) - 1])

    return (a - b)/(x[len(y) - 1] - x[0])
    
def coeficientes_newton(x,y):
    """ Calcula os coeficientes para interpolação de newton a partir da função de diferencas_divididas
        @param x: lista de x
        @param y: lista de y
        @returns: Coeficientes interpolação de Newton
    """

    coef = [] 
    for i in range(1,len(x) + 1):
        coef.append(diferencas_divididas(x[:i],y[:i])) 
    return coef

def interpolacao_newton(x,y):
    """A partir dos coeficientes, retorna uma função que calcula o valor de y para um dado x, a partir da interpolação
    @param x: lista de x
    @param y: lista de y
    @returns: funcao interpoladora
    """
    coef = coeficientes_newton(x,y)[1:]

    f = []

    def termos(i):
        l = x[:i]
        return lambda x: np.product([(x-x0) for x0 in l])
    
    for i in range(1,len(x)):
        f.append(termos(i))

    return lambda p: y[0] + sum([f[i](p) * coef[i] for i in range(len(x) - 1)])

def coeficiente_lagrange(x,n):
    """Calcula os coeficientes da interpolação de lagrange
    @param x: lista de x
    @param n: termo
    @returns: Função anônima que corresponde a cada termo da interpolação de lagrange
    """
    def numerador(i):
        l = x[:i] + x[i + 1:]
        return lambda x: np.product([(x-x0) for x0 in l])
    
    def denomidador(i):
        l = x[:i] + x[i + 1:]
        return np.product([(x[i] - x0) for x0 in l])

    return lambda x: numerador(n)(x)/denomidador(n)

def interpolacao_lagrange(x,y):
    """Retorna a função de interpolação de lagrange
    @param x: lista de x
    @param y: lista de y
    @returns: Função que corresponde à interpolação
    """

    return lambda p: sum([y[i]*coeficiente_lagrange(x,i)(p) for i in range(len(x))])

if __name__ == '__main__':
    x = [-1,2,4]
    y = [3,6,-2]
    #a = interpolacao(x,y)
    a = interpolacao_lagrange(x,y) 
    print(a(4))
    #print(coef)
    #a0 = a[0](2.5)
    #a1 = a[1](2.5)
    #print(a[0](2.5),a[1](2.5))
    #print(y[0] + coef[0]*a0 + coef[1]*a1)
      
