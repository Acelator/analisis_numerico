"""
Raices e interpolacion - metodos univariantes
Biseccion, regula falsi, secante, punto fijo y Newton.
Interpolacion: Newton (diferencias divididas), Horner, Chebyshev
y lineal a trozos. Analisis de epsilon maquina.
Ver docs/enunciados_resumidos.md#06_raices_interpolacion
"""

# Tema 1
from numpy import *
from matplotlib.pyplot import *

print("Caso 1")


def sumanveces(a, n):
    sum = 0
    while 1 <= n:
        sum += a
        n -= 1
    return sum


print("Caso 2")


def epsilon():
    x = 1
    while 1 + x > 1:
        x = x / 2
    return 2 * x


print("Caso 3")
# Caso referencia para graficas

# Crea el enviroment
figure()

# Informacion acerca del eje X:
#     Los dos primero param son el intervalo en cual queremos representar la funcion
#     El ultima la cantidad de "puntos" utilizadaos para la representacion
#       A mayor cantidad de puntos, mas "suave" sera la representacion
x = linspace(-2, 2, 100)
y = x - pow(e, -x)
plot(x, y, "k")

# Establece una linea horizontal en x=0, el 0*x es necesario
plot(x, x * 0, "g")

# O equivalentemente la siguiente instruccion
axhline(y=3, color="r")

xlabel("Eje x")
ylabel("Eje Y")
title("Ejemplo de un asombro titulo")


print("Caso 4")


def aproximateE(n):
    value = (1 + (1 / n)) ** n
    err = abs(value - exp(1))
    print("n = ", n, " value= = ", value)
    return (value, err)


print("Caso 5")


def sumaparcial(n):
    iter = n
    suma = 0

    # for k in range(1, n+1)

    while iter >= 1:
        suma += 1 / sqrt(iter)
        iter -= 1

    # print('n = ', n, ' value= = ', suma)
    return suma


n = 500
while sumaparcial(n) < 50:
    n += 1


print("Caso 6")
figure()

x = linspace(0, 100, 1)
y = sumaparcial(x)

for k in range(1, 101):
    plot(k, sumaparcial(k), "bo")


plot(x, y, "r")
plot(x, 0 * x, "b")
xlabel("Eje XS")
ylabel("Eje Y")


print("Caso 7")


def sumaparcialPrima(n):
    k = n
    suma = 0

    # for k in range(1, n+1)

    while k >= 1:
        suma += 1 / (k * (k + 1))
        k -= 1

    print("n = ", n, " value= = ", suma, "err= ", abs(1 - suma))
    return suma


def otraForma(n):
    return 1 - (1 / (n + 1))


# print("Probar caso 7: ", sumaparcialPrima(1e8))

print("Caso 8")


def factorial(n):
    ft = 1
    for k in range(1, n + 1):
        ft = ft * k

    return ft


def aproximarEConTaylor(n):
    k = n
    suma = 0

    while k >= 0:
        suma += 1 / (factorial(k))
        k -= 1

    print("n = ", n, " value= = ", suma)
    return suma


print("Caso 9")


def aproximarEpxConTaylor(x, n):
    k = n
    suma = 0

    # for k in range(1, n+1)

    while k >= 1:
        suma += x**k / factorial(k)
        k -= 1
    print("n = ", n, " value= = ", suma, "err= ", abs(1 - suma))
    return suma


print("Caso 10")


def sumaLista(lista):
    suma = 0
    for k in range(0, len(lista)):
        suma += lista[k]

    print("sum= ", suma, " media= ", suma / len(lista))
    print("sum= ", sum(lista), " media= ", mean(lista))
    return suma


# Ejemplos extra
x1 = linspace(0, pi, 10)
y1 = sin(x1)
plot(x1, y1, "o")
plot(x1, y1, "o-")
plot(x1, y1, "*r-")
xlabel("Eje x")
ylabel("Eje Y")


# Multiples funciones a la vez
figure()
x = linspace(0, pi, 200)
y = exp(x) * sin(10 * x)


# Cada dos elem (o tres si incluimos  kargs) constituyen una funcion a representar en la grafica
plot(x, y, "o-", x, x**2, "k", x, zeros(200))


#############################################################################################################
# Tema 2.1

# En el campus
print("Caso 1")


def bisec(f, a, b, N):
    an = a
    bn = b
    fan = f(an)
    fbn = f(bn)
    if fan == 0:
        print(str(a) + "es raiz de la funcion")
        return a
    elif fbn == 0:
        print(str(b) + "es raiz de la funcion")
        return b
    elif fan * fbn > 0:
        print("No hay cambio de signo: no se puede aplicar el metodo")
        return
    for k in range(N):
        cn = (an + bn) / 2.0
        fcn = f(cn)
        print("cn: " + str(cn) + ", fn: " + str(fcn))
        if fcn == 0:
            print(str(cn) + "es raiz de la funcion")
            return cn
        elif fan * fcn < 0:
            bn = cn
            fbn = fcn
        else:
            an = cn
            fan = fcn
    print("La aproximacion de la raiz tras " + str(N) + " iteraciones es " + str(cn))
    return cn


def funcB(x):
    return pow(x, 5) - 5 * pow(x, 3) + 1


def funC(x):
    return cos(x) - x


def aplicarApartadoUno(a, b):
    bisec(funC, a, b, 20)


x = linspace(-2, 2, 100)
figure()
plot(x, funC(x))
plot(x, 0 * x)


def bisecPrima(f, a, b, eps):
    # Parte entera = int()

    N = int((log(b - a) - log(eps)) / log(2)) + 1
    print(N)
    an = a
    bn = b
    fan = f(an)
    fbn = f(bn)

    if fan == 0:
        print(str(a) + "es raiz de la funcion")
        return a
    elif fbn == 0:
        print(str(b) + "es raiz de la funcion")
        return b
    elif fan * fbn > 0:
        print("No hay cambio de signo: no se puede aplicar el metodo")
        # return
    for k in range(N):
        cn = (an + bn) / 2.0
        fcn = f(cn)
        print("cn: " + str(cn) + ", fn: " + str(fcn))
        if fcn == 0:
            print(str(cn) + "es raiz de la funcion")
            return cn
        elif fan * fcn < 0:
            bn = cn
            fbn = fcn
        else:
            an = cn
            fan = fcn
    print("La aproximacion de la raiz tras " + str(N) + " iteraciones es " + str(cn))
    return cn


# 10^n = 1en


def regulaFalsi(f, a, b, eps, nMax):
    an = a
    bn = b
    fan = f(an)
    fbn = f(bn)
    if fan == 0:
        print(str(a) + "es raiz de la funcion")
        return a
    elif fbn == 0:
        print(str(b) + "es raiz de la funcion")
        return b
    elif fan * fbn > 0:
        print("No hay cambio de signo: no se puede aplicar el metodo")
        return

    error = eps + 1
    it = 0
    cn_old = an
    while error > eps and it < nMax:
        cn = bn - (bn - an) / (fbn - fan) * fbn
        fcn = f(cn)
        print("Iter:" + str(it) + ", cn: " + str(cn) + ", fn: " + str(fcn))
        error = abs(cn - cn_old)
        it += 1
        cn_old = cn
        if fcn == 0:
            print(str(cn) + "es raiz de la funcion")
            return cn
        elif fan * fcn < 0:
            bn = cn
            fbn = fcn
        else:
            an = cn
            fan = fcn

    if error <= eps:
        print("Alcanzada una aproximacion satisfactoria")
    else:
        print("Alcanzado numero maximo de iteraciones")
    print("La aproximacion de la raiz tras " + str(it) + " iteraciones es " + str(cn))
    return cn


def secante(f, x0, x1, eps, nMax):
    fx0 = f(x0)
    fx1 = f(x1)
    if fx0 == 0:
        print(str(x0) + "es raiz de la funcion")
        return x0
    elif fx1 == 0:
        print(str(x1) + "es raiz de la funcion")
        return x1

    error = eps + 1
    it = 0
    while error > eps and it < nMax:
        if fx0 == fx1:
            print("El metodo no se puede aplicar")
            return

        x2 = x1 - (x1 - x0) / (fx1 - fx0) * fx1
        fx2 = f(x2)
        print("Iter:" + str(it) + ", cn: " + str(x2) + ", fn: " + str(fx2))
        error = abs(x2 - x1)
        it += 1
        if fx2 == 0:
            print(str(x2) + "es raiz de la funcion")
            return x2

        x0 = x1
        x1 = x2
        fx0 = fx1
        fx1 = fx2

    if error <= eps:
        print("Alcanzada una aproximacion satisfactoria")
    else:
        print("Alcanzado numero maximo de iteraciones")

    print("La aproximacion de la raiz tras " + str(it) + " iteraciones es " + str(x2))
    return x2


def regulaFalsiPrima(f, x0, x1, eps, nMax):
    fx0 = f(x0)
    fx1 = f(x1)
    if fx0 == 0:
        print(str(x0) + "es raiz de la funcion")
        return x0
    elif fx1 == 0:
        print(str(x1) + "es raiz de la funcion")
        return x1

    error = eps + 1
    it = 0
    while error > eps and it < nMax:
        if fx0 == fx1:
            print("Metodo no posible poder aplicar")
            return

        x2 = x1 - (x1 - x0) / (fx1 - fx0) * fx1
        fx2 = f(x2)
        print("Iter:" + str(it) + ", cn: " + str(x2) + ", fn: " + str(fx2))
        error = abs(fx2)
        it += 1
        if fx2 == 0:
            print(str(x2) + "es raiz de la funcion")
            return x2

        x0 = x1
        x1 = x2
        fx0 = fx1
        fx1 = fx2

    if error <= eps:
        print("Alcanzada una aproximacion satisfactoria")
    else:
        print("Alcanzado numero maximo de iteraciones")
    print("La aproximacion de la raiz tras " + str(it) + " iteraciones es " + str(x2))
    return x2


#############################################################################################################
# Tema 2.2

print("Caso 1")


def puntofijo(g, x0, eps, nmax):
    err = eps + 1
    it = 0  # Contador iteraciones
    while err > eps and it < nmax:
        x1 = g(x0)
        it += 1
        err = abs(x0 - x1)
        x0 = x1

    if err <= eps:
        print("ALCANZADO CRITERIO de PARADA")
        print(
            "Tras ",
            it,
            " iteraciones la solucion obtenida es ",
            x1,
            " con ESTIMADO error ",
            err,
        )
    else:
        print(
            "Hemos ALCANZADO el NUMERO maximo de ITERACIONES sin encontrar EL PUNTO FIJO"
        )
        print(
            "Tras ",
            it,
            " iteraciones la solucion obtenida es ",
            x1,
            " con ESTIMADO error ",
            err,
        )

    return x0


def g1(x):
    return exp(-x)


def Gnewton(x):
    f = x - exp(-x)
    df = 1 + exp(-x)
    return x - f / df


puntofijo(g1, 0.5, 1e-7, 100)
puntofijo(Gnewton, 0.5, 1e-7, 100)


print("Caso III")


def kepler(x):
    K = 2 / 3
    alp = 0.093

    return K + (alp * sin(x))


def keplerNewton(x):
    K = 2 / 3
    alp = 0.093

    f = K - x + (alp * sin(x))
    df = -1 + (alp * cos(x))
    return x - f / df


puntofijo(kepler, 0.5, 1e-7, 100)

puntofijo(keplerNewton, 0.5, 1e-7, 100)


print("Caso IV")


def f(x):
    return cos(x)


def fNewton(x):
    f = cos(x) - x
    df = -sin(x) - 1
    return x - f / df


puntofijo(f, 0.5, 1e-7, 100)
puntofijo(fNewton, 0.5, 1e-7, 100)


print("CASO V")


def superF(x):
    return exp(5 * x ^ 3 - 1, 1 / 5)


def superFNewton(x):
    f = x**5 - 5 * (x**3) + 1
    df = 5 * (x**4) - 15 * (x**2)
    return x - f / df


figure()
x = linspace(0.3, 0.7, 100)
plot(x, superFNewton(x), x, x)
title("Caso 5.b indicacion")


print("Caso VI")


def puntoFijoPrima(f, g, x0, eps, nmax):
    err = eps + 1
    it = 0  # Contador iteraciones
    while err > eps and it < nmax:
        x1 = g(x0)
        it += 1
        err = abs(f(x1))
        x0 = x1

    if err <= eps:
        print("Hemos ALCANZADO el CRITERIO de PARADA")
        print(
            "Tras ",
            it,
            " iteraciones la solucion obtenida es ",
            x1,
            " con ESTIMADO error ",
            err,
        )
    else:
        print(
            "Hemos ALCANZADO el NUMERO maximo de ITERACIONES sin encontrar EL PUNTO FIJO"
        )
        print(
            "Tras ",
            it,
            " iteraciones la solucion obtenida es ",
            x1,
            " con ESTIMADO error ",
            err,
        )

    return x0


def g5(x):
    return ((x**5 + 1) / 5) ** (1 / 3)


def func(x):
    return x + (x - 1) * exp(x)


figure()
x = linspace(0, 1, 100)
plot(x, func(x))
axhline(y=0, color="r")
title("Caso 6.b")


def g(x):
    return -(x - 1) * exp(x)


def g_newton(x):
    f = x + (x - 1) * exp(x)
    df = 1 + x * exp(x)
    return x - f / df


puntofijo(g_newton, 0.6, 1e-8, 100)


###########################################################################################################################
# Tema 3
from scipy.interpolate import interp1d


# Funciones del campus virtual
def tabla_diferencias_divididas(x, y):
    """Calcula la tabla completa de las diferencias divididas a partir de los datos x e y.
    Devuelve una matriz (df) triangular inferior que en la columna k-esima contiene las
    diferencias divididas de orden k"""

    n = len(y)
    df = zeros([n, n])
    df[:, 0] = y
    yn = y
    for i in range(0, len(x) - 1):
        dx = x[i + 1 : len(x)] - x[0 : n - (i + 1)]
        yn = diff(yn) / dx
        df[i + 1 : n, i + 1] = yn
    return df


def eval_forma_newton(x, y, z_0):
    """Calcula en primer lugar el polinomio de interpolacion de Lagrange que interpola los datos x e
    y mediante la formula de Newton y lo evalua en z0."""
    n = len(y)
    df = tabla_diferencias_divididas(x, y)
    peval = df[0, 0]
    prod = 1.0
    for i in range(1, n):
        prod = prod * (z_0 - x[i - 1])
        peval = peval + df[i, i] * prod
    return peval


def eval_forma_Horner(x, y, z0):
    n = len(x)
    df = tabla_diferencias_divididas(x, y)
    peval = df[n - 1, n - 1]
    for i in range(n - 2, -1, -1):
        peval = peval * (z0 - x[i]) + df[i, i]
    return peval


print("Caso I")

# Array de 0 hasta 1 "en trozos" de 1/5 de longitud
x = linspace(0, 1, 5)
y = exp(x)
tabla_dif = tabla_diferencias_divididas(x, y)
print(tabla_dif)


print("Caso II")

# Seccion a
eval1 = eval_forma_newton(x, y, 1 / 3)
print("El polinomio de interpolacion evaluado en 1/3 vale", eval1)
print("Pasa por los puntos de interpolacion?", eval_forma_newton(x, y, x) == y)

# Seccion c
eval2 = eval_forma_Horner(x, y, 1 / 3)
print("El polinomio de interpolacion evaluado en 1/3 vale", eval2)


# Seccion d
def evalpol_eqd(f, a, b, N, z0):
    x = linspace(a, b, N + 1)
    y = f(x)
    pz0 = eval_forma_Horner(x, y, z0)
    error = max(abs(f(z0) - pz0))
    return pz0, error


# Seccion e
a, b = -3, 3
n = (b - a) / 0.01
z0 = linspace(a, b, int(n + 1))

for N in array([5, 10, 15, 20]):
    x = linspace(a, b, N + 1)
    pz0, error = evalpol_eqd(exp, a, b, N, z0)
    plot(z0, exp(z0), z0, pz0, x, exp(x), "o")
    print("error para", N, "intervalos:", error)


# Seccion g
def evalpol_cherysev(f, a, b, N, z0):
    k = linspace(0, N, N + 1)
    x = cos(((2 * k + 1) * pi) / (2 * N + 1))  # nodos en [-1,1]
    x = a + (b - a) / 2 * (x + 1)  # Nodos en [a,b]
    y = f(x)
    pz0 = eval_forma_Horner(x, y, z0)
    error = max(abs(f(z0) - pz0))
    return pz0, error


print("Caso 3")


def lineal_trozos(f, a, b, N):
    x = linspace(a, b, N + 1)
    y = f(x)
    pol = interp1d(x, y, kind="linear")
    return pol


a = -3
b = 3
n = (b - a) / 0.01
z0 = linspace(a, b, int(n + 1))
for N in array([5, 10, 15, 20]):
    figure()
    pol = lineal_trozos(exp, a, b, N)  # polinomio de interpolacion lineal a trozos
    pz0 = pol(z0)
    x = linspace(a, b, N + 1)  # nodos de interpolacion
    plot(z0, exp(z0), z0, pz0, x, exp(x), "o")
