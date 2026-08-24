"""
EDOs - metodos multipaso Adams
AB2/AB3/AB4, AM3, predictor-corrector ABM3, arranque con RK4,
analisis de estabilidad y frontera.
Ver docs/enunciados_resumidos.md#04_edos_pvi
"""

from pylab import *
from time import perf_counter


def fun(t, y):
    return -y + exp(-t) * cos(t)


def exacta(t):
    return exp(-t) * sin(t)


def AB2(a, b, fun, N, y0):
    y = zeros(N + 1)
    t = zeros(N + 1)
    f = zeros(N + 1)

    h = (b - a) / float(N)
    t[0] = a
    y[0] = y0
    f[0] = fun(a, y[0])

    # Aplicamos metodos unipaso para poder empezar el metodo multipaso (Euler en este caso)
    y[1] = y[0] + h * f[0]
    t[1] = a + h
    f[1] = fun(t[1], y[1])

    for k in range(1, N):
        y[k + 1] = y[k] + (1 / 2) * h * (3.0 * f[k] - f[k - 1])
        t[k + 1] = t[k] + h
        f[k + 1] = fun(t[k + 1], y[k + 1])

    return (t, y)


# Prametros para ejecucion
y0 = 0.0
a = 0.0
b = 5.0
N = 30

tini = perf_counter()
(t, y) = AB2(a, b, fun, N, y0)
tfin = perf_counter()

ye = exacta(t)

# clf() # clear figure
plot(t, y, "*")
plot(t, ye)

h = (b - a) / float(N)
error = max(abs(y - ye))
tcpu = tfin - tini

print("---------------")
print("h = " + str(h))
print("Error= " + str(error))
print("Tiempo CPU= " + str(tcpu))
print("---------------")

print("Seccion a)")

malla = [10, 20, 40, 80, 160]
for N in malla:
    tini = perf_counter()
    (t, y) = AB2(a, b, fun, N, y0)
    tfin = perf_counter()

    ye = exacta(t)

    # Dibujamos las soluciones
    figure("figura 1a")
    plot(t, y, "-*")  # dibuja la solucion aproximada

    # Calculo del error cometido
    error = max(abs(y - ye))

    # Resultados
    print("-----")
    print("Tiempo CPU: ", (tfin - tini))
    print("Error: ", error)
    if N != malla[0]:
        print("Order:", (log(error_old) - log(error)) / log(2))
    print("Paso de malla: ", (b - a) / N)
    print("-----")

    error_old = error
show()  # muestra la grafica


print("Seccion b)")


# AB USANDO METODO DEL PUNTO MEDIO COMO ARRANQUE
def AB3(a, b, fun, N, y0):
    y = zeros(N + 1)
    t = zeros(N + 1)
    f = zeros(N + 1)

    t[0] = a
    h = (b - a) / float(N)
    y[0] = y0
    f[0] = fun(a, y[0])

    # Metodo unipaso para arranque
    #   Usamos el metodo del punto medio
    for k in range(2):
        z = y[k] + 0.5 * h * f[k]
        y[k + 1] = y[k] + h * fun(t[k] + 0.5 * h, z)
        t[k + 1] = t[k] + h
        f[k + 1] = fun(t[k + 1], y[k + 1])

    for k in range(2, N):
        y[k + 1] = y[k] + h / 12.0 * (23.0 * f[k] - 16 * f[k - 1] + 5 * f[k - 2])
        t[k + 1] = t[k] + h
        f[k + 1] = fun(t[k + 1], y[k + 1])

    return (t, y)


# Esta vez, como es un metodo de 3 pasos, aplico un metiodo unipaso de al menos orden 2
y0 = 0.0
a = 0.0
b = 5.0
N = 10

tini = perf_counter()
(t, y) = AB3(a, b, fun, N, y0)
tfin = perf_counter()
ye = exacta(t)

# clf()
plot(t, y, "*")
plot(t, ye)

h = (b - a) / float(N)
error = max(abs(y - ye))
tcpu = tfin - tini

print("---------------")
print("h = " + str(h))
print("Error= " + str(error))
print("Tiempo CPU= " + str(tcpu))
print("---------------")


malla = [10, 20, 40, 80, 160]
for N in malla:
    tini = perf_counter()
    (t, y) = AB3(a, b, fun, N, y0)  # llamada al metodo de Euler
    tfin = perf_counter()

    ye = exacta(t)  # calculo de la solucion exacta

    # Dibujamos las soluciones
    figure("figura 1b")
    plot(t, y, "-*")  # dibuja la solucion aproximada

    # Calculo del error
    error = max(abs(y - ye))

    # Resultados
    print("-----")
    print("Tiempo CPU: ", (tfin - tini))
    print("Error: ", error)
    if N != malla[0]:
        print("Order:", (log(error_old) - log(error)) / log(2))
    print("Paso de malla: ", (b - a) / N)
    print("-----")

    error_old = error
show()  # muestra la grafica

########################################################################
########################################################################
########################################################################
########################################################################
########################################################################

print("caso 2")
print("seccion a)")


#####! NO ES CASO GENERAL
#### SOLAMENTE VALE PARA EL PRIMER SECCION DEL EJ 2 DE LA SECCION
####    YA QUE SE DESPEJA YK POR LA LINEALIDAD DE f
def AM3Particular(a, b, fun, N, y0):
    y = zeros(N + 1)
    t = zeros(N + 1)
    f = zeros(N + 1)

    t[0] = a
    h = (b - a) / float(N)
    y[0] = y0
    f[0] = fun(a, y[0])

    # Runge kutta - 4 pasos para rranque
    for k in range(2):
        t[k + 1] = t[k] + h
        k1 = f[k]
        k2 = fun(t[k] + h / 2, y[k] + h / 2 * k1)
        k3 = fun(t[k] + h / 2, y[k] + h / 2 * k2)
        k4 = fun(t[k + 1], y[k] + h * k3)

        y[k + 1] = y[k] + h / 6 * (k1 + 2 * k2 + 2 * k3 + k4)
        f[k + 1] = fun(t[k + 1], y[k + 1])

    for k in range(2, N):
        Ck = y[k] + (h / 24.0) * (19 * f[k] - 5 * f[k - 1] + f[k - 2])
        t[k + 1] = t[k] + h
        y[k + 1] = ((9 * h / 24.0) * exp(-t[k + 1]) * cos(t[k + 1]) + Ck) / (
            1 + 9 * h / 24
        )
        f[k + 1] = fun(t[k + 1], y[k + 1])

    return (t, y)


y0 = 0.0
a = 0.0
b = 5.0
N = 10

tini = perf_counter()
(t, y) = AM3Particular(a, b, fun, N, y0)
tfin = perf_counter()
ye = exacta(t)

# clf()
plot(t, y, "*")
plot(t, ye)

h = (b - a) / float(N)
error = max(abs(y - ye))
tcpu = tfin - tini

print("---------------")
print("h = " + str(h))
print("Error= " + str(error))
print("Tiempo CPU= " + str(tcpu))
print("---------------")

malla = [10, 20, 40, 80, 160]
for N in malla:
    tini = perf_counter()
    (t, y) = AM3Particular(a, b, fun, N, y0)  # llamada al metodo de Euler
    tfin = perf_counter()

    ye = exacta(t)  # calculo de la solucion exacta

    figure("figura 2a")

    # Dibujamos las soluciones
    plot(t, y, "-*")  # dibuja la solucion aproximada

    # Calculo del error cometido
    error = max(abs(y - ye))

    # Resultados
    print("-----")
    print("Tiempo CPU: ", (tfin - tini))
    print("Error: ", error)
    if N != malla[0]:
        print("Order:", (log(error_old) - log(error)) / log(2))
    print("Paso de malla: ", (b - a) / N)
    print("-----")

    error_old = error
show()  # muestra la grafica


print("seccion 2.b)")


def AM3(a, b, fun, N, y0):
    tol = 1.0e-12
    Nmax = 200

    y = zeros(N + 1)
    t = zeros(N + 1)
    f = zeros(N + 1)

    t[0] = a
    h = (b - a) / float(N)
    y[0] = y0
    f[0] = fun(a, y[0])

    # Metodo Runge Kutta 4 pasos para obtener datos inciales
    for k in range(2):
        t[k + 1] = t[k] + h
        k1 = f[k]
        k2 = fun(t[k] + h / 2, y[k] + h / 2 * k1)
        k3 = fun(t[k] + h / 2, y[k] + h / 2 * k2)
        k4 = fun(t[k + 1], y[k] + h * k3)
        y[k + 1] = y[k] + h / 6 * (k1 + 2 * k2 + 2 * k3 + k4)
        f[k + 1] = fun(t[k + 1], y[k + 1])

    # Adam-Bashforth usando metodo algoritmo de punto fijo
    for k in range(2, N):
        z = y[k]
        i = 0
        error = (
            1 + tol
        )  # Valor inicial arbitrario para asegurarnos que comenzamos bucle
        t[k + 1] = t[k] + h

        Ck = y[k] + h / 24.0 * (19 * f[k] - 5 * f[k - 1] + f[k - 2])

        # Punto fijo
        while error >= tol and i < Nmax:
            znew = h * 9 / 24 * fun(t[k + 1], z) + Ck
            error = abs(z - znew)
            z = znew
            i += 1

        if i == Nmax:
            print("No converge")

        maxiter = max(maxiter, i)
        y[k + 1] = z
        f[k + 1] = fun(t[k + 1], y[k + 1])
    return (t, y, maxiter)


print("Seccion 2.c)")

# Param:
# a,b       -> Extremos del intervalo
# fun       -> Funcion a considerar
# dyfun     -> Derivada de la funcion de la variable y
# N         -> Numero de elementos a particionar
# t0        -> Valor en el instante inicial

# El metodo de newton resuelve los problemas lineales en una iteraccion,
#   la segunda iteraccion es resultado de que el metodo se percata de que es
#   la solucion


# Se obtiene 2 porque esta aproximando un problema lineal, por ende el metodo de newton
#   que aproxima linearmente no tiene nada que hacer
def AM3Newton(a, b, fun, dyfun, N, y0):
    tol = 1.0e-12
    Nmax = 200
    maxiter = 0

    y = zeros(N + 1)
    t = zeros(N + 1)
    f = zeros(N + 1)

    t[0] = a
    h = (b - a) / N
    y[0] = y0
    f[0] = fun(a, y[0])

    # Metodo unipaso para obtener datos iniciales
    for k in range(2):
        t[k + 1] = t[k] + h

        k1 = f[k]
        k2 = fun(t[k] + h / 2, y[k] + h / 2 * k1)
        k3 = fun(t[k] + h / 2, y[k] + h / 2 * k2)
        k4 = fun(t[k + 1], y[k] + h * k3)

        y[k + 1] = y[k] + h / 6 * (k1 + 2 * k2 + 2 * k3 + k4)
        f[k + 1] = fun(t[k + 1], y[k + 1])

    # Metodo multipaso con test de parada
    for k in range(2, N):
        Ck = y[k] + h / 24.0 * (19 * f[k] - 5 * f[k - 1] + f[k - 2])
        t[k + 1] = t[k] + h
        z = y[k]
        dist = 1 + tol
        count = 0

        # Metodo de newton como punto fijo
        while dist > tol and count < Nmax:
            znew = z - (
                (z - h * 9 / 24 * fun(t[k + 1], z) - Ck)
                / (1 - h * 9 / 24 * (dyfun(t[k + 1], z)))
            )
            dist = abs(z - znew)
            z = znew
            count = count + 1
        maxiter = max(maxiter, count)

        if count == Nmax:
            print("No converge")
        maxiter = max(Nmax, count)

        y[k + 1] = znew
        f[k + 1] = fun(t[k + 1], y[k + 1])
    return (t, y, maxiter)


def fun(t, y):
    return -y + exp(-t) * cos(t)


def dyfun(t, y):
    return -1


def exacta(t):
    return exp(-t) * sin(t)


y0 = 0.0
a = 0.0
b = 5.0
N = 10

malla = [10, 20, 40, 80, 160]
for N in malla:
    tini = perf_counter()
    (t, y, maxiter) = AM3Newton(a, b, fun, dyfun, N, y0)  # llamada al metodo de Euler
    tfin = perf_counter()

    ye = exacta(t)  # calculo de la solucion exacta

    # Dibujamos las soluciones
    figure("figura 2c")
    plot(t, y, "-*")  # dibuja la solucion aproximada

    # Calculo del error cometido
    error = max(abs(y - ye))

    # Resultados
    print("-----")
    print("N=", N)
    print("Tiempo CPU: ", (tfin - tini))
    print("Error: ", error)
    if N != malla[0]:
        print("Order:", (log(error_old) - log(error)) / log(2))
    print("Paso de malla: ", (b - a) / N)
    print("Maximo numero de iteracciones del metodo del punto fijo es:", maxiter)
    print("-----")

    error_old = error


print("Seccion 2.d)")

# Para problemas no lineales no sirve, con los anteriores datos no me convergia
# Con estos nuevos si.

a = 0
b = 1
y0 = 0


def fun(t, y):
    return 1 + y**2


def dyfun(t, y):
    return 2 * y


def exacta(t):
    return tan(t)


malla = [10, 20, 40, 80, 160]
for N in malla:
    tini = perf_counter()
    (t, y, maxiter) = AM3Newton(a, b, fun, dyfun, N, y0)  # llamada al metodo de Euler
    tfin = perf_counter()

    ye = exacta(t)  # calculo de la solucion exacta

    # Dibujamos las soluciones
    figure("figura 2d")
    plot(t, y, "-*")  # dibuja la solucion aproximada

    # Calculo del error cometido
    error = max(abs(y - ye))

    # Resultados
    print("-----")
    print("N=", N)
    print("Tiempo CPU: ", (tfin - tini))
    print("Error: ", error)
    if N != malla[0]:
        print("Order:", (log(error_old) - log(error)) / log(2))
    print("Paso de malla: ", (b - a) / N)
    print("Maximo numero de iteracciones del metodo del punto fijo es:", maxiter)
    print("-----")

    error_old = error
show()

print("Seccion 2.e)")


def AM3NewModificada(a, b, fun, dyfun, N, y0):
    tol = 1.0e-12
    Nmax = 200
    maxiter = 0

    y = zeros(N + 1)
    t = zeros(N + 1)
    f = zeros(N + 1)

    t[0] = a
    h = (b - a) / N
    y[0] = y0
    f[0] = fun(a, y[0])

    for k in range(2):
        t[k + 1] = t[k] + h
        k1 = f[k]
        k2 = fun(t[k] + h / 2, y[k] + h / 2 * k1)
        k3 = fun(t[k] + h / 2, y[k] + h / 2 * k2)
        k4 = fun(t[k + 1], y[k] + h * k3)

        y[k + 1] = y[k] + h / 6 * (k1 + 2 * k2 + 2 * k3 + k4)
        f[k + 1] = fun(t[k + 1], y[k + 1])

    for k in range(2, N):
        Ck = y[k] + h / 24.0 * (19 * f[k] - 5 * f[k - 1] + f[k - 2])
        t[k + 1] = t[k] + h
        z = y[k] + h / 12 * (23 * f[k] - 16 * f[k - 1] + 5 * f[k - 2])
        dist = 1 + tol
        count = 0

        while dist > tol and count < Nmax:
            znew = z - (
                (z - h * 9 / 24 * fun(t[k + 1], z) - Ck)
                / (1 - h * 9 / 24 * (dyfun(t[k + 1], z)))
            )
            dist = abs(z - znew)
            z = znew
            count = count + 1

        maxiter = max(maxiter, count)

        if count == Nmax:
            print("No converge")
        # maxiter = max(Nmax, count)

        y[k + 1] = znew
        f[k + 1] = fun(t[k + 1], y[k + 1])
    return (t, y, maxiter)


def AM3pfModificada(a, b, fun, N, y0):
    tol = 1.0e-12
    Nmax = 200
    maxiter = 0

    y = zeros(N + 1)
    t = zeros(N + 1)
    f = zeros(N + 1)

    t[0] = a
    h = (b - a) / N
    y[0] = y0
    f[0] = fun(a, y[0])

    # RK4 para obtener semillas iniciales
    for k in range(2):
        t[k + 1] = t[k] + h
        k1 = f[k]
        k2 = fun(t[k] + h / 2, y[k] + h / 2 * k1)
        k3 = fun(t[k] + h / 2, y[k] + h / 2 * k2)
        k4 = fun(t[k + 1], y[k] + h * k3)

        y[k + 1] = y[k] + h / 6 * (k1 + 2 * k2 + 2 * k3 + k4)
        f[k + 1] = fun(t[k + 1], y[k + 1])

    # Metodo multipaso por punto fijo con AB3 como semilla
    for k in range(2, N):
        Ck = y[k] + h / 24.0 * (19 * f[k] - 5 * f[k - 1] + f[k - 2])
        t[k + 1] = t[k] + h

        # Usamos AB3 como semilla del metodo del punto fijo
        z = y[k] + h / 12 * (23 * f[k] - 16 * f[k - 1] + 5 * f[k - 2])
        dist = 1 + tol
        count = 0

        while dist > tol and count < Nmax:
            znew = h * 9 / 24 * fun(t[k + 1], z) + Ck
            dist = abs(z - znew)
            z = znew
            count = count + 1
        maxiter = max(maxiter, count)

        if count == Nmax:
            print("No converge")
        maxiter = max(Nmax, count)

        y[k + 1] = znew
        f[k + 1] = fun(t[k + 1], y[k + 1])
    return (t, y, maxiter)


for N in malla:
    tini = perf_counter()
    (t, y, maxiter) = AM3pfModificada(a, b, fun, N, y0)  # llamada al metodo de Euler
    tfin = perf_counter()

    ye = exacta(t)  # calculo de la solucion exacta
    figure("figura 2e.1")
    # Dibujamos las soluciones
    plot(t, y, "-*")  # dibuja la solucion aproximada

    # Calculo del error cometido
    error = max(abs(y - ye))

    # Resultados
    print("-----")
    print("N=", N)
    print("Tiempo CPU: ", (tfin - tini))
    print("Error: ", error)
    if N != malla[0]:
        print("Order:", (log(error_old) - log(error)) / log(2))
    print("Paso de malla: ", (b - a) / N)
    print("Maximo numero de iteracciones del metodo del punto fijo es:", maxiter)
    print("-----")

    error_old = error
show()


for N in malla:
    tini = perf_counter()
    (t, y, maxiter) = AM3NewModificada(
        a, b, fun, dyfun, N, y0
    )  # llamada al metodo de Euler
    tfin = perf_counter()

    ye = exacta(t)  # calculo de la solucion exacta

    figure("figura 2e.2")

    # Dibujamos las soluciones
    plot(t, y, "-*")  # dibuja la solucion aproximada
    plot(t, ye)

    # Calculo del error cometido
    error = max(abs(y - ye))

    # Resultados
    print("-----")
    print("N=", N)
    print("Tiempo CPU: ", (tfin - tini))
    print("Error: ", error)
    if N != malla[0]:
        print("Order:", (log(error_old) - log(error)) / log(2))
    print("Paso de malla: ", (b - a) / N)
    print("Maximo numero de iteracciones del metodo del punto fijo es:", maxiter)
    print("-----")

    error_old = error
show()


#################################################################
#################################################################
#################################################################
#################################################################
#################################################################
#################################################################
## TERCERA SECCION

print("Caso 3")


def fun(t, y):
    return -y + exp(-t) * cos(t)


def exacta(t):
    return exp(-t) * sin(t)


# Orden 4, como debia ser
# Modelo predictor-corrector
def ABM3(a, b, fun, N, y0):
    y = zeros(N + 1)
    t = zeros(N + 1)
    f = zeros(N + 1)

    t[0] = a
    h = (b - a) / N
    y[0] = y0
    f[0] = fun(a, y[0])

    for k in range(2):
        t[k + 1] = t[k] + h
        k1 = f[k]
        k2 = fun(t[k] + h / 2, y[k] + h / 2 * k1)
        k3 = fun(t[k] + h / 2, y[k] + h / 2 * k2)
        k4 = fun(t[k + 1], y[k] + h * k3)

        y[k + 1] = y[k] + h / 6 * (k1 + 2 * k2 + 2 * k3 + k4)
        f[k + 1] = fun(t[k + 1], y[k + 1])

    # Metodo multipaso
    for k in range(2, N):
        t[k + 1] = t[k] + h

        ykestrella = y[k] + h / 12 * (23 * f[k] - 16 * f[k - 1] + 5 * f[k - 2])
        fkestrella = fun(t[k + 1], ykestrella)
        y[k + 1] = y[k] + h / 24 * (
            9 * fkestrella + 19 * f[k] - 5 * f[k - 1] + f[k - 2]
        )
        f[k + 1] = fun(t[k + 1], y[k + 1])

    return (t, y)


y0 = 0.0
a = 0.0
b = 5.0
N = 30

malla = [10, 20, 40, 80, 160]

for N in malla:
    tini = perf_counter()
    (t, y) = ABM3(a, b, fun, N, y0)  # llamada al metodo de Euler
    tfin = perf_counter()

    ye = exacta(t)  # calculo de la solucion exacta

    figure("caso 3")

    # Dibujamos las soluciones
    plot(t, y, "-*")  # dibuja la solucion aproximada

    # Calculo del error cometido
    error = max(abs(y - ye))

    # Resultados
    print("-----")
    print("Tiempo CPU: ", (tfin - tini))
    print("Error: ", error)
    if N != malla[0]:
        print("Order:", (log(error_old) - log(error)) / log(2))
    print("Paso de malla: ", (b - a) / N)
    print("-----")

    error_old = error

show()

#####################################################
print("caso 4")


def AM3sistemas(a, b, fun, N, y0):
    tol = 1.0e-12
    Nmax = 200
    maxiter = 0

    y = zeros([len(y0), N + 1])
    t = zeros(N + 1)
    f = zeros([len(y0), N + 1])

    t[0] = a
    h = (b - a) / N
    y[:, 0] = y0
    f[:, 0] = fun(a, y[:, 0])

    # Obtener datos iniciales por metodo uniapaso
    for k in range(2):
        t[k + 1] = t[k] + h
        k1 = f[:, k]
        k2 = fun(t[k] + h / 2, y[:, k] + h / 2 * k1)
        k3 = fun(t[k] + h / 2, y[:, k] + h / 2 * k2)
        k4 = fun(t[k + 1], y[:, k] + h * k3)
        y[:, k + 1] = y[:, k] + h / 6 * (k1 + 2 * k2 + 2 * k3 + k4)
        f[:, k + 1] = fun(t[k + 1], y[:, k + 1])

    # Metodo multipaso
    for k in range(2, N):
        Ck = y[:, k] + h / 24.0 * (19 * f[:, k] - 5 * f[:, k - 1] + f[:, k - 2])
        t[k + 1] = t[k] + h
        z = y[:, k] + h / 12 * (23 * f[:, k] - 16 * f[:, k - 1] + 5 * f[:, k - 2])
        dist = 1 + tol
        count = 0

        # Condiccion de parada
        while dist > tol and count < Nmax:
            znew = h * 9 / 24 * fun(t[k + 1], z) + Ck
            dist = max(abs(z - znew))
            z = znew
            count = count + 1
        maxiter = max(maxiter, count)

        if count == Nmax:
            print("No converge")
        # maxiter = max(Nmax, count)

        y[:, k + 1] = znew
        f[:, k + 1] = fun(t[k + 1], y[:, k + 1])
    return (t, y, maxiter)


def fsis(t, y):
    f1 = 0.25 * y[0] - 0.01 * y[0] * y[1]
    f2 = -y[1] + 0.01 * y[0] * y[1]
    return array([f1, f2])


a = 0.0
b = 20.0
N = 100
y0 = array([80, 30])


malla = [10, 20, 40, 80, 160, 320, 640]

for N in malla:
    tini = perf_counter()
    (t, y, maxiter) = AM3sistemas(a, b, fsis, N, y0)  # llamada al metodo de Euler
    tfin = perf_counter()

    ye = exacta(t)  # calculo de la solucion exacta
    figure("figura 4a")

    # Dibujamos las soluciones
    plot(t, y[0, :], "-*")  # dibuja la solucion aproximada
    plot(t, y[1, :], "-*")  # dibuja la solucion aproximada

    # Calculo del error cometido
    error = max(abs(y[0, :] - ye))

    # Resultados
    print("-----")
    print("Tiempo CPU: ", (tfin - tini))
    print("Error: ", error)
    if N != malla[0]:
        print("Order:", (log(error_old) - log(error)) / log(2))
    print("Paso de malla: ", (b - a) / N)
    print("-----")

    error_old = error

xlabel("t")
ylabel("y")
leyenda = ["N =" + str(N) for N in malla]
legend(leyenda)
grid(False)

show()  # muestra la grafica


# Dibujamos las soluciones
figure("Figura 4 AB3 2")
subplot(121)
plot(t, y[0, :], t, y[1, :])
xlabel("t")
ylabel("x,y")
legend(["Presa", "Depredador"])
subplot(122)
plot(y[0, :], y[1, :])
xlabel("x")
ylabel("y")
legend(["Trayectoria"])
grid(False)

show()  # muestra la grafica

a = 0.0
b = 20.0
malla = [20, 40, 80, 160, 320, 640]
Y0 = array([80, 30])
k = len(malla)

for i in range(k):
    N = malla[i]
    tini = perf_counter()
    (t, Y, maxiter) = AM3sistemas(a, b, fsis, N, Y0)
    tfin = perf_counter()
    print("\nN = " + str(N))
    print("Paso de malla: " + str((b - a) / N))
    print("Tiempo CPU: " + str(tfin - tini))
    plot(Y[0, :], Y[1, :])

xlabel("x")
ylabel("y")
legend(["N = " + str(N) for N in malla])
grid(True)
