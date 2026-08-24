"""
EDOs - sistemas y casos aplicados
Lotka-Volterra, oscilador rigido, modelo SIR y cohete con empuje
y masa variable resueltos con RK4 y metodos unipaso.
Ver docs/enunciados_resumidos.md#04_edos_pvi
"""

from pylab import (
    exp,
    zeros,
    plot,
    xlabel,
    ylabel,
    legend,
    grid,
    show,
    title,
    array,
    cos,
    sin,
    pi,
    column_stack,
    append,
    subplot,
)
from time import perf_counter


# Para ver las graficas en ventanas a parte: Tools > IPython consola > Graphics > Backend: Automatic
# Si cambio "title" por "figures", las graficas se muestran en ventanas diferentes; si no, se muestran todas en la misma ventana


# Implementacion del metodo de Euler en el intervalo [a, b] usando N particiones y valor inicial y0
#   f funcion toma dos valores
def euler(a, b, f, N, y0):
    h = (b - a) / N
    t = zeros(N + 1)  # inicializacion del vector de nodos
    y = zeros(N + 1)  # inicializacion del vector de resultados
    t[0] = a  # nodo inicial
    y[0] = y0  # valor inicial

    for k in range(N):
        y[k + 1] = y[k] + h * f(t[k], y[k])
        t[k + 1] = t[k] + h
    return (t, y)


####################################################################################################################################

# CASO 1(a)

####################################################################################################################################


def f(t, y):  # Funcion que define la ecuacion diferencial
    return 0.5 * (t**2 - y)


def exacta(t):  # Solucion exacta del problema de valor inicial
    return t**2 - 4 * t + 8 - 7.0 * exp(-0.5 * t)


a = 0.0
b = 10.0
y0 = 1.0
malla = [160]
# malla = [10, 20, 40, 80, 160]
k = len(malla)
errores = zeros(k)

print("\n 1(a) ~ METODO DE EULER")

for i in range(k):
    N = malla[i]
    tini = perf_counter()
    (t, y) = euler(a, b, f, N, y0)
    tfin = perf_counter()
    ye = exacta(t)
    errores[i] = max(abs(y - ye))

    # Hacemos un plot de la aproximacion
    plot(t, y, "-*")
    plot(t, ye, "k")
    xlabel("t")
    ylabel("y")
    legend(["Euler (N=" + str(N) + ")", "exacta"])
    title("1(a). METODO DE EULER")
    grid(False)
    show()  # Show imprime todas las graficas que estan en cola

    # Resultados
    print("\n N = " + str(N))
    print("Paso de malla: " + str((b - a) / N))
    print("Tiempo CPU: " + str(tfin - tini))
    print("Error: " + str(errores[i]))
    if i > 0:
        print("Cociente de errores: " + str(errores[i - 1] / errores[i]))

####################################################################################################################################

# CASO 1(b)

####################################################################################################################################


def f(t, y):
    return 6 - y / 10


def exacta(t):
    return 60 * (1 - exp(-t / 10))


a = 0.0
b = 20.0
y0 = 0.0
# malla = [10, 20, 40, 80, 160]
malla = [160]
k = len(malla)
errores = zeros(k)

print("\n\n\n1(b). METODO DE EULER")

for i in range(k):
    N = malla[i]
    tini = perf_counter()
    (t, y) = euler(a, b, f, N, y0)
    tfin = perf_counter()
    ye = exacta(t)
    errores[i] = max(abs(y - ye))

    plot(t, y, "-*")
    plot(t, ye, "k")
    xlabel("t")
    ylabel("y")
    legend(["Euler (N=" + str(N) + ")", "exacta"])
    title("1(b). METODO DE EULER")
    grid(True)
    show()

    print("\nN = " + str(N))  # Resultados
    print("Paso de malla: " + str((b - a) / N))
    print("Tiempo CPU: " + str(tfin - tini))
    print("Error: " + str(errores[i]))
    if i > 0:
        print("Cociente de errores: " + str(errores[i - 1] / errores[i]))

####################################################################################################################################

# CASO 1(c)

####################################################################################################################################

# Analogo

####################################################################################################################################

# CASO 2

####################################################################################################################################


# Intervalo [a,b] en N intervalos de misma longitud, centrado en y0
# fi es la derivada (i-1)-esima de la funcion
def taylor2(a, b, f1, f2, N, y0):
    h = (b - a) / N
    t = zeros(N + 1)
    y = zeros(N + 1)

    t[0] = a
    y[0] = y0

    for k in range(N):
        y[k + 1] = y[k] + h * f1(t[k], y[k]) + h**2 * f2(t[k], y[k]) / 2
        t[k + 1] = t[k] + h
    return (t, y)


# Ver param taylor2, son analogos
def taylor3(a, b, f1, f2, f3, N, y0):
    h = (b - a) / N
    t = zeros(N + 1)
    y = zeros(N + 1)

    t[0] = a
    y[0] = y0

    for k in range(N):
        y[k + 1] = (
            y[k]
            + h * f1(t[k], y[k])
            + h**2 * (f2(t[k], y[k])) / 2
            + h**3 * f3(t[k], y[k]) / 6
        )
        t[k + 1] = t[k] + h
    return (t, y)


def f(t, y):
    return 0.5 * (t**2 - y)


def derf(t, y):
    return t - 0.25 * (t**2 - y)


def derderf(t, y):
    return 1 - 1 / 2 * t + 1 / 8 * t**2 - 1 / 8 * y


def exacta(t):
    return t**2 - 4 * t + 8 - 7.0 * exp(-0.5 * t)


a = 0.0
b = 10.0
y0 = 1.0
malla = [160]

# malla = [40,80,160]
# malla = [10, 20, 40, 80, 160]
k = len(malla)
errores = zeros(k)

print("\n\n\n2. METODO DE TAYLOR DE ORDEN 2")

for i in range(k):
    N = malla[i]
    tini = perf_counter()
    (t, y) = taylor2(a, b, f, derf, N, y0)
    tfin = perf_counter()
    ye = exacta(t)
    errores[i] = max(abs(y - ye))

    plot(t, y, "-*")
    plot(t, ye, "k")
    xlabel("t")
    ylabel("y")
    legend(["Taylor (N=" + str(N) + ")", "exacta"])
    title("2. METODO DE TAYLOR DE ORDEN 2")
    grid(True)
    show()

    print("\nN = " + str(N))  # Resultados
    print("Paso de malla: " + str((b - a) / N))
    print("Tiempo CPU: " + str(tfin - tini))
    print("Error: " + str(errores[i]))

    #! PORQUE EL ORDEN DEL METODO VIENE DADO EN EL COCIENTE ESE? Orden 2 es cociente 4 aprox. Orden i pues 2^i
    if i > 0:
        print("Cociente de errores: " + str(errores[i - 1] / errores[i]))

print("\n\n\n2. METODO DE TAYLOR DE ORDEN 3")

for i in range(k):
    N = malla[i]
    tini = perf_counter()
    (t, y) = taylor3(a, b, f, derf, derderf, N, y0)
    tfin = perf_counter()
    ye = exacta(t)
    errores[i] = max(abs(y - ye))

    plot(t, y, "-*")
    plot(t, ye, "k")
    xlabel("t")
    ylabel("y")
    legend(["Taylor (N=" + str(N) + ")", "exacta"])
    title("2. METODO DE TAYLOR DE ORDEN 3")
    grid(True)
    show()

    print("\nN = " + str(N))  # Resultados
    print("Paso de malla: " + str((b - a) / N))
    print("Tiempo CPU: " + str(tfin - tini))
    print("Error: " + str(errores[i]))
    if i > 0:
        print("Cociente de errores: " + str(errores[i - 1] / errores[i]))

####################################################################################################################################

# CASO 3

####################################################################################################################################


# Mismos param que EULER
def heun(a, b, f, N, y0):
    h = (b - a) / N
    t = zeros(N + 1)
    y = zeros(N + 1)

    t[0] = a
    y[0] = y0

    for k in range(N):
        t[k + 1] = t[k] + h
        ff = f(t[k], y[k])  # para evaluar la funcion menos veces
        yy = y[k] + h * ff
        y[k + 1] = y[k] + 0.5 * h * (ff + f(t[k + 1], yy))
    return (t, y)


# Misma parametrizacion que EULER
def RK4(a, b, f, N, y0):
    h = (b - a) / N
    t = zeros(N + 1)
    y = zeros(N + 1)

    t[0] = a
    y[0] = y0

    for k in range(N):
        t[k + 1] = t[k] + h
        k1 = f(t[k], y[k])
        k2 = f(t[k] + h / 2, y[k] + h / 2 * k1)
        k3 = f(t[k] + h / 2, y[k] + h / 2 * k2)
        k4 = f(t[k + 1], y[k] + h * k3)
        y[k + 1] = y[k] + h / 6 * (k1 + 2 * k2 + 2 * k3 + k4)

    return (t, y)


def f(t, y):
    return 0.5 * (t**2 - y)


def exacta(t):
    return t**2 - 4 * t + 8 - 7.0 * exp(-0.5 * t)


a = 0.0
b = 10.0
y0 = 1.0
# malla = [10, 20, 40, 80, 160]
malla = [160]
k = len(malla)
errores = zeros(k)

print("\n\n\n3. METODO DE HEUN")

for i in range(k):
    N = malla[i]
    (t, y) = heun(a, b, f, N, y0)
    ye = exacta(t)
    errores[i] = max(abs(y - ye))

    plot(t, y, "-*")
    plot(t, ye, "k")
    xlabel("t")
    ylabel("y")
    legend(["Heun (N=" + str(N) + ")", "exacta"])
    title("3. METODO DE HEUN")
    grid(True)
    show()

    print("\nN = " + str(N))  # Resultados
    print("Error: " + str(errores[i]))
    if i > 0:
        print(
            "Cociente de errores: " + str(errores[i - 1] / errores[i])
        )  # Se comprueba que el metodo de Heun es de orden 2

print("\n\n\n3. METODO RK4")

for i in range(k):
    N = malla[i]
    (t, y) = RK4(a, b, f, N, y0)
    ye = exacta(t)
    errores[i] = max(abs(y - ye))

    plot(t, y, "-*")
    plot(t, ye, "k")
    xlabel("t")
    ylabel("y")
    legend(["RK4 (N=" + str(N) + ")", "exacta"])
    title("3. METODO RK4")
    grid(True)
    show()

    print("\nN = " + str(N))  # Resultados
    print("Error: " + str(errores[i]))
    if i > 0:
        print(
            "Cociente de errores: " + str(errores[i - 1] / errores[i])
        )  # Se comprueba que el metodo RK4 es de orden 4

####################################################################################################################################

# CASO 4

####################################################################################################################################

x = [1, 2, 3]  # Lista
y = [4, 5, 6]
x + y  # Concatenacion x e y como listas

X = array([1, 2, 3])  # Array
Y = array([4, 5, 6])
X + Y  # Devuelve un array que resulta de sumar x e y


# y es un array de valores en los que queremos realizar la evaluacion en cada componente
def f_vec(t, y):
    f_vec1 = y[0] * cos(t)
    f_vec2 = y[1] * sin(t)
    return array([f_vec1, f_vec2])


f_vec(pi, array([1, 1]))  # Devuelve un vector bidimensional

A = array([[1, 2, 3], [4, 5, 6]])
# A[col, fila]
A[0, 0]  # Devuelve el elemento (1, 1)
A[:, 1]  # Devuelve la columna correspondiente entera
A[1, :]  # Devuelve la  fila correspondiente entera


# Valido tambien para vectores unidimensionales
def euler(a, b, F, N, Y0):
    h = (b - a) / N
    t = zeros(N + 1)
    Y = zeros((len(Y0), N + 1))

    t[0] = a
    Y[:, 0] = Y0

    for k in range(N):
        Y[:, k + 1] = Y[:, k] + h * F(t[k], Y[:, k])
        t[k + 1] = t[k] + h
    return (t, Y)


def F(t, Y):
    F1 = 0.25 * Y[0] - 0.01 * Y[0] * Y[1]
    F2 = -Y[1] + 0.01 * Y[0] * Y[1]
    return array([F1, F2])


a = 0.0
b = 20.0
# malla = [20, 40, 80, 160, 320, 640]
malla = [320, 640]
Y0 = array([80, 30])
k = len(malla)

print("\n\n\n 4 ~ METODO DE EULER")

for i in range(k):
    N = malla[i]
    tini = perf_counter()
    (t, Y) = euler(a, b, F, N, Y0)
    tfin = perf_counter()

    print("\nN = " + str(N))  # Resultados
    print("Paso de malla: " + str((b - a) / N))
    print("Tiempo CPU: " + str(tfin - tini))

    plot(Y[0, :], Y[1, :])


xlabel("x")
ylabel("y")
legend(["N = " + str(N) for N in malla])
title("4. METODO DE EULER")
show()

# REPRESENTAR DOS GRAFICAS EN UNA

# subplot(121)
# plot(t, Y[0,:], t, Y[1,:])
# xlabel('t')
# ylabel('x, y')
# legend(['Presa', 'Depredador'])

# subplot(122)
# plot(Y[0,:], Y[1,:])
# xlabel('x')
# ylabel('y')
# legend(['Trayectoria'])

# show()


def RK4(a, b, f, N, y0):  # Hacemos lo mismo pero con el metodo RK4
    h = (b - a) / N
    t = zeros(N + 1)
    y = zeros((len(y0), N + 1))

    t[0] = a
    y[:, 0] = y0

    for k in range(N):
        t[k + 1] = t[k] + h
        k1 = f(t[k], y[:, k])
        k2 = f(t[k] + h / 2, y[:, k] + h / 2 * k1)
        k3 = f(t[k] + h / 2, y[:, k] + h / 2 * k2)
        k4 = f(t[k + 1], y[:, k] + h * k3)
        y[:, k + 1] = y[:, k] + h / 6 * (k1 + 2 * k2 + 2 * k3 + k4)
    return (t, y)


a = 0.0
b = 20.0
# malla = [20, 40, 80, 160, 320, 640]
malla = [320, 640]
Y0 = array([80, 30])
k = len(malla)

print("\n\n\n4 ~ METODO RK4")

for i in range(k):
    N = malla[i]
    tini = perf_counter()
    (t, Y) = RK4(a, b, F, N, Y0)
    tfin = perf_counter()

    print("\nN = " + str(N))
    print("Paso de malla: " + str((b - a) / N))
    print("Tiempo CPU: " + str(tfin - tini))

    plot(Y[0, :], Y[1, :])

xlabel("x")
ylabel("y")
legend(["N = " + str(N) for N in malla])
title("4 ~ METODO RK4")
show()  # Las soluciones son mucho mejores

####################################################################################################################################

# CASO 5

####################################################################################################################################


def F(t, Y):
    F1 = Y[1]
    F2 = -20 * Y[1] - 101 * Y[0]
    return array([F1, F2])


def exacta(t):
    return exp(-10 * t) * cos(t)


a = 0.0
b = 7.0
malla = [20, 40, 80, 160, 320, 640]
Y0 = array([1, -10])
k = len(malla)
errores = zeros(k)

print("\n\n\n5 ~ METODO DE EULER")

for i in range(k):
    N = malla[i]
    tini = perf_counter()
    (t, Y) = euler(a, b, F, N, Y0)
    tfin = perf_counter()
    ye = exacta(t)
    error = max(abs(Y[0, :] - ye))

    print("\nN = " + str(N))
    print("Paso de malla: " + str((b - a) / N))
    print("Tiempo CPU: " + str(tfin - tini))
    print("Error: " + str(error))

    plot(t, Y[0, :])  # Ahora solo se necesita pintar la primera componente de Y

plot(t, ye, "k")
xlabel("t")
ylabel("x")
leyenda = ["N = " + str(N) for N in malla]
leyenda.append("exacta")
legend(leyenda)
title("5. METODO DE EULER")
show()  # Hay anomalias para N = 20

print("\n\n\n5. METODO RK4")

for i in range(k):
    N = malla[i]
    tini = perf_counter()
    (t, Y) = RK4(a, b, F, N, Y0)
    tfin = perf_counter()
    ye = exacta(t)
    error = max(abs(Y[0, :] - ye))

    print("\nN = " + str(N))
    print("Paso de malla: " + str((b - a) / N))
    print("Tiempo CPU: " + str(tfin - tini))
    print("Error: " + str(error))

    plot(t, Y[0, :])  # Ahora solo se necesita pintar la primera componente de Y

plot(t, ye, "k")
xlabel("t")
ylabel("x")
leyenda = ["N = " + str(N) for N in malla]
leyenda.append("exacta")
legend(leyenda)
title("5. METODO RK4")
show()  # Hay anomalias para N = 20 (hay problemas de estabilidad; ya se vera en clase)

####################################################################################################################################

# CASO 7(a)

####################################################################################################################################


# Modificamos el programa para que pare cuando la altura del cohete llegue a cero
def RK4mod(a, f, h, Y0):
    t = zeros(1)
    Y = zeros((len(Y0), 1))

    t[0] = a
    Y[:, 0] = Y0
    k = 0

    while Y[0, k] >= 0:
        k1 = f(t[k], Y[:, k])
        k2 = f(t[k] + h / 2, Y[:, k] + h / 2 * k1)
        k3 = f(t[k] + h / 2, Y[:, k] + h / 2 * k2)
        k4 = f(t[k] + h, Y[:, k] + h * k3)
        Ynew = Y[:, k] + h / 6 * (k1 + 2 * k2 + 2 * k3 + k4)

        t = append(t, t[k] + h)
        Y = column_stack((Y, Ynew))
        k = k + 1
    return (t, Y)


def F(t, Y):  # Y[0] = z, Y[1] = v, Y[2] = m_f
    # Constantes
    g = 9.81
    alpha = 0.02
    M = 7.5

    m = M + Y[2]
    T = T0 * (Y[2] > 0)  # Y[2] > 0 devuelve un 1 y, en caso contrario, devuelve un 0

    # Funciones
    F1 = Y[1]
    F2 = -g + T / m - C * Y[1] * abs(Y[1]) / m + alpha * T * Y[1] / m
    F3 = -alpha * T
    return array([F1, F2, F3])


####################################################################################################################################

# CASO 7(b)

####################################################################################################################################

a = 0.0
h = 0.05
Y0 = array([0, 50, 7.5])

print("\n\n\n7(b)(i). METODO RK4")

T0 = 0.0
C = 0.0

tini = perf_counter()
(ti, Yi) = RK4mod(a, F, h, Y0)
tfin = perf_counter()

print("\nPaso de malla: " + str(h))
print(
    "Tiempo de vuelo: " + str(ti[-1])
)  # La entrada -1 del vector es la ultima componente
print("Altura maxima: " + str(max(Yi[0, :])))
print("Tiempo CPU: " + str(tfin - tini))

print("\n\n\n7(b)(ii). METODO RK4")

T0 = 0.0
C = 0.02

tini = perf_counter()
(tii, Yii) = RK4mod(a, F, h, Y0)
tfin = perf_counter()

print("\nPaso de malla: " + str(h))
print("Tiempo de vuelo: " + str(tii[-1]))
print("Altura maxima: " + str(max(Yii[0, :])))
print("Tiempo CPU: " + str(tfin - tini))

print("\n\n\n7(b)(iii). METODO RK4")

T0 = 50.0
C = 0.02

tini = perf_counter()
(tiii, Yiii) = RK4mod(a, F, h, Y0)
tfin = perf_counter()

print("\nPaso de malla: " + str(h))
print("Tiempo de vuelo: " + str(tiii[-1]))
print("Altura maxima: " + str(max(Yiii[0, :])))
print("Tiempo CPU: " + str(tfin - tini))

####################################################################################################################################

# CASO 7(c)

####################################################################################################################################

title("7(c). METODO RK4")
plot(ti, Yi[0, :], tii, Yii[0, :], tiii, Yiii[0, :])
xlabel("t")
ylabel("z")
legend(["T0 = 0, C = 0", "T0 = 0, C = 0.02", "T0 = 50, C = 0.02"])
show()

title("7(c). METODO RK4")
plot(tiii, Yiii[2])
xlabel("t")
ylabel("mf")
legend(["T0 = 50, C = 0.02"])
show()

k = 0
while Yiii[2, k] > 0:
    k = k + 1

print("Momento en que se acaba el combustible: t = " + str(tiii[k - 1]))

"""

OTRA FORMA 
    
De la ecuacion de m_f en el problema se deduce que m_f = -alpha*T*t + m_{f,0}
Poniendo m_f = 0 y despejando t se halla el tiempo en el que se acaba el combustible

"""
