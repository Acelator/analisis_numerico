"""
EDOs - biblioteca de metodos unipaso
Euler, Taylor 2/3, Heun, punto medio y RK4 (escalar y sistemas).
Incluye ejemplos de verificacion de orden y estabilidad.
Ver docs/enunciados_resumidos.md#04_edos_pvi
"""

from pylab import *
from time import perf_counter


# METODOS unidimensionales
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


def ptm(a, b, f, N, y0):
    h = (b - a) / N
    t = zeros(N + 1)
    y = zeros(N + 1)

    t[0] = a
    y[0] = y0

    for k in range(N):
        t[k + 1] = t[k] + h
        auxY = y[k] + h / 2 * f(t[k], y[k])
        y[k + 1] = y[k] + h * f(t[k] + h / 2, auxY)
    return (t, y)


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


######################################################


def f(t, y):
    return -y + 2 * sin(t)


def exacta(t):
    return (pi + 1) * exp(-t) + sin(t) - cos(t)


############
a = 0.0
b = 10.0
y0 = pi
N = 50

(t, y) = heun(a, b, f, N, y0)
ye = exacta(t)
err = max(abs(y - ye))

figure("Metodo de heun")
plot(t, y, "-*")
plot(t, ye, "k")
xlabel("t")
ylabel("y")
legend(["Heun (N=" + str(N) + ")", "exacta"])
grid(True)

# Resultados
print("\nN = " + str(N))
print("Error heun: " + str(err))

############

(t, y) = ptm(a, b, f, N, y0)
ye = exacta(t)
err = max(abs(y - ye))

figure("Metodo de ptm")
plot(t, y, "-*")
plot(t, ye, "k")
xlabel("t")
ylabel("y")
legend(["PTM (N=" + str(N) + ")", "exacta"])
grid(True)

print("\nN = " + str(N))  # Resultados
print("Error PTM: " + str(err))

#############

(t, y) = euler(a, b, f, N, y0)
ye = exacta(t)
err = max(abs(y - ye))

figure("Heun")
plot(t, y, "-*")
plot(t, ye, "k")
xlabel("t")
ylabel("y")
legend(["Heun (N=" + str(N) + ")", "exacta"])
grid(True)

print("\nN = " + str(N))  # Resultados
print("Error euler: " + str(err))

#############

(t, y) = RK4(a, b, f, N, y0)
ye = exacta(t)
err = max(abs(y - ye))

figure("Heun")
plot(t, y, "-*")
plot(t, ye, "k")
xlabel("t")
ylabel("y")
legend(["Heun (N=" + str(N) + ")", "exacta"])
grid(True)

print("\nN = " + str(N))  # Resultados
print("Error rk4: " + str(err))


######################################################
######################################################
######################################################


# METODOS BIDIMENSIONALES
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


def heun_sistemas(a, b, fun, N, Y0):
    h = (b - a) / N
    t = zeros(N + 1)
    Y = zeros((len(Y0), N + 1))

    t[0] = a
    Y[:, 0] = Y0

    for k in range(N):
        t[k + 1] = t[k] + h
        auxF = fun(t[k], Y[:, k])
        auxY = Y[:, k] + h * auxF
        Y[:, k + 1] = Y[:, k] + 0.5 * h * (auxF + fun(t[k + 1], auxY))

    return (t, Y)


def puntomedio_sistemas(a, b, fun, N, Y0):
    h = (b - a) / N
    t = zeros(N + 1)
    Y = zeros((len(Y0), N + 1))

    t[0] = a
    Y[:, 0] = Y0

    for k in range(N):
        t[k + 1] = t[k] + h
        auxY = Y[:, k] + h / 2 * fun(t[k], Y[:, k])
        Y[:, k + 1] = Y[:, k] + h * fun(t[k] + h / 2, auxY)

    return (t, Y)


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


#######


def F(t, Y):
    F1 = 3 * Y[0] - 2 * Y[1]
    F2 = -Y[0] + 3 * Y[1] - 2 * Y[2]
    F3 = -Y[1] + 3 * Y[2]

    return array([F1, F2, F3])


def exacta(t):
    F1 = -1 / 4 * exp(5 * t) + 3 / 2 * exp(3 * t) - 1 / 4 * exp(t)
    F2 = 1 / 4 * exp(5 * t) - 1 / 4 * exp(t)
    F3 = -1 / 8 * exp(5 * t) - 3 / 4 * exp(3 * t) - 1 / 8 * exp(t)

    return array([F1, F2, F3])


a = 0.0
b = 1.0
N = 50
Y0 = array([1, 0, -1])

print("\n\n\n4 ~ METODO RK4")

tini = perf_counter()
(t, Y) = RK4(a, b, F, N, Y0)
tfin = perf_counter()

ye = exacta(t)
# Calculate errors for each component separately
error1 = max(abs(Y[0, :] - ye[0]))
error2 = max(abs(Y[1, :] - ye[1]))
error3 = max(abs(Y[2, :] - ye[2]))
max_error = max(error1, error2, error3)

print("\nN = " + str(N))
print("Paso de malla: " + str((b - a) / N))
print("Tiempo CPU: " + str(tfin - tini))
print("Error componente 1: " + str(error1))
print("Error componente 2: " + str(error2))
print("Error componente 3: " + str(error3))
print("Error maximo: " + str(max_error))

# Visualize all three components
figure("Caso 1")
subplot(2, 2, 1)
plot(t, Y[0, :], "b-")
plot(t, ye[0], "r--")
xlabel("t")
ylabel("Primera componente")
legend(["Numerica", "Exacta"])

subplot(2, 2, 2)
plot(t, Y[1, :], "g-")
plot(t, ye[1], "r--")
xlabel("t")
ylabel("Segunda componente")
legend(["Numerica", "Exacta"])

subplot(2, 2, 3)
plot(t, Y[2, :], "m-")
plot(t, ye[2], "r--")
xlabel("t")
ylabel("Tercera componente")
legend(["Numerica", "Exacta"])

subplot(2, 2, 4)
plot(Y[0, :], Y[1, :], "k-")
xlabel("Y[0]")
ylabel("Plano de fase Y[0]-Y[1]")

# show()


#########################################################################
from pylab import zeros


def euler(a, b, fun, N, y0):
    h = (b - a) / N
    t = zeros(N + 1)
    y = zeros(N + 1)
    t[0] = a
    y[0] = y0
    for k in range(N):
        y[k + 1] = y[k] + h * fun(t[k], y[k])
        t[k + 1] = t[k] + h
    return (t, y)


def euler_sistemas(a, b, fun, N, Y0):
    h = (b - a) / N
    t = zeros(N + 1)
    Y = zeros((len(Y0), N + 1))
    t[0] = a
    Y[:, 0] = Y0
    for k in range(N):
        Y[:, k + 1] = Y[:, k] + h * fun(t[k], Y[:, k])
        t[k + 1] = t[k] + h
    return (t, Y)


def heun(a, b, fun, N, y0):
    h = (b - a) / N
    t = zeros(N + 1)
    y = zeros(N + 1)
    t[0] = a
    y[0] = y0
    for k in range(N):
        t[k + 1] = t[k] + h
        auxF = fun(t[k], y[k])
        auxY = y[k] + h * auxF
        y[k + 1] = y[k] + 0.5 * h * (auxF + fun(t[k + 1], auxY))
    return (t, y)


def heun_sistemas(a, b, fun, N, Y0):
    h = (b - a) / N
    t = zeros(N + 1)
    Y = zeros((len(Y0), N + 1))
    t[0] = a
    Y[:, 0] = Y0
    for k in range(N):
        t[k + 1] = t[k] + h
        auxF = fun(t[k], Y[:, k])
        auxY = Y[:, k] + h * auxF
        Y[:, k + 1] = Y[:, k] + 0.5 * h * (auxF + fun(t[k + 1], auxY))
    return (t, Y)


def puntomedio(a, b, fun, N, y0):
    h = (b - a) / N
    t = zeros(N + 1)
    y = zeros(N + 1)
    t[0] = a
    y[0] = y0
    for k in range(N):
        t[k + 1] = t[k] + h
        auxY = y[k] + h / 2 * fun(t[k], y[k])
        y[k + 1] = y[k] + h * fun(t[k] + h / 2, auxY)
    return (t, y)


def puntomedio_sistemas(a, b, fun, N, Y0):
    h = (b - a) / N
    t = zeros(N + 1)
    Y = zeros((len(Y0), N + 1))
    t[0] = a
    Y[:, 0] = Y0
    for k in range(N):
        t[k + 1] = t[k] + h
        auxY = Y[:, k] + h / 2 * fun(t[k], Y[:, k])
        Y[:, k + 1] = Y[:, k] + h * fun(t[k] + h / 2, auxY)
    return (t, Y)


def RK4(a, b, fun, N, y0):
    h = (b - a) / N
    t = zeros(N + 1)
    y = zeros(N + 1)
    t[0] = a
    y[0] = y0
    for k in range(N):
        t[k + 1] = t[k] + h
        k1 = fun(t[k], y[k])
        k2 = fun(t[k] + h / 2, y[k] + h / 2 * k1)
        k3 = fun(t[k] + h / 2, y[k] + h / 2 * k2)
        k4 = fun(t[k + 1], y[k] + h * k3)
        y[k + 1] = y[k] + h / 6 * (k1 + 2 * k2 + 2 * k3 + k4)
    return (t, y)


def RK4_sistemas(a, b, fun, N, Y0):
    h = (b - a) / N
    t = zeros(N + 1)
    Y = zeros((len(Y0), N + 1))
    t[0] = a
    Y[:, 0] = Y0
    for k in range(N):
        t[k + 1] = t[k] + h
        k1 = fun(t[k], Y[:, k])
        k2 = fun(t[k] + h / 2, Y[:, k] + h / 2 * k1)
        k3 = fun(t[k] + h / 2, Y[:, k] + h / 2 * k2)
        k4 = fun(t[k + 1], Y[:, k] + h * k3)
        Y[:, k + 1] = Y[:, k] + h / 6 * (k1 + 2 * k2 + 2 * k3 + k4)
    return (t, Y)


####################
def eulerimplicito(a, b, fun, N, y0):
    h = (b - a) / N  # paso de malla
    t = zeros(N + 1)  # inicializacion del vector de nodos
    y = zeros(N + 1)  # inicializacion del vector de resultados

    t[0] = a  # nodo inicial
    y[0] = y0  # valor inicial

    for k in range(N):
        t[k + 1] = t[k] + h
        k1 = fun(t[k], y[k] + h)
        y[k + 1] = y[k] + h * k1

    return (t, y)


################################################
################################################
################################################
################################################
################################################
################################################
################################################
################################################
################################################
################################################
################################################


alfa = 10
beta = 1
A = array([[0, 1], [-alfa, -beta]])

print("\nCASO 1(a)")
print("\nAutovalores de A: " + str(eigvals(A)))


def AB4(a, b, fun, N, Y0):
    Y = zeros((len(Y0), N + 1))
    F = zeros((len(Y0), N + 1))
    t = zeros(N + 1)
    h = (b - a) / float(N)

    t[0] = a
    Y[:, 0] = Y0
    F[:, 0] = fun(a, Y[:, 0])

    for k in range(3):
        t[k + 1] = t[k] + h

        k1 = fun(t[k], Y[:, k])
        k2 = fun(t[k] + h / 2, Y[:, k] + h / 2 * k1)
        k3 = fun(t[k] + h / 2, Y[:, k] + h / 2 * k2)
        k4 = fun(t[k + 1], Y[:, k] + h * k3)

        Y[:, k + 1] = Y[:, k] + h / 6 * (k1 + 2 * k2 + 2 * k3 + k4)
        F[:, k + 1] = fun(t[k + 1], Y[:, k + 1])

    for k in range(3, N):
        t[k + 1] = t[k] + h
        Y[:, k + 1] = Y[:, k] + h / 24 * (
            55 * F[:, k] - 59 * F[:, k - 1] + 37 * F[:, k - 2] - 9 * F[:, k - 3]
        )
        F[:, k + 1] = fun(t[k + 1], Y[:, k + 1])

    return (t, Y)


figure("Metodo AB4 (a = 10, beta = 1); seccion (a)")

(t, Y) = AB4(a, b, F, N, Y0)

subplot(131)
plot(t, Y[0, :])
xlabel("t")
ylabel("x")
legend(["Grafica de x frente a t"])

subplot(132)
plot(t, Y[1, :])
xlabel("t")
ylabel("y")
legend(["Grafica de y frente a t"])

# Plano de fases
subplot(133)
plot(Y[0, :], Y[1, :])
xlabel("x")
ylabel("y")
legend(["Trayectoria"])
show()

figure("Metodo AB4 (frontera de la region de estabilidad absoluta); seccion (a)")


def locfron(rho, sigma):
    theta = arange(0, 2.0 * pi, 0.01)
    numer = polyval(rho, exp(theta * 1j))
    denom = polyval(sigma, exp(theta * 1j))
    mu = numer / denom
    x = real(mu)
    y = imag(mu)
    plot(x, y)
    grid(True)
    axis("equal")


print("autovalores:", eigvals(array([[0, 1], [-10, -1]])))

rho = array([1, -1, 0, 0, 0])
sigma = array([0, 55, -59, 37, -9]) / 24
locfron(rho, sigma)

re = -0.5
im = 3.122499

plot([re, re], [im, -im], "*k")
plot([0, re], [0, im], "--", [0, re], [0, -im], "--")
show()

"""

Como la region de estabilidad absoluta D_A no contiene al eje real positivo en un entorno del origen, la region de estabilidad 
absoluta del metodo AB4 es la region del plano encerrada por la frontera.

(i) Sea r1 la semirrecta que pasa por el autovalor lambda1 = -0.5+3.122499j y por 0. Los puntos de r1 son de la forma -0.5h+3.122499hj, 
    con h > 0. En la grafica se observa que la interseccion de r1 con la frontera de la region de estabilidad absoluta es, 
    aproximadamente, (-0.062675, 0.391405), es decir, h = 0.062675/0.5 = 0.12535. Esto nos dice que si 0 < h < 0.12535, entonces
    h*lambda1 esta en D_A.
    
(ii) Sea r2 la semirrecta que pasa por el autovalor lambda2 = -0.5-3.122499j y por 0. Los puntos de r2 son de la forma -0.5h-+3.122499hj, con 
     h > 0. En la grafica se observa que la interseccion de r2 con la frontera de la region de estabilidad absoluta es, 
     aproximadamente, (-0.062675, -0.391405), es decir, h = 0.062675/0.5 = 0.12535. Esto nos dice que si 0 < h < 0.12535, entonces
     h*lambda2 esta en D_A.
     
Para N = 200 es h = 0.1 < 0.12535, asi que el metodo no deberia presentar comportamientos extranos, que es lo que sucede en la 
grafica de las aproximaciones.
"""


################################################
################################################
################################################
################################################
################################################
################################################
################################################
################################################
################################################
############################# Segunda seccion


def AB2(a, b, fun, N, y0):
    y = zeros(N + 1)
    t = zeros(N + 1)
    f = zeros(N + 1)

    h = (b - a) / float(N)
    t[0] = a
    y[0] = y0
    f[0] = fun(a, y[0])

    # Aplicamos metodos unipaso para poder empezar el metodo multipaso (Euler explicito in this instance)
    y[1] = y[0] + h * f[0]
    t[1] = a + h
    f[1] = fun(t[1], y[1])

    for k in range(1, N):
        y[k + 1] = y[k] + (1 / 2) * h * (3.0 * f[k] - f[k - 1])
        t[k + 1] = t[k] + h
        f[k + 1] = fun(t[k + 1], y[k + 1])

    return (t, y)


# Parametros de entrada estandar
def AB3(a, b, fun, N, y0):
    y = zeros(N + 1)
    t = zeros(N + 1)
    f = zeros(N + 1)

    t[0] = a
    h = (b - a) / float(N)
    y[0] = y0
    f[0] = fun(a, y[0])

    # Usamos RK-4 para semillas iniciales
    for k in range(2):
        t[k + 1] = t[k] + h

        K1 = f[k]
        K2 = fun(t[k] + 0.5 * h, y[k] + 0.5 * h * K1)
        K3 = fun(t[k] + 0.5 * h, y[k] + 0.5 * h * K2)
        K4 = fun(t[k + 1], y[k] + h * K3)

        y[k + 1] = y[k] + h / 6 * (K1 + 2 * K2 + 2 * K3 + K4)
        f[k + 1] = fun(t[k + 1], y[k + 1])

    for k in range(2, N):
        y[k + 1] = y[k] + h / 12 * (23.0 * f[k] - 16 * f[k - 1] + 5 * f[k - 2])
        t[k + 1] = t[k] + h
        f[k + 1] = fun(t[k + 1], y[k + 1])

    return (t, y)


# Parametros de entrada estandar
def AM3(a, b, fun, N, y0):
    tol = 1.0e-12
    Nmax = 200
    maxiter = 0

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


# Predictor corrector
def ABM3(a, b, fun, N, y0):
    y = zeros(N + 1)
    t = zeros(N + 1)
    f = zeros(N + 1)

    t[0] = a
    h = (b - a) / N
    y[0] = y0
    f[0] = fun(a, y[0])

    # Metodo unipaso (RK4) para inicializar
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

        # Prediccion
        ykestrella = y[k] + h / 12 * (23 * f[k] - 16 * f[k - 1] + 5 * f[k - 2])

        # Evaluacion
        fkestrella = fun(t[k + 1], ykestrella)

        # Correcion
        y[k + 1] = y[k] + h / 24 * (
            9 * fkestrella + 19 * f[k] - 5 * f[k - 1] + f[k - 2]
        )

        # Evaluacion
        f[k + 1] = fun(t[k + 1], y[k + 1])

    return (t, y)


##############################################
######## SISTEMAS


def AM3sistemas(a, b, fun, N, y0):
    tol = 1.0e-12
    Nmax = 200
    maxiter = 0

    y = zeros([len(y0), N + 1])
    t = zeros(N + 1)
    f = zeros([len(y0), N + 1])

    t[0] = a
    h = (b - a) / float(N)
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
        dist = 1 + tol
        count = 0

        t[k + 1] = t[k] + h
        Ck = y[:, k] + h / 24 * (19 * f[:, k] - 5 * f[:, k - 1] + f[:, k - 2])
        z = y[:, k] + h / 12 * (23 * f[:, k] - 16 * f[:, k - 1] + 5 * f[:, k - 2])

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


def AB3sis(a, b, fun, N, y0):
    y = zeros([len(y0), N + 1])
    t = zeros(N + 1)
    f = zeros([len(y0), N + 1])

    t[0] = a
    h = (b - a) / float(N)
    y[:, 0] = y0
    f[:, 0] = fun(a, y[:, 0])

    for k in range(2):
        t[k + 1] = t[k] + h

        K1 = f[:, k]
        K2 = fun(t[k] + 0.5 * h, y[:, k] + 0.5 * h * K1)
        K3 = fun(t[k] + 0.5 * h, y[:, k] + 0.5 * h * K2)
        K4 = fun(t[k + 1], y[:, k] + h * K3)

        y[:, k + 1] = y[:, k] + h / 6 * (K1 + 2 * K2 + 2 * K3 + K4)
        f[:, k + 1] = fun(t[k + 1], y[:, k + 1])

    for k in range(2, N):
        y[:, k + 1] = y[:, k] + h / 12 * (
            23.0 * f[:, k] - 16 * f[:, k - 1] + 5 * f[:, k - 2]
        )
        t[k + 1] = t[k] + h
        f[:, k + 1] = fun(t[k + 1], y[:, k + 1])

    return (t, y)


def AM3pfsis(a, b, fun, N, y0):
    tol = 1e-12
    Nmax = 200
    maxiter = 0

    y = zeros([len(y0), N + 1])
    t = zeros(N + 1)
    f = zeros([len(y0), N + 1])

    t[0] = a
    h = (b - a) / float(N)
    y[:, 0] = y0
    f[:, 0] = fun(a, y[:, 0])

    for k in range(2):
        t[k + 1] = t[k] + h
        K1 = f[:, k]
        K2 = fun(t[k] + 0.5 * h, y[:, k] + 0.5 * h * K1)
        K3 = fun(t[k] + 0.5 * h, y[:, k] + 0.5 * h * K2)
        K4 = fun(t[k + 1], y[:, k] + h * K3)
        y[:, k + 1] = y[:, k] + h / 6 * (K1 + 2 * K2 + 2 * K3 + K4)
        f[:, k + 1] = fun(t[k + 1], y[:, k + 1])

    for k in range(2, N):
        cont = 0
        error = 1 + tol

        t[k + 1] = t[k] + h
        z = y[:, k] + h / 12 * (23 * f[:, k] - 16 * f[:, k - 1] + 5 * f[:, k - 2])
        Ck = y[:, k] + h / 24 * (19 * f[:, k] - 5 * f[:, k - 1] + f[:, k - 2])

        while error >= tol and cont < Nmax:  # Metodo del punto fijo
            znew = h * 9 / 24 * fun(t[k + 1], z) + Ck  # Iteracion del punto fijo
            error = max(abs(z - znew))
            z = znew
            cont += 1

        if cont == Nmax:
            print("El metodo no va bien: numero maximo de iteraciones alcanzado")
        maxiter = max(maxiter, cont)

        y[:, k + 1] = z
        f[:, k + 1] = fun(t[k + 1], y[:, k + 1])

    return (t, y, maxiter)


##########################
### OTROS


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

    for k in range(2):
        t[k + 1] = t[k] + h

        k1 = f[k]
        k2 = fun(t[k] + h / 2, y[k] + h / 2 * k1)
        k3 = fun(t[k] + h / 2, y[k] + h / 2 * k2)
        k4 = fun(t[k + 1], y[k] + h * k3)

        y[k + 1] = y[k] + h / 6 * (k1 + 2 * k2 + 2 * k3 + k4)
        f[k + 1] = fun(t[k + 1], y[k + 1])

    for k in range(2, N):
        count = 0
        dist = 1 + tol

        Ck = y[k] + h / 24.0 * (19 * f[k] - 5 * f[k - 1] + f[k - 2])
        t[k + 1] = t[k] + h
        z = y[k]

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

        y[k + 1] = znew
        f[k + 1] = fun(t[k + 1], y[k + 1])
    return (t, y, maxiter)


def AM3pfAB3(a, b, fun, N, y0):
    tol = 1e-12
    Nmax = 200
    maxiter = 0

    y = zeros(N + 1)
    t = zeros(N + 1)
    f = zeros(N + 1)

    t[0] = a
    h = (b - a) / float(N)
    y[0] = y0
    f[0] = fun(a, y[0])

    # RK4 para obtener semillas iniciales
    for k in range(2):
        t[k + 1] = t[k] + h

        K1 = f[k]
        K2 = fun(t[k] + 0.5 * h, y[k] + 0.5 * h * K1)
        K3 = fun(t[k] + 0.5 * h, y[k] + 0.5 * h * K2)
        K4 = fun(t[k + 1], y[k] + h * K3)

        y[k + 1] = y[k] + h / 6 * (K1 + 2 * K2 + 2 * K3 + K4)
        f[k + 1] = fun(t[k + 1], y[k + 1])

    # Metodo multipaso por punto fijo con AB3 como semilla
    for k in range(2, N):
        cont = 0
        error = 1 + tol

        # Usamos AB3 como semilla del metodo del punto fijo
        z = y[k] + h / 12 * (23 * f[k] - 16 * f[k - 1] + 5 * f[k - 2])

        # AM3
        Ck = y[k] + h / 24 * (19 * f[k] - 5 * f[k - 1] + f[k - 2])
        t[k + 1] = t[k] + h

        while error >= tol and cont < Nmax:  # Metodo del punto ujo
            znew = h * 9 / 24 * fun(t[k + 1], z) + Ck  # Iteracion del punto ujo
            error = abs(z - znew)
            z = znew
            cont += 1

        if cont == Nmax:
            print("El metodo no va bien: numero maximo de iteraciones alcanzado")

        maxiter = max(maxiter, cont)
        y[k + 1] = z
        f[k + 1] = fun(t[k + 1], y[k + 1])

    return (t, y, maxiter)


#################################################################
#################################################################
#################################################################
#################################################################
#################################################################
#################################################################


def f(t, y):
    return -y + 2 * sin(t)


def exacta(t):
    return (pi + 1) * exp(-t) + sin(t) - cos(t)


(t, y, maxiter) = AM3pfAB3(0, 10, f, 50, pi)
ye = exacta(t)

error = max(abs(y - ye))
print(error)

# Resultados
# print('-----')
# print('Tiempo CPU: ', (tfin-tini))
# print('Error: ', error)
# if N != malla[0]:
# print('Order:', (log(errorold)-log(error))/log(2))
# print('Paso de malla: ', (b-a)/N)
# print('-----')


def F(t, y):
    f1 = 3 * y[0] - 2 * y[1]
    f2 = -y[0] + 3 * y[1] - 2 * y[2]
    f3 = -y[1] + 3 * y[2]

    return array([f1, f2, f3])


def Fexacta(t):
    f1 = -1 / 4 * exp(5 * t) + 3 / 2 * exp(3 * t) - 1 / 4 * exp(t)
    f2 = 1 / 4 * exp(5 * t) - 1 / 4 * exp(t)
    f3 = -1 / 8 * exp(5 * t) - 3 / 4 * exp(3 * t) - 1 / 8 * exp(t)

    return array([f1, f2, f3])


y0 = array([1.0, 0.0, -1.0])

(t, y, maxiter) = AM3pfsis(0, 1, F, 100, y0)
ye = Fexacta(t)
error = max(abs(y[2, :] - ye[2]))
print(error)
# max_error = max(error1, error2, error3)

