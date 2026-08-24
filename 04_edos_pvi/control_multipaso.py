"""
Control de multipaso - frontera de estabilidad
Exploracion de region de estabilidad y comportamiento con sistemas.
"""

from pylab import *

print("Caso 1")


def f1(t, y):
    dx = -5 * y[0] + y[1] + 100 * cos(t) - 21 * sin(t)
    dy = -y[0] - 5 * y[1] + 21 * cos(t) + 5 * sin(t)
    return array([dx, dy])


def f1exacta(t):
    return array([20 * cos(t), sin(t)])


def M3sist(a, b, fun, N, y0):
    y = zeros((len(y0), N + 1))
    t = zeros(N + 1)
    f = zeros((len(y0), N + 1))

    t[0] = a
    h = (b - a) / float(N)
    y[:, 0] = y0
    f[:, 0] = fun(t[0], y[:, 0])

    # RK4
    for k in range(2):
        t[k + 1] = t[k] + h
        k1 = fun(t[k], y[:, k])
        k2 = fun(t[k] + 0.5 * h, y[:, k] + 0.5 * h * k1)
        k3 = fun(t[k] + 0.5 * h, y[:, k] + 0.5 * h * k2)
        k4 = fun(t[k + 1], y[:, k] + h * k3)

        y[:, k + 1] = y[:, k] + h / 6 * (k1 + 2 * k2 + 2 * k3 + k4)
        f[:, k + 1] = fun(t[k + 1], y[:, k + 1])

    for k in range(2, N):
        y[:, k + 1] = (
            -1 / 4 * y[:, k]
            + 1 / 2 * y[:, k - 1]
            + 3 / 4 * y[:, k - 2]
            + h / 8 * (19 * f[:, k] + 5 * f[:, k - 2])
        )
        t[k + 1] = t[k] + h
        f[:, k + 1] = fun(t[k + 1], y[:, k + 1])

    return (t, y)


# Datos del problema
a = 0
b = 2 * pi
malla = [160, 320, 640]  # Numero de particiones
y0 = array([20, 0])
error = 0
errorAntiguo = 0

figure("Caso 1")

for N in malla:
    (t, y) = M3sist(a, b, f1, N, y0)
    ye = f1exacta(t)

    subplot(121)
    plot(t, y[0])
    subplot(122)
    plot(y[0], y[1])

    if N != 160:
        errorAntiguo = error
    errorx = max(abs(y[0] - ye[0]))
    errory = max(abs(y[1] - ye[1]))
    error = max(errorx, errory)

    # Resultados
    print("---------------")
    print("Error con N = " + str(N) + " es " + str(error))
    if N != 160:
        orden = (log(errorAntiguo) - log(error)) / log(2)
        print("El orden aproximado es " + str(orden))
    print("---------------")


subplot(121)
xlabel("t")
ylabel("x")
plot(t, ye[0], "k")
leyenda = ["N=" + str(N) for N in malla]
leyenda.append("exacta")
legend(leyenda)

subplot(122)
xlabel("x")
ylabel("y")
plot(ye[0], ye[1], "k")
leyenda = ["N=" + str(N) for N in malla]
leyenda.append("exacta")
legend(leyenda)
show()

print("Caso 2")

figure("Caso 2: Frontera y autovalores")


def locfron(rho, sigma):
    # Dibuja la frontera de la region de estabilidad absoluta
    # de un metodo multipaso.
    # rho y sigma son los coeficientes de los polinomios caracteristicos
    # ordenados de mayor a menor grado '''
    theta = arange(0, 2.0 * pi, 0.01)
    numer = polyval(rho, exp(theta * 1j))  # rho(e^{theta*i})
    denom = polyval(sigma, exp(theta * 1j))  # sigma(e^{theta*i})
    mu = numer / denom
    x = real(mu)
    y = imag(mu)
    plot(x, y)
    grid(True)
    axis("equal")


rho = array([1.0, 1.0 / 4.0, -1.0 / 2.0, -3.0 / 4.0])  # primero
sigma = array([0.0, 19.0 / 8.0, 0, 5.0 / 8.0])  # segundo
locfron(rho, sigma)

# Los autovalores que obtenemos son -5+i y -5-i calculados en la hoja de examen
re = -5
im = 1

plot([re, re], [im, -im], "*")
plot([0, re], [0, im], "--", [0, re], [0, -im], "--")
# Obtenemos el h critica mirando el corte con las rectas pintadas anteriormente,
# como la parte imaginaria de nuestros autovalores es 1, h critica sera la componente 'y'.
# Ya que dividimos la componente 'y' con la parte imaginaria del autovalor.
hcrit = 0.0525
plot([-5 * hcrit, -5 * hcrit], [-1 * hcrit, 1 * hcrit], "o")


figure("Caso 2: Metodo de valores de N menores y mayores")
# Datos del problema
a = 0
b = 2 * pi
z0 = array([100, 0])

N = int((b - a) / hcrit) + 1

mesh = [N - 5, N, N + 5]

leyenda = []

for N in mesh:
    (t, z) = M3sist(a, b, f1, N, z0)
    plot(t, z[0])
    leyenda.append("N = " + str(N))

xlabel("t")
ylabel("x")
legend(leyenda)

# Para N=115 se obtienen oscilaciones, pero para N=120 y N=125 no se obtienen oscilaciones.
