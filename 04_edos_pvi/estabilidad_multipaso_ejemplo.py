"""
Ejemplo de estabilidad - trapecio y RK4 aplicado a casos de prueba
Incluye calculo de error y orden para ecuacion de prueba especifica.
"""

from pylab import *
from time import perf_counter
from matplotlib.pyplot import *

# get_ipython().run_line_magic('matplotlib','qt')


########################################################
def f1(t, y):
    dz = y[1]
    ddz = -g - (C(y[0]) / M) * y[1] * abs(y[1])
    return array([dz, ddz])


# Datos del problema:
a = 0  # Extremo inferior intervalo
M = 72
y0 = array([2000, 0])  # Condicion inicial
g = 9.81
h = 0.05  # paso del metodo


# Definimos la C
def C(z):
    if z < 1000:
        return 0.3
    else:
        return 40


"""Implementacion del metodo de RK4 para sistemas en el intervalo [a, b]
    usando N particiones y condicion inicial y0"""


def rk4sistemasb(a, f, h, y0):
    t = zeros(1)  # inicializacion del vector de nodos
    y = zeros((len(y0), 1))  # inicializacion del vector de resultados
    t[0] = a  # nodo inicial
    y[:, 0] = y0  # valor inicial
    k = 0  # Iteraccion actual

    # Metodo de rk4
    while y[0, k] >= 0:
        k1 = f(t[k], y[:, k])
        k2 = f(t[k] + 0.5 * h, y[:, k] + 0.5 * h * k1)
        k3 = f(t[k] + 0.5 * h, y[:, k] + 0.5 * h * k2)
        k4 = f(t[k] + h, y[:, k] + h * k3)
        ynext = y[:, k] + h / 6 * (k1 + 2 * k2 + 2 * k3 + k4)

        t = append(t, t[k] + h)
        y = column_stack((y, ynext))
        k = k + 1

    return (t, y)


# Inicializamos el entorno grafico
figure("Caso 1")
(ti, yi) = rk4sistemasb(a, f1, h, y0)

# Dibujamos la grafica de la altura frente al tiempo
subplot(121)
plot(ti, yi[0])
xlabel("t")
ylabel("Altura")

# Dibujamos la grafica de la velocidad frente al tiempo
subplot(122)
plot(ti, yi[1])
xlabel("t")
ylabel("Velocidad")


# Escogemos el ultimo elemento del array yi que es justamente la velocidad en el momento de aterrizaje.
print("La velocidad en el momento de aterrizaje es v =", yi[1, -1])

# Miramos el tiempo cuando la altura es 1000 en la primera grafica y utilizamos ese tiempo para encontrar en la segunda grafica la velocidad.
print("La velocidad en el instante en el que se abre el paracaidas es v2=", -4.13)

##########################################################################
print("\n Caso 2\n")


def f2(t, y):
    return cos(8 * pi * t) * (1 - 5 * y)


def f2exacta(t):
    return 0.2 + 0.8 * exp((-5 / (8 * pi)) * sin(8 * pi * t))


"""Implementacion del metodo del trapecio en el intervalo [a, b]
    usando N particiones y condicion inicial y0"""


def trapecio(a, b, f, N, y0):
    h = (b - a) / N  # paso de malla
    t = zeros(N + 1)  # inicializacion del vector de nodos
    y = zeros(N + 1)  # inicializacion del vector de resultados

    t[0] = a  # nodo inicial
    y[0] = y0  # valor inicial

    # Metodo de trapecio
    for k in range(N):
        t[k + 1] = t[k] + h
        y[k + 1] = (
            y[k] * (1 - 2.5 * h * cos(8 * pi * t[k]))
            + 0.5 * h * (cos(8 * pi * t[k]) + cos(8 * pi * t[k + 1]))
        ) / (1 + h * (5 / 2) * cos(8 * pi * t[k + 1]))

    return (t, y)


# Datos del problema
a = 0.0  # extremo inferior del intervalo
b = 1.0  # extremo superior del intervalo
y0 = 1.0  # condicion inicial

figure("Caso 2")
malla = [40, 80, 160, 320]  # Numero de particiones
for N in malla:
    (t, y) = trapecio(a, b, f2, N, y0)
    ye = f2exacta(t)  # Calculamos la solucion exacta

    if N != 40:
        errorAntiguo = error

    # Calculamos el error
    error = max(abs(y - ye))

    print("-----")
    print("Error: " + str(error))

    # Calculo del cociente
    if N != 40:
        coeficiente = errorAntiguo / error
        print(
            "El coeficiente entre N="
            + str(int(N / 2))
            + " y 2N="
            + str(N)
            + " es "
            + str(coeficiente)
        )
    print("Paso de malla: " + str((b - a) / N))
    print("-----")
    plot(t, y, "-*")  # dibuja la solucion aproximada

plot(t, ye, "k")  # Dibuja la solucion exacta
xlabel("t")
ylabel("y")
leyenda = ["N=" + str(N) for N in malla]
leyenda.append("exacta")
legend(leyenda)
grid(True)
show()

print(
    "Al dividir h por 2, el coeficiente del error se acerca a 4 entonces se trata de un metodo de orden 2"
)
