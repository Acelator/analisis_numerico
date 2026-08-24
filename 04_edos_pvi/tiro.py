"""
Metodo del tiro para problemas de contorno
Convierte contorno de 2o orden en PVI y resuelve ecuacion no lineal
G(theta0)=0 con biseccion/secante. Caso cohete balistico.
Ver docs/enunciados_resumidos.md#04_edos_pvi
"""

import time

from numpy import *
import matplotlib.pyplot as plt

# METODO DEL TIRO
#   Convertimos un problema de contorno en un prob.valor.inciales el cual suponemos que tiene solucion
#       Asi buscando los ceros de la resta con el valor incial en el instante b, obtenemos los datos iniciales para el PVI
#       Dicha solucion sera tambien solucion del problema incial.

#   Para encontrar los ceros usamos tanto dicotomia (siempre convergente cuando tengamos un intervalo adeucodo) y secante, mas rapido y no necesitamos
#       encuadrar la solucion aunque se necesita una semilla inicial suficientemente buena


# Tenemos una discretizacion de la funcion, no una funcion en un intervalo
# Por ello cambiamos el metodo para unicamente usar el ultimo valor de la discretizacion, esto es, evaluar en b, justo lo que queremos resolver
def bisec(f, a, b, N, tol):
    print("BISEC")

    an = a
    bn = b
    fan = f(an)[-1]
    fbn = f(bn)[-1]

    if fan == 0:
        print(str(a) + "es raiz de la funcion")
        return a
    elif fbn == 0:
        print(str(b) + "es raiz de la funcion")
        return b
    elif fan * fbn > 0:
        print("No hay cambio de signo: no se puede aplicar el metodo")
        return
    else:
        k = 0
        while True:
            if k > N:
                print(
                    "La aproximacion de la raiz tras "
                    + str(N)
                    + " iteraciones es "
                    + str(cn)
                )
                return cn

            cn = (an + bn) / 2.0
            fcn = f(cn)[-1]
            if abs(bn - an) < 2 * tol:
                print(
                    "La aproximacion de la raiz tras "
                    + str(k)
                    + " iteraciones es "
                    + str(cn)
                )
                return cn

            # print("Aproximacion de la raiz", cn, "y valor f(cn)", fcn)
            if fcn == 0:
                print(str(cn) + "es raiz de la funcion")
                return cn
            elif fan * fcn < 0:
                bn = cn
                fbn = fcn
            else:
                an = cn
                fan = fcn
            k += 1


# Bisec modificado por la cota teorica obtenida en su dia en MNI para encontrar la solucion
def bisecmod(f, a, b, N, tol):
    print("BISECMOD")

    an = a
    bn = b
    fan = f(an)[-1]
    fbn = f(bn)[-1]

    if fan == 0:
        print(str(a) + "es raiz de la funcion")
        return a
    elif fbn == 0:
        print(str(b) + "es raiz de la funcion")
        return b
    elif fan * fbn > 0:
        print("No hay cambio de signo: no se puede aplicar el metodo")
        return
    else:
        # Calculamos N con la cota a priori
        T = int(ceil((log(b - a) - log(tol)) / log(2)))
        for k in range(T):
            cn = (an + bn) / 2.0
            fcn = f(cn)[-1]
            # print('Aproximacion de la raiz',cn,'y valor f(cn)',fcn)
            if fcn == 0:
                print(str(cn) + "es raiz de la funcion")
                return cn
            elif fan * fcn < 0:
                bn = cn
                fbn = fcn
            else:
                an = cn
                fan = fcn
        print(
            "La aproximacion de la raiz tras "
            + str(T)
            + " iteraciones es "
            + str(cn)
            + f" con valor {f(cn)[-1]}"
        )
        return cn


# Necesitamos que la raiz buscada sea simple. No necesitamos que haya cambio de signo en el intervalo
# El metado no busca en el intervalo [a,b] sino en todo R, simplemente, estamos tomando a y b como semillas iniciales
def secante(f, a, b, N, tol):
    print("SECANTE")

    x0 = a
    x1 = b
    fx0 = f(x0)[-1]
    fx1 = f(x1)[-1]

    if fx0 == 0:
        print(str(a) + " es raiz de la funcion")
        return a
    elif fx1 == 0:
        print(str(b) + " es raiz de la funcion")
        return b
    else:
        err = 2 * tol
        n = 0

        while err > tol and n < N:
            x2 = x1 - (x1 - x0) / (fx1 - fx0) * fx1
            n += 1
            fx2 = f(x2)[-1]

            if fx2 == 0:
                print(str(x2) + "es raiz de la funcion")
                return x2

            else:
                err = abs(x2 - x1)
                x0 = x1
                x1 = x2
                fx0 = f(x0)[-1]
                fx1 = f(x1)[-1]

        print(
            f"La aproximacion de la raiz tras {n} iteraciones es {x2}, con un error {err}"
        )
        return x2


# Devuelve una discretizacion de la funcion
# EL ORDEN ES O(4)
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


########################
########################
########################

# Valores para la realizacion del caso 2
a = 0
b = pi
alfa = 0
beta = 25
N = 200
eps = 1e-12


def f(t, Y):
    return array([Y[1], 2 * cos(t) * Y[1] + 0.01 * Y[0]])


# La evaluacion de la aproximacion de la solucion del (PVI) en b
def Fh(a, b, alfa, v, f, N):
    F = RK4_sistemas(a, b, f, N, array([alfa, v]))

    return F[1][0]


def ejer1(a, b, alfa, beta, f, N, tol, variante):
    # Los valores en los que queremos acotar dicotomia (obtenidos mediante estudio grafico de la solucion)
    v0 = 1
    v1 = 2

    z = linspace(a, b, N + 1)

    def dibujar():
        plt.figure()
        plt.title("Valores para v")

        for v in range(0, 5):
            Fh0 = Fh(a, b, alfa, v, f, N)
            plt.plot(z, Fh0, label=f"v: {v}")

        plt.plot(b, beta, "*", label="valor inicial en b")
        plt.legend()
        plt.show()

    dibujar()

    def g(v):
        return Fh(a, b, alfa, v, f, N) - beta

    # Medicion del tiempo de la resolucion de la ecuacion
    ti = time.time()

    if variante == "dicotomia":
        v = bisec(g, v0, v1, N, tol)
    elif variante == "secante":
        v = secante(g, v0, v1, N, tol)
    else:
        print("Variante no valida")
        return

    tf = time.time()
    print(f"Tiempo de ejecucion:  {format(tf - ti)} sec")

    # Las siguientes lineas comentadas pintarian la grafica de F(v) sobre la cual se va a aplicar el metodo de la tangente
    #   Se ve directamente que la grafica es una linea recta, motivo por el cual el metodo de la tangente converge en una unica iteraccion
    #   (dos en mi caso debido a como cuento que es una iteraccion)
    # s = linspace(0, 100, 200)
    # F = array([])
    # for t in s:
    #     Ft = g(t)[-1]
    #     # print(Ft)
    #     F = append(F, Ft)
    # plt.plot(z[:-1], F)
    # plt.show()

    F = Fh(a, b, alfa, v, f, N)
    plt.plot(z, F)
    plt.plot(b, beta, "*")
    plt.title("Grafica de la solucion")
    plt.show()

    return


ejer1(a, b, alfa, beta, f, N, eps, "dicotomia")
ejer1(a, b, alfa, beta, f, N, eps, "secante")

# --- Secante variant ---

import time

from numpy import *
import matplotlib.pyplot as plt


# Necesitamos que la raiz buscada sea simple. No necesitamos que haya cambio de signo en el intervalo
def secante(f, x0, x1, N, tol):
    print("SECANTE")

    fx0 = f(x0)
    fx1 = f(x1)

    if fx0 == 0:
        print(str(a) + "es raiz de la funcion")
        return a
    elif fx1 == 0:
        print(str(b) + "es raiz de la funcion")
        return b
    else:
        err = 2 * tol
        n = 0

        while err > tol and n < N:
            x2 = x1 - (x1 - x0) / (fx1 - fx0) * fx1
            n += 1
            fx2 = f(x2)

            if fx2 == 0:
                print(str(x2) + " es raiz de la funcion")
                return x2

            else:
                err = abs(x2 - x1)
                x0 = x1
                x1 = x2
                fx0 = f(x0)
                fx1 = f(x1)

        print(
            f"La aproximacion de la raiz tras {n} iteraciones es {x2}, con un error {err}"
        )
        return x2


# Devuelve una discretizacion de la funcion
# EL ORDEN ES O(4)
def RK4(a, h, fun, Y0):
    t = zeros(1)
    Y = zeros((len(Y0), 1))

    t[0] = a
    Y[:, 0] = Y0

    k = 0

    while Y[1, -1] >= 0:
        t = append(t, t[k] + h)
        k1 = fun(t[k], Y[:, k])
        k2 = fun(t[k] + h / 2, Y[:, k] + h / 2 * k1)
        k3 = fun(t[k] + h / 2, Y[:, k] + h / 2 * k2)
        k4 = fun(t[k + 1], Y[:, k] + h * k3)

        # Y[:, k + 1] = Y[:, k] + h / 6 * (k1 + 2 * k2 + 2 * k3 + k4)
        Y = column_stack((Y, Y[:, k] + h / 6 * (k1 + 2 * k2 + 2 * k3 + k4)))
        k += 1

    return (t, Y)


########################
########################
########################

# Valores de 2)
a = 0
b = pi
alfa = 0
beta = 25
N = 200
eps = 1e-12

g = 9.81

xf = 50
v0 = 25
M = 7.5
C = 0.03
h = 0.01
N = 300  # Numero de iteracciones maximas realizables por el metodo de la secante


def f(t, Y):
    return array(
        [
            Y[2] * cos(Y[3]),
            Y[2] * sin(Y[3]),
            -C / M * Y[2] * Y[2] - g * sin(Y[3]),
            -g / Y[2] * cos(Y[3]),
        ]
    )


# La evaluacion de la aproximacion de la solucion del (PVI) en b
def F(a, h, v0, theta0, f):
    F = RK4(a, h, f, array([0, 0, v0, theta0]))

    return F


def ejer2(xf, v0, f, tol):
    def g(theta):
        return F(a, h, v0, theta, f)[1][0][-1] - xf

    # Dibujamos los resultados de todas las aprox de angulos entre 0 y pi/2. Vemos que existen dos soluciones
    for t in range(0, 200, 5):
        R = F(a, h, v0, pi / 400 * t, f)
        plt.plot(R[1][0], R[1][1], label=f"theta: {t}")
    plt.title("Grafica asociada a multiples valores de theta")
    plt.show()

    # Mediccion del tiempo
    ti = time.time()

    # Los valores en los que queremos acotar dicotomia (obtenidos mediante estudio grafico de la solucion)
    A0 = pi / 16
    A1 = pi / 4

    theta1 = secante(g, A0, A1, N, tol)

    tf = time.time()

    # Alternativamente tenemos otra solucion
    A0 = pi / 4
    A1 = pi / 2
    theta2 = secante(g, A0, A1, N, tol)

    # Soluciones para los valores del angulo obtenido anteriormente
    R = F(a, h, v0, theta1, f)
    S = F(a, h, v0, theta2, f)

    print(
        f"Tiempo de vuelo de la solucion con theta {round(theta1, 5)}: {round(R[0][-1], 4)} seg"
    )
    print(
        f"Altura maxima de la solucion con theta {round(theta1, 5)}: {round(max(R[1][1]), 5)} metros"
    )
    print("-------------------------")
    print(
        f"Tiempo de vuelo de la solucion con theta {round(theta2, 5)}: {round(R[0][-1], 4)} seg"
    )
    print(
        f"Altura maxima de la solucion con theta {round(theta2, 5)}: {round(max(S[1][1]), 5)} metros"
    )

    plt.plot(R[1][0], R[1][1], label=f"{round(theta1, 5)}")
    plt.plot(S[1][0], S[1][1], label=f"{round(theta2, 5)}")
    plt.title("Grafica de la solucion")
    plt.legend()
    plt.show()

    # plt.plot(R[0], R[1][1])
    # plt.show()

    print(
        f"Tiempo de ejecucion del calculo de la primera solucion:  {format(tf - ti)} sec"
    )

    return


ejer2(xf, v0, f, eps)
