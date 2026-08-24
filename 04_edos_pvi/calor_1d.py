"""
Calor 1D - explicito, implicito y Crank-Nicolson
Ecuacion u_t - c u_xx = 0 y extensiones adveccion-difusion
(u_t -c u_xx + v u_x=0) y ondas u_tt = c2 u_xx.
Ver docs/enunciados_resumidos.md#04_edos_pvi
"""

from matplotlib.pyplot import *
from numpy import *
import time

"""Resuelve u_t - c u_xx = 0 en el intervalo [a,b]
con condicion inicial u(x,0) = ci(x)
y condiciones de Dirichlet u(a,t) = alpha, u(b,t) = beta
usando el metodo explicito con N+1 puntos y pasos dx = (b-a)/(N +1) y dt dado.
Devuelve la malla x y la solucion en el tiempo final
Si iplot = 1 pone en pantalla la animacion."""


print("-------------- CASO 1 --------------------")


def calor(a, b, T, N, ci, alpha, beta, c, dt, iplot, exac):
    dx = (b - a) / (N + 1)
    x = linspace(a, b, N + 2)
    u0 = ci(x)
    u1 = zeros(N + 2)

    if iplot == 1:
        clf()
        plot(x, u0, "*-")
        xlabel("x")
        ylabel("u")
        title("Tiempo: 0")
        z0 = min(u0)
        z1 = max(u0)
        axis([a, b, z0, z1])
        pause(0.1)

    t = 0
    print(f"dx: {round(dx, 5)}, dt: {round(dt, 5)}")

    # Comparamos con la condiccion de estabilidad obtenida en clase. Si devuelve true es estable.
    # el dt que viene predeterminado en la seccion (dt=0.3) es inestable, si lo hacemos mas pequeno
    # obtenemos la estabilidad que deseamos.
    print(f"Es el metodo estable? {dt <= dx * dx / 2 * c}")

    while t < T:
        # print(t)
        dt = min([dt, T - t])
        t = t + dt
        coef = c * dt / (dx * dx)
        u1[1 : N + 1] = (1 - 2 * coef) * u0[1 : N + 1] + coef * (
            u0[0:N] + u0[2 : N + 2]
        )
        u1[0] = alpha
        u1[N + 1] = beta

        if iplot == 1:
            # print(f"t = {t:.2f}")
            clf()
            plot(x, u1, "*-")
            plot(x, exac(x, t), label="Exacta")
            title(f"Tiempo: {t:.2f}")
            axis([a, b, z0, z1])
            pause(0.1)

        u0 = u1.copy()

    return x, u1


def ci(x):
    y = sin(pi * x / 10)
    return y


def exacta(x, t):
    y = sin(pi * x / 10) * exp(-t * (pi / 10) ** 2)
    return y


a = 0
b = 10
c = 1
T = 10
N = 20
dt = 0.3
alpha = 0
beta = 0

dx = (b - a) / (N + 1)
print(dx, 0.5 * dx * dx / c)


x, u = calor(a, b, T, N, ci, alpha, beta, c, 0.3, 0, exacta)  # Inestable
x, u = calor(a, b, T, N, ci, alpha, beta, c, 0.2, 0, exacta)  # Inestable
x, u = calor(a, b, T, N, ci, alpha, beta, c, 0.1, 0, exacta)  # Estable


print("-------------- CASO 2 --------------------")


# Imponemos el paso de tiempo como se pide en el enunciado
def calor_estable(a, b, T, N, ci, alpha, beta, c, iplot, exac):
    dx = (b - a) / (N + 1)
    x = linspace(a, b, N + 2)
    u0 = ci(x)
    u1 = zeros(N + 2)

    if iplot == 1:
        clf()
        plot(x, u0, "*-")
        xlabel("x")
        ylabel("u")
        title("Tiempo: ")
        z0 = min(u0)
        z1 = max(u0)
        axis([a, b, z0, z1])
        pause(0.1)

    dt = 1 / 2 * dx * dx / c
    print(f"dx: {round(dx, 5)}, dt: {round(dt, 5)}")
    t = 0

    while t < T:
        dt = min([dt, T - t])
        t = t + dt
        coef = c * dt / (dx * dx)
        u1[1 : N + 1] = (1 - 2 * coef) * u0[1 : N + 1] + coef * (
            u0[0:N] + u0[2 : N + 2]
        )
        u1[0] = alpha
        u1[N + 1] = beta

        if iplot == 1:
            # print(f"t = {t:.2f}")
            clf()
            plot(x, u1, "*-")
            plot(x, exac(x, t), label="Exacta")
            title(f"Tiempo: {t:.2f}")
            axis([a, b, z0, z1])
            pause(0.1)

        u0 = u1.copy()

    return x, u1


# x, u = calor_estable(a, b, T, 200, ci, alpha, beta, c, 1, exacta)


# Midamos el orden
errorAntiguo = 0

# calculo = []
calculo = [10, 20, 40, 80, 160]
if calculo.__len__() != 0:
    # print("Caso 2")
    # print("-------------------------")

    for i in calculo:
        x, u = calor_estable(a, b, T, i, ci, alpha, beta, c, 0, exacta)
        Ue = exacta(x, T)

        print("-------------------------")

        nuevo_error = max(abs(Ue - u))  # error cometido
        print(nuevo_error)

        if errorAntiguo == 0:
            print("ERROR", nuevo_error)
        else:
            print("Orden aproximado", log2(errorAntiguo / nuevo_error))

        errorAntiguo = nuevo_error

    # pause(50000)

# Como se puede ver el orden es dos, en concordancia con lo estudiado en clase. (Realmente vimos que dos en espacio y uno en tiempo, pero por la imposicion
#   de la condiccion de estabilidad, tenemos unicamente orden 2).


print("-------------- CASO 3 --------------------")
figure()


def calor_fuente(a, b, T, N, ci, alpha, g, c, f, iplot):
    dx = (b - a) / (N + 1)
    x = linspace(a, b, N + 2)
    u0 = ci(x)
    u1 = zeros(N + 2)

    if iplot == 1:
        clf()
        plot(x, u0, "*-")
        xlabel("x")
        ylabel("u")
        title("Tiempo: 0")
        z0 = min(u0)
        z1 = max(u0)
        axis([a, b, z0, z1])
        pause(0.1)

    # Para asegurar que los errores de redondeo no me tiran el dt por encima de la barrera
    dt = 0.49 * dx * dx / c
    print(f"dx: {round(dx, 5)}, dt: {round(dt, 5)}")
    t = 0

    # Cuando hize inicialmente el caso estaba puesto T=10 o T=5 (no me acuerdo ya) y entonces tardaba mucho en pintar....
    #   Por eso anadi el siguiente codigo que pone las constantes que usa el bucle de ploteo para determinar
    #       si ha pasado aproximadamente 0.01 seg desde la ultima vez que se dibujo la grafica y en dicho
    #       caso representarla.
    plot_interval = 0.01
    next_plot = plot_interval
    tol = 1e-9
    tiempo = 0

    while t < T:
        # No contamos tiempo en dibujar la grafica
        start_time = time.perf_counter()

        dt = min([dt, T - t])
        t = t + dt
        coef = c * dt / (dx * dx)
        u1[1 : N + 1] = (
            (1 - 2 * coef) * u0[1 : N + 1]
            + coef * (u0[0:N] + u0[2 : N + 2])
            + f(x[1 : N + 1], t) * dt
        )
        u1[0] = alpha

        # Condiccion contorno de neumann
        u1[N + 1] = (
            u0[N + 1] + 2 * coef * (u0[N] - u0[N + 1] + g * dx) + f(x[N + 1], t) * dt
        )

        u0 = u1.copy()

        end_time = time.perf_counter()
        tiempo += end_time - start_time

        # Bucle de ploteo
        while iplot == 1 and next_plot <= t + tol:
            # print(f"t = {next_plot:.2f}")
            clf()
            plot(x, u1, "*-")
            title(f"Tiempo: {next_plot:.2f}")
            axis([a, b, z0, z1])
            pause(0.1)
            next_plot += plot_interval

    print(f"Tiempo CPU de resolucion (Crank-Nicolson): {tiempo:.6f} segundos")

    show(block=True)
    return x, u1


def ci(x):
    return 4 * x * (1 - x)


def fuente(x, t):
    return 1 + cos(2 * pi * x)


a = 0
b = 1
c = 1
T = 0.05
N = 100
dt = 0.01
alpha = 0
g = 0

dx = (b - a) / (N + 1)
print(dx, 0.5 * dx * dx / c)


x, u = calor_fuente(a, b, T, N, ci, alpha, g, c, fuente, 1)

# --- Implicito y Crank ---


from numpy import *
from matplotlib.pyplot import (
    clf,
    plot,
    xlabel,
    ylabel,
    title,
    axis,
    pause,
    figure,
    legend,
    show,
)
from scipy.sparse import spdiags
from scipy.sparse.linalg import spsolve, splu

import time


# Datos caso 1 y 2
def ci(x):
    y = sin(pi * x / 10)
    return y


def exacta(x, t):
    y = sin(pi * x / 10) * exp(-t * (pi / 10) ** 2)
    return y


a = 0
b = 10
c = 1
T = 10
N = 20
dt = 0.3
alpha = 0
beta = 0

dx = (b - a) / (N + 1)


print("----------- Caso 1  --------------")


def calor_implicito(a, b, T, N, ci, alpha, beta, c, iplot, exac):
    dx = (b - a) / (N + 1)
    x = linspace(a, b, N + 2)
    u0 = ci(x)
    u1 = zeros(N + 2)

    z0 = min(u0)
    z1 = max(u0)

    if iplot:
        figure()

    # Imposicion de la condicion k = h exigida en el enunciado
    dt = dx
    print(f"dx: {round(dx, 5)}, dt: {round(dt, 5)}")
    t = 0

    # Construccion del operador discreto
    coef = c * dt / (dx * dx)

    main_diag = (1 + 2 * coef) * ones(N)
    off_diag = -coef * ones(N)
    A = spdiags([off_diag, main_diag, off_diag], [-1, 0, 1], N, N).tocsc()

    LU = splu(A)

    while t < T:
        dt = min([dt, T - t])
        t = t + dt

        if t == T:
            # Como dt varia en este caso tenemos que reconstruir la matriz
            coef = c * dt / (dx * dx)

            main_diag = (1 + 2 * coef) * ones(N)
            off_diag = -coef * ones(N)
            A = spdiags([off_diag, main_diag, off_diag], [-1, 0, 1], N, N).tocsc()

            LU = splu(A)

        # Termino independiente
        F = u0[1 : N + 1].copy()
        F[0] += coef * alpha
        F[-1] += coef * beta

        # Resolucion del sistema lineal AU^{n+1} = F
        u1[1 : N + 1] = LU.solve(F)

        u1[0] = alpha
        u1[N + 1] = beta

        if iplot == 1:
            # if iplot == 1 and abs(t * 20 - round(t * 20)) < 1e-9:
            clf()
            plot(x, u1, "*-", label=f"N: {N}")
            plot(x, exac(x, t), label="Exacta")
            legend()
            title(f"Tiempo: {t:.2f} | Caso 1")
            axis([a, b, z0, z1])
            pause(0.1)

        u0 = u1.copy()

    return x, u1


# Midamos el orden
errorAntiguo = 0

# calculo = []
calculo = [10, 20, 40, 80, 160]
if calculo.__len__() != 0:
    print("Calor implicito")
    print("-------------------------")

    for i in calculo:
        x, u = calor_implicito(a, b, T, i, ci, alpha, beta, c, 0, exacta)
        Ue = exacta(x, T)

        nuevo_error = max(abs(Ue - u))  # error cometido
        print(f"error: {nuevo_error}")

        if errorAntiguo == 0:
            print("ERROR", nuevo_error)
        else:
            print("Orden aproximado", log2(errorAntiguo / nuevo_error))

        print("-------------------------")
        errorAntiguo = nuevo_error

    # pause(50000)

print("----------- Caso 2  --------------")


def calor_crank_nicolson(a, b, T, N, ci, alpha, beta, c, iplot, exac):
    dx = (b - a) / (N + 1)
    x = linspace(a, b, N + 2)
    u0 = ci(x)
    u1 = zeros(N + 2)

    z0 = min(u0)
    z1 = max(u0)

    if iplot:
        figure()

    # Imposicion de la condicion k = h exigida en el enunciado
    dt = dx
    print(f"dx: {round(dx, 5)}, dt: {round(dt, 5)}")
    t = 0

    # Construccion del operador discreto Crank-Nicolson
    coef = c * dt / (dx * dx)

    main_diag = (1 + coef) * ones(N)
    off_diag = -coef / 2 * ones(N)
    A = spdiags([off_diag, main_diag, off_diag], [-1, 0, 1], N, N).tocsc()
    LU = splu(A)

    while t < T:
        dt = min([dt, T - t])
        t = t + dt

        if t == T:
            coef = c * dt / (dx * dx)

            # Como dt varia en este caso tenemos que reconstruir la matriz
            main_diag = (1 + coef) * ones(N)
            off_diag = -coef / 2 * ones(N)
            A = spdiags([off_diag, main_diag, off_diag], [-1, 0, 1], N, N).tocsc()

            LU = splu(A)

        # Evaluacion del operador explicito
        F = (1 - coef) * u0[1 : N + 1] + (coef / 2) * (u0[0:N] + u0[2 : N + 2])
        F[0] += (coef / 2) * alpha
        F[-1] += (coef / 2) * beta

        # Resolucion del sistema lineal AU^{n+1} = F
        u1[1 : N + 1] = LU.solve(F)

        # Imponemos condicciones de contorno
        u1[0] = alpha
        u1[N + 1] = beta

        # if iplot == 1 and abs(t * 20 - round(t * 20)) < 1e-9:
        if iplot == 1:
            clf()
            plot(x, u1, "*-", label=f"N={N}")
            plot(x, exac(x, t), label="Exacta")
            title(f"Tiempo: {t:.2f} | Caso 2")
            legend()
            axis([a, b, z0, z1])
            pause(0.1)

        u0 = u1.copy()

    return x, u1


# Midamos el orden
errorAntiguo = 0

# calculo = []
calculo = [10, 20, 40, 80, 160]
if calculo.__len__() != 0:
    print("Calor esquema crank nicolson")
    print("-------------------------")

    for i in calculo:
        x, u = calor_crank_nicolson(a, b, T, i, ci, alpha, beta, c, 0, exacta)
        Ue = exacta(x, T)

        nuevo_error = max(abs(Ue - u))  # error cometido
        print(f"error: {nuevo_error}")

        if errorAntiguo == 0:
            print("ERROR", nuevo_error)
        else:
            print("Orden aproximado", log2(errorAntiguo / nuevo_error))

        print("-------------------------")
        errorAntiguo = nuevo_error

    # pause(50000)


print("----------- Caso 3  --------------")


def calor_fuente_CN(a, b, T, N, ci, alpha, g, c, f, iplot):
    dx = (b - a) / (N + 1)
    x = linspace(a, b, N + 2)
    u0 = ci(x)
    u1 = zeros(N + 2)

    if iplot == 1:
        z0 = min(u0)
        z1 = max(u0)
        figure()

    # Imponemos condicion del enunciado para los pasos de malla
    dt = dx
    # dt = 0.49 * dx * dx / c
    print(f"dx: {round(dx, 5)}, dt: {round(dt, 5)}")
    t = 0

    coef = c * dt / (dx * dx)

    # Construccion de la matriz del sistema Crank-Nicolson
    main_diag = (1 + coef) * ones(N + 1)
    off_diag = -(coef / 2) * ones(N + 1)

    A = spdiags([off_diag, main_diag, off_diag], [-1, 0, 1], N + 1, N + 1).tolil()

    # Imposicion del nodo fantasma en x=b
    A[N, N - 1] = -coef
    A = A.tocsc()

    LU = splu(A)

    # Control en dibujado
    plot_interval = 0.0010
    next_plot = plot_interval
    tol = 1e-9

    # Inicio de la medicion de tiempo CPU
    tiempo = 0

    while t < T:
        # No contamos tiempo en dibujar la grafica
        start_time = time.perf_counter()
        dt = min([dt, T - t])
        t = t + dt

        if t == T:
            # Como cambia el dt, cambia el coeficiente y hay que recalcular
            coef = c * dt / (dx * dx)

            # Construccion de la matriz del sistema Crank-Nicolson
            main_diag = (1 + coef) * ones(N + 1)
            off_diag = -(coef / 2) * ones(N + 1)

            A = spdiags(
                [off_diag, main_diag, off_diag], [-1, 0, 1], N + 1, N + 1
            ).tolil()

            # Imposicion del nodo fantasma en x=b
            A[N, N - 1] = -coef
            A = A.tocsc()

            LU = splu(A)

        # Construccion del termino independiente F
        F = zeros(N + 1)

        # Nodos interiores
        F[0:N] = (
            (1 - coef) * u0[1 : N + 1]
            + (coef / 2) * (u0[0:N] + u0[2 : N + 2])
            + (f(x[1 : N + 1], t - dt) + f(x[1 : N + 1], t)) * dt * 1 / 2
        )
        # Ajuste de Dirichlet en x=a
        F[0] += (coef / 2) * alpha

        # Ajuste en x=b (nodo fantasma al lado derecho)
        F[N] = (
            coef * u0[N]
            + (1 - coef) * u0[N + 1]
            + 2 * coef * g * dx
            + (f(x[N + 1], t - dt) + f(x[N + 1], t)) * dt * 1 / 2
        )

        # Resolucion del sistema lineal
        u1[1 : N + 2] = LU.solve(F)
        u1[0] = alpha

        end_time = time.perf_counter()
        tiempo += end_time - start_time

        # Bucle de ploteo
        while iplot == 1 and next_plot <= t + tol:
            clf()
            plot(x, u1, "*-", label=f"N={N}")
            title(f"Tiempo: {next_plot:.2f}")
            legend()
            axis([a, b, z0, z1])
            pause(0.1)
            next_plot += plot_interval

        u0 = u1.copy()

    # Fin de la medicion de tiempo CPU
    print(f"Tiempo CPU de resolucion (Crank-Nicolson): {tiempo:.6f} segundos")

    # Bloqua la ejecucion hasta cerrar la ventana de dibujado
    # show(block=True)
    return x, u1


def ci(x):
    return 4 * x * (1 - x)


def fuente(x, t):
    return 1 + cos(2 * pi * x)


a = 0
b = 1
c = 1
T = 0.05
N = 100
dt = 0.01
alpha = 0
g = 0

dx = (b - a) / (N + 1)


x, u = calor_fuente_CN(a, b, T, N, ci, alpha, g, c, fuente, 1)
print(u[-1])


# Tarda 0.000764 segundos el metodo de Crank-Nicolson con k=h aplicado al ultimo caso de la seccion 5 (media 5 ejecuciones)
# Tarda 0.007441 segundos el metodo de diferencias finitas explicito de segundo orden con k=1/2 * dx^2 / c aplicado al ultimo caso de la seccion 5 (media 5 ejecuciones)

# --- Adveccion y ondas ---


from numpy import *
from matplotlib.pyplot import (
    clf,
    plot,
    xlabel,
    ylabel,
    title,
    axis,
    pause,
    figure,
    legend,
    show,
)
from scipy.sparse import spdiags, diags
from scipy.sparse.linalg import spsolve, splu

import time


# Datos caso 1 y 2
def u0(x):
    return x + 0.1 * x * (10 - x)


a = 0
b = 10
c = 1
T = 10
N = 20
alpha = 0
beta = 10


print("----------- Caso 1  --------------")


def advDifImplicito(a, b, T, N, ci, alpha, beta, c, v, iplot):
    dx = (b - a) / (N + 1)
    x = linspace(a, b, N + 2)
    u0 = ci(x)
    u1 = zeros(N + 2)

    if iplot:
        figure()

    # Imposicion de la condicion k = h exigida en el enunciado
    dt = dx
    print(f"dx: {round(dx, 5)}, dt: {round(dt, 5)}")
    t = 0

    # Construccion del operador discreto
    coefAdv = v * dt / (2 * dx)
    coefDif = c * dt / (dx * dx)

    main_diag = (1 + 2 * coefDif) * ones(N)
    off_diag_sup = (coefAdv - coefDif) * ones(N)
    off_diag_inf = -(coefAdv + coefDif) * ones(N)
    A = diags([off_diag_inf, main_diag, off_diag_sup], [-1, 0, 1], shape=(N, N)).tocsc()

    LU = splu(A)

    while t < T:
        dt_actual = min([dt, T - t])
        t = t + dt_actual

        # No comparamos con t o T directamente para evitar errores de aritmetica flotante.
        if dt != dt_actual:
            dt = dt_actual
            # Como dt varia en este caso, tenemos que reconstruir la matriz
            # Construccion del operador discreto
            coefAdv = v * dt / (2 * dx)
            coefDif = c * dt / (dx * dx)

            main_diag = (1 + 2 * coefDif) * ones(N)
            off_diag_sup = (coefAdv - coefDif) * ones(N)
            off_diag_inf = -(coefAdv + coefDif) * ones(N)
            A = diags(
                [off_diag_inf, main_diag, off_diag_sup], [-1, 0, 1], shape=(N, N)
            ).tocsc()
            LU = splu(A)

        # Termino independiente
        F = u0[1 : N + 1].copy()
        F[0] += (coefAdv + coefDif) * alpha
        F[-1] += (-coefAdv + coefDif) * beta

        # Resolucion del sistema lineal AU^{n+1} = F
        u1[1 : N + 1] = LU.solve(F)

        u1[0] = alpha
        u1[N + 1] = beta

        if iplot == 1:
            # if iplot == 1 and abs(t * 20 - round(t * 20)) < 1e-9:
            clf()
            plot(x, u1, "*-", label=f"v: {v}")
            # plot(x, exac(x, t), label="Exacta")
            legend()
            title(f"Tiempo: {t:.2f} | Caso 1")
            # axis([a, b, z0, z1])
            pause(0.1)

        u0 = u1.copy()

    show(block=True)
    return x, u1


# v = []
v = [0, 0.01, 1, 5, 10]
if v.__len__() != 0:
    print("Ec adv-dif / Dirichlet / Implicito")
    print("-------------------------")

    soluciones = []
    for i in v:
        x, u = advDifImplicito(a, b, T, N, u0, alpha, beta, c, i, 0)
        soluciones.append(u)
        print("-------------------------")

    figure()
    for idx, i in enumerate(v):
        plot(x, soluciones[idx], "*-", label=f"v={i}")
    plot(x, u0(x), label="Condicion inicial", linestyle="--", linewidth=2)
    title("Caso 1 - Ec adv-dif / Dirichlet / Implicito")
    legend()
    show()


# Si v=0 -> Tenemos un sistema que modela un proceso de difusion puro
# Segun aumentamos el valor de v, incrementa la aportacion del proceso de difusion al modelo
# Se comprueba que segun aumentamos el valor de v, la solucion va adquiriendo paulatinamente un peor comportamiento
# Para un v grande, realmente lo que tenemos es un modelo de transporte y aqui sabemos que la solucion unicamente depende de la condicion inicial y de las condicciones de contorno

# Para el modelo que aqui se trata en clase estudiamos su estabilidad en norma infinito.
#   Vimos que es estable si h <= 2c/(norma infinito p). (Se vio para q >= 0, pero revisando la demostracion esta no cambia si q=0)
#   Asi en este caso tenemos que cuando v = 5 o 10, esta condicion no se verifica y no tenemos garantizada la estabilidad en norma infinito.
#   Es este el motivo por el que empezamos a ver las oscilaciones, ya que no tenemos garantizada que el maximo de los nodos este acotado.
#   Si aumentamos el mallado, por ejemplo, N -> 100, la condicion se verificaria y tendriamos estabilidad bajo dicha norma.

print("----------- Caso 2  --------------")


# Me equivoque e implemente inicialmente el metodo upwind explicito, es decir, en el tiempo n en vez de n+1 el termino correspondiente al upwind.
#   Dejo el codigo porque bueno, ya lo tenia implementado, pero el que esta bien y el que uso
#       para hacer el caso es la funcion que sigue
def advDifExplicitoUpwind(a, b, T, N, ci, alpha, beta, c, v, iplot):
    dx = (b - a) / (N + 1)
    x = linspace(a, b, N + 2)
    u0 = ci(x)
    u1 = zeros(N + 2)

    if iplot:
        figure()

    # Imposicion de la condicion k = h exigida en el enunciado
    dt = dx
    print(f"dx: {round(dx, 5)}, dt: {round(dt, 5)}")
    t = 0

    # Construccion del operador discreto
    coefAdv = v * dt / (dx)
    coefDif = c * dt / (dx * dx)

    main_diag = (1 + 2 * coefDif) * ones(N)
    off_diag = -coefDif * ones(N)
    A = spdiags([off_diag, main_diag, off_diag], [-1, 0, 1], N, N).tocsc()

    LU = splu(A)

    while t < T:
        dt_actual = min([dt, T - t])
        t = t + dt_actual

        if dt != dt_actual:
            dt = dt_actual
            # Como dt varia en este caso tenemos que reconstruir la matriz
            # Construccion del operador discreto
            coefAdv = v * dt / (dx)
            coefDif = c * dt / (dx * dx)

            main_diag = (1 + 2 * coefDif) * ones(N)
            off_diag = -coefDif * ones(N)
            A = spdiags([off_diag, main_diag, off_diag], [-1, 0, 1], N, N).tocsc()
            LU = splu(A)

        # Termino independiente
        F = u0[1 : N + 1].copy() * (1 - coefAdv) + u0[0:N].copy() * coefAdv
        F[0] += coefDif * alpha
        F[-1] += coefDif * beta

        # Resolucion del sistema lineal AU^{n+1} = F
        u1[1 : N + 1] = LU.solve(F)

        u1[0] = alpha
        u1[N + 1] = beta

        if iplot == 1:
            # if iplot == 1 and abs(t * 20 - round(t * 20)) < 1e-9:
            clf()
            plot(x, u1, "*-", label=f"v: {v}")
            # plot(x, exac(x, t), label="Exacta")
            legend()
            title(f"Tiempo: {t:.2f} | Caso 1")
            # axis([a, b, z0, z1])
            pause(0.1)

        u0 = u1.copy()

    return x, u1


def advDifImplicitoUpwind(a, b, T, N, ci, alpha, beta, c, v, iplot):
    dx = (b - a) / (N + 1)
    x = linspace(a, b, N + 2)
    u0 = ci(x)
    u1 = zeros(N + 2)

    if iplot:
        figure()

    # Imposicion de la condicion k = h, exigida en el enunciado
    dt = dx
    print(f"dx: {round(dx, 5)}, dt: {round(dt, 5)}")
    t = 0

    # Constructor del operador
    coefAdv = v * dt / dx
    coefDif = c * dt / (dx * dx)

    main_diag = (1 + 2 * coefDif + coefAdv) * ones(N)
    off_diag_sup = -coefDif * ones(N)
    off_diag_inf = -(coefDif + coefAdv) * ones(N)

    A = spdiags([off_diag_inf, main_diag, off_diag_sup], [-1, 0, 1], N, N).tocsc()
    LU = splu(A)

    while t < T:
        dt_actual = min([dt, T - t])
        t = t + dt_actual

        if dt != dt_actual:
            dt = dt_actual
            # Como dt varia en este caso, tenemos que reconstruir la matriz
            # Construccion del operador discreto
            coefAdv = v * dt / dx
            coefDif = c * dt / (dx * dx)

            main_diag = (1 + 2 * coefDif + coefAdv) * ones(N)
            off_diag_sup = -coefDif * ones(N)
            off_diag_inf = -(coefDif + coefAdv) * ones(N)

            A = spdiags(
                [off_diag_inf, main_diag, off_diag_sup], [-1, 0, 1], N, N
            ).tocsc()
            LU = splu(A)

        # Termino independiente
        F = u0[1 : N + 1].copy()

        # Condicciones de contorno dirichlet
        F[0] += (coefDif + coefAdv) * alpha
        F[-1] += coefDif * beta

        # Resolucion
        u1[1 : N + 1] = LU.solve(F)
        u1[0] = alpha
        u1[N + 1] = beta

        if iplot == 1:
            # if iplot == 1 and abs(t * 20 - round(t * 20)) < 1e-9:
            clf()
            plot(x, u1, "*-", label=f"v: {v}")
            # plot(x, exac(x, t), label="Exacta")
            legend()
            title(f"Tiempo: {t:.2f} | Caso 1")
            # axis([a, b, z0, z1])
            pause(0.1)

        u0 = u1.copy()

    return x, u1


# # v = []
# v = [0, 0.01, 1, 5, 10]
# if v.__len__() != 0:
#     print("Ec adv-dif / Dirichlet / Explicito con Upwind")
#     print("-------------------------")

#     soluciones = []
#     for i in v:
#         x, u = advDifExplicitoUpwind(a, b, T, N, u0, alpha, beta, c, i, 0)
#         soluciones.append(u)
#         print("-------------------------")

#     figure()
#     for idx, i in enumerate(v):
#         plot(x, soluciones[idx], label=f"v={i}")
#     plot(x, u0(x), label="Condicion inicial", linestyle="--", linewidth=2)
#     title("Caso 2 - Ec adv-dif / Dirichlet / Explicito con Upwind")
#     legend()
#     show()


# v = []
v = [0, 0.01, 1, 5, 10]
if v.__len__() != 0:
    print("Ec adv-dif / Dirichlet / Implicito con Upwind")
    print("-------------------------")

    soluciones = []
    for i in v:
        x, u = advDifImplicitoUpwind(a, b, T, N, u0, alpha, beta, c, i, 0)
        soluciones.append(u)
        print("-------------------------")

    figure()
    for idx, i in enumerate(v):
        plot(x, soluciones[idx], "*-", label=f"v={i}")
    plot(x, u0(x), label="Condicion inicial", linestyle="--", linewidth=2)
    title("Caso 2 - Ec adv-dif / Dirichlet / Implicito con Upwind")
    legend()
    show()

    print("-------------------------")


# Observamos que tras implementar upwind se mejora el comportamiento de las soluciones, eliminando las oscilaciones no fisicas.
# Para v pequeno, como el factor de advencion tiene poco peso, no se nota diferencia practicamente respecto al caso anterior.
# Para v mas grandes, el modelo de transporte gana peso y el esquema upwind capta mejor la direccion del flujo.

# Si se coge papel y boli y se pone uno a echar las cuentas de estabilidad en norma infinito para este esquema, se obtiene que el esquema es incondiccionalmente estable bajo || . ||_infty.
#   Esto se aprecia en la grafica, donde como ya he mencionado, no se ven oscilaciones extranas.

print("----------- Caso 3  --------------")


# Datos caso 1 y 2
def u0(x):
    return exp(-(x - 5) * (x - 5))


def v0(x):
    # Multiplicamos por x para que numpy pueda vectorizar la funcion
    return 0 * x


a = 0
b = 10
c = 1
T = 10
N = 50
alpha = 0
beta = 0


def ondasExplicito(a, b, T, N, ci, vi, alpha, beta, c, iplot):
    dx = (b - a) / (N + 1)
    x = linspace(a, b, N + 2)

    if iplot:
        figure()

    # Imposicion de la condicion k = h/c impuesta en el enunciado
    dt = dx / c
    print(f"dx: {round(dx, 5)}, dt: {round(dt, 5)}")
    t = 0

    # Iniciacion valores
    u0 = ci(x)
    u1 = u0 + dt * vi(x)
    u2 = zeros(N + 2)

    # Construccion del operador discreto
    coef = c * c * dt * dt / (dx * dx)

    while t < T:
        dt_actual = min([dt, T - t])
        t = t + dt_actual

        if dt != dt_actual:
            dt = dt_actual
            # Como dt cambia, tenemos que recalcular
            coef = c * c * dt * dt / (dx * dx)

        # Resolucion del sistema lineal obtenido por el metodo explicito
        u2[1 : N + 1] = (
            2 * (1 - coef) * u1[1 : N + 1]
            + coef * u1[0:N]
            + coef * u1[2 : N + 2]
            - u0[1 : N + 1]
        )
        u2[0] = alpha
        u2[N + 1] = beta

        if iplot == 1:
            # if iplot == 1 and abs(t * 20 - round(t * 20)) < 1e-9:
            clf()
            plot(x, u1, "*-", label=f"N: {N}")
            legend()
            title(f"Tiempo: {t:.2f} | Caso 3")
            # Si no ajustamos los ejes no se aprecia bien el fenomeno
            axis([a, b, -1.25, 1.25])
            pause(0.1)

        u0 = u1.copy()
        u1 = u2.copy()

    # La linea siguiente bloquea la ejecucion hasta que se cierre la grafica
    #   Si no la uso, cuando programo se me cierra automaticamente la grafica y no puedo observala.
    #   Es problema del entorno donde ejecuto python.
    # if iplot:
    # show(block=True)
    return x, u1


# N = []
N = [50]
if v.__len__() != 0:
    print("Ec Ondas / Dirichlet / Explicito")
    print("-------------------------")

    for i in N:
        # ! CAMBIAR
        x, u = ondasExplicito(a, b, T, i, u0, v0, alpha, beta, c, 0)

        # figure()
        # title(f"V={i}")
        # plot(x, u, label="Aproximada")
        # plot(x, u0(x), label="Condicion inicial")
        # legend()
        # plot()

        print("-------------------------")

print("----------- Caso 4  --------------")


# Datos caso 1 y 2
def u0(x):
    return exp(-(x - 5) * (x - 5))


def v0(x):
    # Multiplicamos por x para que numpy pueda vectorizar la funcion
    return 0 * x


a = 0
b = 10
c = 1
T = 10
N = 50
alpha = 0
g = 0


def ondasExplicitoFantasma(a, b, T, N, ci, vi, alpha, g, c, iplot):
    dx = (b - a) / (N + 1)
    x = linspace(a, b, N + 2)

    if iplot:
        figure()

    dt = dx / c
    print(f"dx: {round(dx, 5)}, dt: {round(dt, 5)}")
    t = 0

    u0 = ci(x)
    u1 = u0 + dt * vi(x)
    u2 = zeros(N + 2)

    coef = c * c * dt * dt / (dx * dx)

    while t < T:
        dt_actual = min([dt, T - t])
        t = t + dt_actual

        if dt != dt_actual:
            dt = dt_actual
            coef = c * c * dt * dt / (dx * dx)

        u2[1 : N + 1] = (
            2 * (1 - coef) * u1[1 : N + 1]
            + coef * u1[0:N]
            + coef * u1[2 : N + 2]
            - u0[1 : N + 1]
        )

        u2[0] = alpha
        u2[N + 1] = (
            2 * (1 - coef) * u1[N + 1]
            + 2 * coef * u1[N]
            + 2 * coef * g * dx
            - u0[N + 1]
        )

        if iplot == 1:
            clf()
            plot(x, u1, "*-", label=f"N: {N}")
            legend()

            # Si no ajustamos los ejes no se aprecia bien el fenomeno
            axis([a, b, -1, 1])

            title(f"Tiempo: {t:.2f} | Caso 4")
            pause(0.1)

        u0 = u1.copy()
        u1 = u2.copy()

    # if iplot:
    # show(block=True)
    return x, u1


# N = []
N = [50]
if v.__len__() != 0:
    print("Ec Ondas / Dirichlet / Explicito + Nodo Fantasma")
    print("-------------------------")

    for i in N:
        x, u = ondasExplicitoFantasma(a, b, T, i, u0, v0, alpha, g, c, 1)

        # figure()
        # title(f"V={i}")
        # plot(x, u, label="Aproximada")
        # plot(x, u0(x), label="Condicion inicial")
        # legend()
        # plot()

        print("-------------------------")


# Al cambiar la condicion de frontera de dirichlet por neumann ya la funcion no se tiene porque anular en x=b.
#   Al fijar la derivada nula en vez de la frontera, se observa como cambia la reflexion de la solucion respecto al caso de condicion dirichlet,
#       donde se reflejaba la solucion respecto el eje horizontal.
#   Ademas, se cuando la "ola" llega al lado derecho aumenta el tiempo en el lado derecho como se crea una pendiente elevada en la solucion y entonces rebota.
