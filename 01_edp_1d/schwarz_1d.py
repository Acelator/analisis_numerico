"""
Descomposicion de dominios Schwarz 1D
Problema u - nu u'' = f en (0,L). Metodo alternante con solape l,
variantes Dirichlet-Dirichlet, Dirichlet-Neumann y N subdominios.
Analisis de convergencia vs. solape.
Ver docs/enunciados_resumidos.md#01_edp_1d
"""

# El error teorico me da una cota inferior para el error en el mejor de los casos
#   Por eso en ocasiones el error que me encuentro es mayor, es un minimo teorico.

import time


from numpy import *
from scipy.sparse.linalg import splu
from scipy.sparse import lil_matrix, identity
from scipy.linalg import lu_factor, lu_solve, cho_factor, cho_solve
from matplotlib.pyplot import *

# Modo oscuro para mathplotlib
style.use("dark_background")

# La tasa de convergencia del metodo de Schwarz clasico se comporta aproximadamente como rho ~ (1 - c * (l*h / L_sub)),
#   donde L_sub es el tamano del subdominio. Un l mas grande hace que rho sea mas pequeno, lo que implica una convergencia mas rapida.

# El l optimo es el valor que corresponde al minimo en la grafica de tiempo de ejecucion vs. l.
# En la seccion, se busca un "punto dulce" donde aumentar mas l ya no produce una reduccion significativa en
#   el numero de iteraciones que justifique el aumento en el costo computacional por iteracion.
# Un buen punto de partida suele ser un solapamiento de entre el 5% y el 15% del tamano del subdominio.


# A menor l mas iteracciones hacen falta (se propaga menos informacion en la parte del solapamiento).
# (piensa si l==total entonces realmente estariamos resolviendo dos veces el problema exacto como en la primera seccion
#       por lo que no haria falta ninguna iteraccion)
def schwarz_v1(x0, xf, N, v, ua, ub, f, eps, maxIter, dibujar=False, exacta=None):
    t1 = time.time()

    N = int(N)
    x0 = float(x0)
    xf = float(xf)

    dx = (xf - x0) / float(N)
    dx2 = dx * dx

    v = float(v)
    ua = float(ua)
    ub = float(ub)

    # Tamano de los subintervalos que intervienen
    p1 = int(2.6 * ceil(N / 5))
    # p1 = int(N)
    # p2 = 0

    # Mayor l -> Menos iteracciones necesarias para converger, pero se neceistan computar el error en un mayor numero de puntos
    #               y tambien el tiempo de computo sube para cada subintervalo ya que es mas grande
    # Menor l -> Menos puntos que se tengan que calcular en ambos subdominios, pero aumenta el numero de iteracciones necesarias
    #               para convergencia
    # En la seccion un solapamiento muy pequeno proporciona mucha mejora en la convergencia sin aumentar mucho el costo por iteraccion.
    # l = int(ceil(N / 5))  # noqa: E741
    l = int(ceil(N / 40))  # noqa: E741
    if not l % 2 == 0:
        l += 1  # noqa: E741

    # Tenemos la ligadura -> p1 + p2 = N + l
    p2 = int(N + l - p1)

    print(f"p1={p1}, p2={p2} & l={l}")

    # Como hemos tomado siempre l par los puntos de solapamiento
    #   en este caso que solo tenemos 2 subintervalos seran desde x_(c -l/2 +1) a x_(c + l/2)

    x = linspace(x0, xf, N + 1)

    Ip1 = x[: p1 + 1]
    Ip2 = x[p1 - l :]

    u1 = zeros_like(Ip1)
    u2 = zeros_like(Ip2)

    D1 = lil_matrix((p1 + 1, p1 + 1), dtype="float64")
    D2 = lil_matrix((p2 + 1, p2 + 1), dtype="float64")

    Id1 = identity(p1 + 1, dtype="float64", format="csc")
    Id2 = identity(p2 + 1, dtype="float64", format="csc")

    D1.setdiag(2.0 * ones(p1 + 1), 0)
    D1.setdiag(-1.0 * ones(p1), -1)
    D1.setdiag(-1.0 * ones(p1), 1)

    D2.setdiag(2.0 * ones(p2 + 1), 0)
    D2.setdiag(-1.0 * ones(p2), -1)
    D2.setdiag(-1.0 * ones(p2), 1)

    D1 = D1.tocsc()
    A1 = Id1 + v / dx2 * D1

    D2 = D2.tocsc()
    A2 = Id2 + v / dx2 * D2

    # Condiccion contorno dirichlet en ambos lados
    A1[0, 0] = 1
    A1[0, 1] = 0
    A1[p1, p1] = 1
    A1[p1, p1 - 1] = 0

    # Condiccion contorno dirichlet en ambos lados
    A2[0, 0] = 1
    A2[0, 1] = 0
    A2[p2, p2] = 1
    A2[p2, p2 - 1] = 0

    LU1 = splu(A1)
    LU2 = splu(A2)

    b1 = f(Ip1)
    b2 = f(Ip2)

    iter = 0
    error = 0.0

    if dibujar:
        figure()

    while iter < maxIter:
        b1[0] = ua
        b2[p2] = ub

        # Como valor inicial damos una aproximacion lineal de los valores de contorno.
        if iter == 0:
            b1[p1] = (ub - ua) / (xf - x0)
            b2[0] = (ub - ua) / (xf - x0)
        else:
            b1[p1] = u2[l]
            b2[0] = u1[p1 - l]

        # print(f"Contorno: u2 en 0 -> {u1[p1 - l]}, u1 en p1 -> {u2[l]}")

        usol1 = LU1.solve(b1)
        usol2 = LU2.solve(b2)

        error = max(abs(usol1[p1 - l :] - usol2[: l + 1]))
        # print("Error cometido:", format(error))
        # print("-------------------")

        u1 = usol1
        u2 = usol2

        if iter % 25 == 0 and dibujar:
            # print(f"Iter {iter} para N={N}")
            clf()

            # Use a 2x2 grid: top row contains two side-by-side plots, bottom row a single plot spanning both columns
            # params -> size - location
            ax1 = subplot2grid((2, 2), (0, 0))
            ax2 = subplot2grid((2, 2), (0, 1))
            ax3 = subplot2grid((2, 2), (1, 0), colspan=2)

            ax1.plot(Ip1, u1, "b")
            ax2.plot(Ip2, u2, "r")

            ax3.plot(Ip1, u1, "b", Ip2, u2, "r")

            # make both top subplots use the same axis limits so their scales match
            x_min = min(Ip1[0], Ip2[0])
            x_max = max(Ip1[-1], Ip2[-1])
            y_min = min(u1.min(), u2.min())
            y_max = max(u1.max(), u2.max())

            ax1.set_xlim(x_min, x_max)
            ax2.set_xlim(x_min, x_max)
            ax1.set_ylim(y_min, y_max)
            ax2.set_ylim(y_min, y_max)

            # Establece el titulo global de la figura
            suptitle(f"Aproximacion con N={N} ~ PUNTO FIJO")

            ax3.legend([f"Aproximacion en iter={round(iter, 2)}", "Exacta"])

            # Rectangulo que encuadra a la grafica conjunta
            tight_layout(rect=[0, 0, 1, 0.95])
            pause(0.01)
            show()

        if error < eps:
            if dibujar:
                clf()

                print(f"u2 en {l} -> {u2[l - 1]}")
                print(f"u1 en {p1} -> {u1[p1 - 1]}")

                plot(Ip1, u1, "b")
                plot(x[p1:], u2[l:], "r")
                xlabel("x values")

                title(f"Aproximacion con N={N}")

            print(f"Break en la iteraccion {iter} por convergencia")
            break

        iter += 1

    if dibujar:
        show()

    print(
        f"Finalizacion de la aproximacion N={N} in iter={iter} y con error final err={round(error, 10)}"
    )

    tf = time.time()
    print("Tiempo de ejecucion:", format(tf - t1))

    return error


def f(x):
    return 5 * exp(-pow((x - 1 / 2), 2))


# Comprobacion
error = 0
eps = 10e-10

# calculo = [200, 400, 800]
calculo = []

if calculo.__len__() != 0:
    print("Metodo schwarz")
    for i in calculo:
        print("----------------------")
        nuevo_error = schwarz_v1(0, 1.0, i, 2.0, 0.0, 0.0, f, eps, 500)
        print("-------------------------")

        if error == 0:
            print("ERROR", nuevo_error)
        else:
            print("Cociente de error", (error / nuevo_error))

        error = nuevo_error

    pause(50000)


###################################################################
###################################################################
###################################################################


# Condiccion contorno neumann en x=b
def schwarz_v2(x0, xf, N, v, ua, g, f, eps, maxIter, exacta=None, dibujar=True):
    t1 = time.time()

    N = int(N)
    x0 = float(x0)
    xf = float(xf)

    dx = (xf - x0) / float(N)
    dx2 = dx * dx

    v = float(v)
    ua = float(ua)
    g = float(g)

    # Tamano de los subintervalos que intervienen
    p1 = int(3 * ceil(N / 5))
    # p2 = 0

    l = int(ceil(N / 5))  # noqa: E741
    if not l % 2 == 0:
        l += 1  # noqa: E741

    # Tenemos la ligadura -> p1 + p2 = N + l
    p2 = int(N + l - p1)

    print(f"p1={p1}, p2={p2} & l={l}")

    # Como hemos tomado siempre l par los puntos de solapamiento
    #   en este caso que solo tenemos 2 subintervalos seran desde x_(c -l/2 +1) a x_(c + l/2)

    # Realizamos particion del intervalo
    x = linspace(x0, xf, N + 1)

    Ip1 = x[: p1 + 1]
    Ip2 = x[p1 - l :]

    u1 = zeros_like(Ip1)
    u2 = zeros_like(Ip2)

    D1 = lil_matrix((p1 + 1, p1 + 1), dtype="float64")
    D2 = lil_matrix((p2 + 1, p2 + 1), dtype="float64")

    Id1 = identity(p1 + 1, dtype="float64", format="csc")
    Id2 = identity(p2 + 1, dtype="float64", format="csc")

    D1.setdiag(2.0 * ones(p1 + 1), 0)
    D1.setdiag(-1.0 * ones(p1), -1)
    D1.setdiag(-1.0 * ones(p1), 1)

    D2.setdiag(2.0 * ones(p2 + 1), 0)
    D2.setdiag(-1.0 * ones(p2), -1)
    D2.setdiag(-1.0 * ones(p2), 1)

    D1 = D1.tocsc()
    A1 = Id1 + v / dx2 * D1

    D2 = D2.tocsc()
    A2 = Id2 + v / dx2 * D2

    # Condiccion contorno dirichlet en x=a
    A1[0, 0] = 1
    A1[0, 1] = 0
    A1[p1, p1] = 1
    A1[p1, p1 - 1] = 0

    # Condiccion tipo neumann en x=b
    A2[0, 0] = 1
    A2[0, 1] = 0
    A2[-1, -1] = 1 + 2 * v / dx2
    A2[-1, -2] = -2 * v / dx2

    LU1 = splu(A1)
    LU2 = splu(A2)

    b1 = f(Ip1)
    b2 = f(Ip2)

    iter = 0
    error = 0.0

    if dibujar:
        figure()

    while iter < maxIter:
        b1[0] = ua
        b1[p1] = u2[l]

        b2[0] = u1[p1 - l + 1]
        b2[-1] += 2 * g * v / dx

        usol1 = LU1.solve(b1)
        usol2 = LU2.solve(b2)

        error = max(abs(usol1[p1 - l :] - usol2[: l + 1]))
        # print("Error cometido:", format(error))

        u1 = usol1
        u2 = usol2

        if iter % 25 == 0 and dibujar:
            clf()

            # Mismo algoritmo para pintar que el caso anterior
            # Use a 2x2 grid: top row contains two side-by-side plots, bottom row a single plot spanning both columns
            # params -> size - location
            ax1 = subplot2grid((2, 2), (0, 0))
            ax2 = subplot2grid((2, 2), (0, 1))
            ax3 = subplot2grid((2, 2), (1, 0), colspan=2)

            ax1.plot(Ip1, u1, "b")
            ax2.plot(Ip2, u2, "r")

            ax3.plot(Ip1, u1, "b", Ip2, u2, "r")

            # make both top subplots use the same axis limits so their scales match
            x_min = min(Ip1[0], Ip2[0])
            x_max = max(Ip1[-1], Ip2[-1])
            y_min = min(u1.min(), u2.min())
            y_max = max(u1.max(), u2.max())

            ax1.set_xlim(x_min, x_max)
            ax2.set_xlim(x_min, x_max)
            ax1.set_ylim(y_min, y_max)
            ax2.set_ylim(y_min, y_max)

            suptitle(f"Aproximacion con N={N} ~ PUNTO FIJO")

            ax3.legend([f"Aproximacion en iter={round(iter, 2)}", "Exacta"])

            tight_layout(rect=[0, 0, 1, 0.95])
            pause(0.01)
            show()

        if error < eps:
            if dibujar:
                clf()

                print(f"u2 en {l} -> {u2[l - 1]}, u1 en {p1} -> {u1[p1 - 1]}")

                plot(Ip1, u1, "b")
                plot(x[p1:], u2[l:], "r")
                xlabel("x values")

                title(f"Aproximacion con N={N}")

            print(f"Break en la iteraccion {iter} por convergencia")
            break

        print("-------------------")
        iter += 1

    if dibujar:
        show()

    print(
        f"Finalizacion de la aproximacion N={N} in iter={iter} y con error final err={round(error, 8)}"
    )

    tf = time.time()
    print("Tiempo de ejecucion:", format(tf - t1))

    return error


# Comprobacion
error = 0
eps = 10e-10

# calculo = [100, 200, 400]
calculo = []

if calculo.__len__() != 0:
    print("Schwarz ~ Extremo neumann")
    for i in calculo:
        print("----------------------")
        nuevo_error = schwarz_v2(0, 1.0, i, 2.0, 0.0, 0.0, f, eps, 500)
        print("-------------------------")

        if error == 0:
            print("ERROR", nuevo_error)
        else:
            print("Cociente de error", (error / nuevo_error))

        error = nuevo_error

    pause(50000)


####################################################
####################################################
####################################################
####################################################
####################################################
####################################################
####################################################
# Puede ocurrir que si tomamos un numero eleveado de particiones con un N pequeno no exista suficiente informacion
#   para resolver el sistema correctamente llegando a soluciones extranas
def schwarz_arbitrario(
    x0, xf, N, v, ua, ub, f, particiones, eps, maxIter, exacta=None, dibujar=True
):
    if particiones == 1:
        raise ValueError("Utiliza directamente los algoritmo de la primera seccion")

    t1 = time.time()

    N = int(N)
    x0 = float(x0)
    xf = float(xf)

    dx = (xf - x0) / float(N)
    dx2 = dx * dx

    v = float(v)
    ua = float(ua)
    ub = float(ub)

    # Tamano de los subintervalos que intervienen
    #   Para poder tomar tantas particiones como queramos y facilitarnos la vida, tomamos una particion uniforme del intervalo
    size = int(ceil(N / (particiones)))

    if size < 10:
        raise ValueError(
            "Hay demasiados pocos puntos para realizar una aproximacion efectiva"
        )

    l = size // (10 * particiones)  # noqa: E741
    if not l % 2 == 0:
        l += 1  # noqa: E741
    l = int(l)  # noqa: E741

    print(f"dx2={round(dx2, 7)}")
    print(f"noparticiones={particiones}, tamano={size} & l={l}")

    x = linspace(x0, xf, N + 1)
    u = zeros_like(x)

    # Logica para un numero arbitrario de subintervalos
    inicio = []
    final = []

    # Primer subintervalo
    inicio.append(0)
    final.append(size)

    for i in range(1, particiones - 1):
        inicio.append(final[-1] - l // 2)
        final.append(inicio[-1] + size + l // 2)

    # Ultimo subintervalo
    inicio.append(final[-1] - l // 2)
    final.append(N)

    # Crear matriz del sistema
    p = size + l // 2 - 1
    D = lil_matrix((p + 1, p + 1), dtype="float64")

    Id = identity(p + 1, dtype="float64", format="csc")

    D.setdiag(2.0 * ones(p + 1), 0)
    D.setdiag(-1.0 * ones(p), -1)
    D.setdiag(-1.0 * ones(p), 1)

    D = D.tocsc()
    A = Id + v / dx2 * D

    # Estabelecer condicciones de contorno

    # Tomamos una matriz distinta para el subintervalo incial y final ya que estos no podemos garantizar
    #   que tengan el mismo tamano que los subintervalos interiores.
    A_inicial = Id[:size, :size] + v / dx2 * D[:size, :size]
    last = final[-1] - inicio[-1]
    A_Final = Id[:last, :last] + v / dx2 * D[:last, :last]

    # Condiccion contorno dirichlet
    A_inicial[0, 0] = 1
    A_inicial[0, 1] = 0
    A_inicial[-1, -2] = 0
    A_inicial[-1, -1] = 1

    A_Final[0, 0] = 1
    A_Final[0, 1] = 0
    A_Final[-1, -2] = 0
    A_Final[-1, -1] = 1

    A[0, 0] = 1
    A[0, 1] = 0
    A[-1, -1] = 1
    A[-1, -2] = 0

    LU_inicial = splu(A_inicial)
    LU_final = splu(A_Final)
    LU = splu(A)

    b = f(x)

    iter = 0
    error = 0

    if dibujar:
        figure()

    # Condiccion de contorno iniciales por interpolacion lineal de datos contorno en extremos del intervalo original.
    m = (ub - ua) / (xf - x0)  # Pendiente recta uniendo extremos
    u = []
    for i in range(0, particiones):
        u.append(zeros_like(x[inicio[i] : final[i]]))
        u[i][0] = ub - m * (xf - x[inicio[i]])
        u[i][-1] = ub - m * (xf - x[final[i]])

    usol = []
    while iter < maxIter:
        # Resolvemos los sistemas y los anadimos a usol para computar despues sus errores.
        usol.append(
            LU_inicial.solve(
                concatenate(
                    (array([ua]), b[1 : final[0] - 1], array([u[1][l // 2 - 1]]))
                )
            )
        )

        for i in range(1, particiones - 1):
            usol.append(
                LU.solve(
                    concatenate(
                        (
                            array([u[i - 1][-l // 2]]),
                            b[inicio[i] + 1 : final[i] - 1],
                            array([u[i + 1][l // 2 - 1]]),
                        )
                    )
                )
            )

        usol.append(
            LU_final.solve(
                concatenate(
                    (
                        array([u[-2][-l // 2]]),
                        b[inicio[-1] + 1 : final[-1] - 1],
                        array([ub]),
                    )
                )
            )
        )

        for i in range(0, particiones):
            u[i][:] = usol[i]

        for i in range(1, particiones):
            error = max(abs(usol[i - 1][-l // 2 :] - usol[i][: l // 2]))
        # print("Error cometido:", format(error))

        if iter % 50 == 0 and dibujar:
            clf()

            # Calcular numero de filas necesarias para albegar 5 graficas por fila
            maximo_fila = 5
            num_filas = (particiones + maximo_fila - 1) // maximo_fila

            # Crear grid: filas para subintervalos + 1 fila para la grafica completa
            total_filas = num_filas + 1

            # Dibujar cada subintervalo en una pequena grafica
            for i in range(0, particiones):
                # Calcular posicion en grid (row, col)
                row = i // maximo_fila
                col = i % maximo_fila

                # Crear subplot en la posicion correcta
                ax = subplot2grid((total_filas, maximo_fila), (row, col))
                ax.plot(x[inicio[i] : final[i]], u[i], "b")
                ax.set_title(f"Particion {i + 1}")
                ax.grid(True, alpha=0.3)

            # Grafica completa
            ax_complete = subplot2grid(
                (total_filas, maximo_fila), (num_filas, 0), colspan=maximo_fila
            )

            # Representar todos los subintervalos en la grafica completa alternando colores
            color_cycle = ["b", "r", "g", "c", "m", "y"]
            for i in range(0, particiones):
                color = color_cycle[particiones % len(color_cycle)]
                ax_complete.plot(
                    x[inicio[i] : final[i]], u[i], color, label=f"Part. {i + 1}"
                )

            ax_complete.set_title(
                f"Solucion completa - Iteracion {iter} \n Error: {error}"
            )
            ax_complete.legend(loc="best", fontsize=8)
            ax_complete.grid(True, alpha=0.3)
            ax_complete.set_xlabel("x")
            ax_complete.set_ylabel("u(x)")

            suptitle(
                f"Schwarz Arbitrario: {particiones} particiones, N={N}, Iter={iter}"
            )
            tight_layout(rect=[0, 0, 1, 0.96])
            pause(0.01)
            show()

        if error < eps:
            if dibujar:
                clf()

                # Grafica unica con todas las particiones
                color_cycle = ["b", "r", "g", "c", "m", "y"]
                for i in range(0, particiones):
                    color = color_cycle[i % len(color_cycle)]
                    plot(
                        x[inicio[i] : final[i]],
                        u[i],
                        color,
                        label=f"Parte. {i + 1}",
                        linewidth=2,
                    )

                xlabel("x")
                ylabel("u(x)")
                title(
                    f"Solucion convergida - N={N}, {particiones} particiones, Iter={iter}"
                )
                legend(loc="best")

                grid(True, alpha=0.3)
                tight_layout()
                show()

            print(f"Break en la iteraccion {iter} por convergencia")
            break

        # Preparacion de la siguiente iteracion.
        usol = []
        iter += 1

    if dibujar:
        show()

    print(
        f"Finalizacion de la aproximacion N={N} in iter={iter} y con error final err={round(error, 8)}"
    )

    tf = time.time()
    print("Tiempo de ejecucion:", format(tf - t1))

    return error


# Comprobacion
error = 0
eps = 10e-7

calculo = [200, 400, 800]
# calculo = []

if calculo.__len__() != 0:
    print("Alg.Schawarz ~ Arbitrario noparticiones ")
    for i in calculo:
        print("----------------------")
        nuevo_error = schwarz_arbitrario(
            0, 1.0, i, 2.0, 0, 0, f, 3, eps, 500, dibujar=False
        )
        print("-------------------------")

        if error == 0:
            print("ERROR", nuevo_error)
        else:
            print("Cociente de error (Orden)", (error / nuevo_error))

        error = nuevo_error
