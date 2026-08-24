"""
EDP no lineal 1D - punto fijo y Newton
Problema -nu u'' - u2 = f con Dirichlet. Iterativos: punto fijo clasico,
matriz dependiente y Newton (J = nu/dx2 D - 2 diag(u)). Extension a
evolucion u_t - nu u_xx = f + u2 con Euler implicito.
Ver docs/enunciados_resumidos.md#01_edp_1d
"""

import time

from matplotlib.pyplot import *  # type: ignore
from numpy import *  # type: ignore
from scipy.linalg import cho_factor, cho_solve, lu_factor, lu_solve
from scipy.sparse import identity, lil_matrix
from scipy.sparse.linalg import splu


style.use("dark_background")
eps = 10e-7


# Metodo punto fijo
# Contorno por penalizacion
def nolineal_v0(x0, xf, N, v, ua, ub, f, eps, maxIter, exacta, dibujar=False):
    t1 = time.time()
    K = 10e20

    N = int(N)
    x0 = float(x0)
    xf = float(xf)
    v = float(v)
    ua = float(ua)
    ub = float(ub)

    dx = (xf - x0) / float(N)
    dx2 = dx * dx

    x = linspace(x0, xf, N + 1)
    u = zeros_like(x)
    D = lil_matrix((N + 1, N + 1), dtype="float64")

    D.setdiag(2.0 * ones(N + 1), 0)
    D.setdiag(-1.0 * ones(N), 1)
    D.setdiag(-1.0 * ones(N), -1)

    D = D.tocsc()
    A = (v / dx2) * D

    # Condiccion contorno dirichlet por penalizacion
    A[0, 0] = K
    # A[0, 1] = 0

    A[-1, -1] = K
    # A[N, N - 1] = 0

    LU = splu(A)

    iter = 0

    if dibujar:
        figure()

    while iter < maxIter:
        # Puede causar overflow or underflow
        b = f(x) + u * u
        b[0] = K * ua
        b[-1] = K * ub

        usol = LU.solve(b)

        error = max(abs(u - usol))
        # print("Error cometido:", format(error))

        u = usol

        if iter % 25 == 0 and dibujar:
            clf()
            plot(x, u, "b", x, exacta(x), "r")
            xlabel("x values")

            title(f"Aproximacion con N={N}")
            legend([f"Aproximacion en iter={round(iter, 2)}", "Exacta"])
            draw()
            pause(0.01)

        if error < eps:
            if dibujar:
                clf()
                plot(x, u, "b", x, exacta(x), "r")
                xlabel("x values")

                title(f"Aproximacion con N={N}")
                legend([f"Aproximacion en iter={round(iter, 2)}", "Exacta"])

            print(f"Break en la iteraccion {iter} por convergencia")
            break

        iter += 1

    if dibujar:
        show()

    error = max(abs(u - exacta(x)))

    print(
        f"Finalizacion de la aproximacion N={N} in iter={iter} y con error final err={round(error, 8)}"
    )

    tf = time.time()
    print("Tiempo de ejecucion:", format(tf - t1))

    return error


# Metodo punto fijo - 2 version
# Ahora aproximamos u^2 por u^l * u^(l+1)
def nolineal_v1(x0, xf, N, v, ua, ub, f, eps, maxIter, exacta, dibujar=False):
    t1 = time.time()

    N = int(N)
    x0 = float(x0)
    xf = float(xf)
    v = float(v)
    ua = float(ua)
    ub = float(ub)

    dx = (xf - x0) / float(N)
    dx2 = dx * dx

    # Realizamos particion del intervalo
    x = linspace(x0, xf, N + 1)
    u = zeros(x.shape)
    D = lil_matrix((N + 1, N + 1), dtype="float64")

    D.setdiag(2.0 * ones(N + 1), 0)
    D.setdiag(-1.0 * ones(N), 1)
    D.setdiag(-1.0 * ones(N), -1)

    D = D.tocsc()
    A = (v / dx2) * D

    error = None
    iter = 0

    if dibujar:
        figure()

    while iter < maxIter:
        C = lil_matrix((N + 1, N + 1), dtype="float64")
        C.setdiag(u, 0)
        # No provoca overflow ya que no estamos multiplicando la solucion con ella misma, simplemente estamos resolviendo un sistema
        E = A - C

        # Condiccion contorno dirichlet
        # Equivalente a utilizar metodo de penalizacion (como caso anterior)
        E[0, 0] = 1
        E[0, 1] = 0

        E[N, N] = 1
        E[N, N - 1] = 0

        LU = splu(E)

        b = f(x)
        b[0] = ua
        b[N] = ub

        usol = LU.solve(b)

        error = max(abs(usol - u))

        u = usol

        if iter % 25 == 0 and dibujar:
            clf()
            plot(x, u, "b", x, exacta(x), "r")
            xlabel("x values")

            title(f"Aproximacion con N={N}")
            legend([f"Aproximacion en iter={round(iter, 2)}", "Exacta"])
            pause(0.01)
            # show()

        if error < eps:
            if dibujar:
                clf()
                plot(x, u, "b", x, exacta(x), "r")
                xlabel("x values")

                title(f"Aproximacion con N={N}")
                legend([f"Aproximacion en iter={round(iter, 2)}", "Exacta"])

            print(f"Break en la iteraccion {iter} por convergencia")
            break

        iter += 1

    if dibujar:
        show()

    error = max(abs(u - exacta(x)))
    print(
        f"Finalizacion de la aproximacion N={N} in iter={iter} y con error final err={round(error, 8)}"
    )

    tf = time.time()
    print("Tiempo de ejecucion:", format(tf - t1))

    return error


def newton(x0, xf, N, v, ua, ub, f, eps, maxIter, exacta, dibujar=False):
    t1 = time.time()
    K = 10e20

    # Saneamos inputs
    N = int(N)
    x0 = float(x0)
    xf = float(xf)
    v = float(v)
    ua = float(ua)
    ub = float(ub)

    dx = (xf - x0) / float(N)
    dx2 = dx * dx

    # Realizamos particion del intervalo
    x = linspace(x0, xf, N + 1)
    u = zeros(x.shape)
    D = lil_matrix((N + 1, N + 1), dtype="float64")

    D.setdiag(2.0 * ones(N + 1), 0)
    D.setdiag(-1.0 * ones(N), 1)
    D.setdiag(-1.0 * ones(N), -1)

    D = D.tocsc()
    r = v / dx2

    f_valores = f(x)
    F = f_valores

    # Condiccion contorno dirichlet
    # Ahora mismo esta implementado penalizacion

    # Alternativamente se podria hacer de la siguiente forma
    # Los valores provienen de despejar en la funcion que da el metodo de newton para j=0,N
    #   antes de aplicar el mismo, por ende en estas dos filas, estamos usando una aplicacion distinta,
    #   una que "ancla" el valor en el contorno del problema
    b_frontera = zeros_like(x)
    b_frontera[0] = r * ua
    b_frontera[-1] = r * ub
    # F = f_valores + b_frontera

    error = None
    iter = 0

    if dibujar:
        figure()

    while iter < maxIter:
        C = lil_matrix((N + 1, N + 1), dtype="float64")
        C.setdiag(u, 0)

        # Jacobiano del metodo iterativo
        J = v / dx2 * D - 2 * C

        # Imponemos condiccion contorno (dirichlet) por medio del uso de penalizacion
        J.tolil()
        J[0, 0] += K
        J[-1, -1] += K
        J.tocsc()

        LU = splu(J)

        # Termino independiente
        b = F - r * D * u + u * u
        b[0] += K * (ua - u[0])
        b[-1] += K * (ub - u[N])

        # b[0] = ua - u[0]
        # b[N] = ub - u[N]

        # Como el termino a resolver es x_l+1 - x_l, calculamos el incremento y despues se lo sumamos
        #   a la solucion anterior para obtener la nueva
        incremento = LU.solve(b)
        usol = u + incremento

        error = max(abs(incremento))

        u = usol

        if iter % 25 == 0 and dibujar:
            clf()
            plot(x, u, "b", x, exacta(x), "r")
            xlabel("x values")

            title(f"Aproximacion Netwon con N={N}")
            legend([f"Aproximacion en iter={round(iter, 2)}", "Exacta"])
            pause(0.01)
            # show()

        if error < eps:
            if dibujar:
                clf()
                plot(x, u, "b", x, exacta(x), "r")
                xlabel("x values")

                title(f"Aproximacion con N={N}")
                legend([f"Aproximacion en iter={round(iter, 2)}", "Exacta"])

            print(f"Break en la iteraccion {iter} por convergencia")
            break

        iter += 1

    if dibujar:
        show()

    error = max(abs(u - exacta(x)))

    print(
        f"Finalizacion de la aproximacion N={N} in iter={iter} y con error final err={round(error, 8)}"
    )

    tf = time.time()
    print("Tiempo de ejecucion:", format(tf - t1))

    return error


# --------------------- CASO 4 (TEMPORAL) ----------------#


def caso4_v1(
    x0,
    xf,
    Nx,
    t0,
    tf,
    Nt,
    alfa,
    ua,
    ub,
    u0,
    f,
    eps=10e-7,
    maxIter=500,
    exacta=None,
    dibujar=False,
):
    time.time()

    Nx = int(Nx)
    Nt = int(Nt)
    x0 = float(x0)
    t0 = float(t0)
    xf = float(xf)
    tf = float(tf)
    alfa = float(alfa)
    ua = float(ua)
    ub = float(ub)

    # Traslaccion del tiempo en el instante inicial al origen
    if t0 != 0:
        tf -= t0
        t0 = 0
    t = float(t0)

    dx = (xf - x0) / float(Nx)
    dx2 = dx * dx

    # Se estudia el mas grande
    dt = (tf - t0) / float(Nt)

    print(f"dx2={round(dx2, 7)}, dt={round(dt, 7)} \n")

    x = linspace(x0, xf, Nx + 1)
    u = u0(x)

    D = lil_matrix((Nx + 1, Nx + 1), dtype="float64")
    Id = identity(Nx + 1, dtype="float64", format="csc")

    D.setdiag(2.0 * ones(Nx + 1), 0)
    D.setdiag(-1.0 * ones(Nx), 1)
    D.setdiag(-1.0 * ones(Nx), -1)
    D = D.tocsc()

    r = alfa * dt / dx2
    A = Id + r * D

    # Condiccion de contorno
    A[0, 0] = 1
    A[0, 1] = 0
    A[-1, -1] = 1
    A[-1, -2] = 0
    A = A.tocsc()

    # La matriz A no es simetrica
    LU = splu(A)

    # Contador para ir pintando la funcion cada un numero dados de iteracciones
    i = 0

    #  Inicializamos mathplotlib
    if dibujar:
        figure()

    # Inicializacion calculo error
    error = array(0.0, dtype="float64")

    # Para que la primera iteraccion sea con t=0
    t = -dt
    while t < tf - dt / 2:
        t += dt
        iter = 0

        # u_old = u.copy()

        # Construccion del vector b
        b = f(x) * dt

        while iter < maxIter:
            #! ESTA MAL EN LOS APUNTES, QUIERO USAR METODO PROGRESIVO, las dif.finitas me da un termino u_i^n que me he olvidado
            b += u**2 * dt
            b[0] = ua
            b[Nx] = ub

            usol = LU.solve(b)

            # Error en iteracciones del metodo de punto fijo
            errorIter = max(abs(u - usol))
            # print("Error cometido:", format(errorIter))

            u = usol

            if iter % 100 == 0 and dibujar:
                clf()
                plot(x, u, "b", x, exacta(x), "r")
                xlabel("x values")

                title(f"Aproximacion con N={Nx}")
                legend([f"Aproximacion en iter={round(iter, 2)}", "Exacta"])
                pause(0.01)

            if errorIter < eps:
                if dibujar:
                    clf()
                    plot(x, u, "b", x, exacta(x), "r")
                    xlabel("x values")

                    title(f"Aproximacion con N={Nx}")
                    legend([f"Aproximacion en iter={round(iter, 2)}, t={t}", "Exacta"])

                break

            iter += 1

        # Errores aproximacion
        # err = max(abs(u - exacta(x, t))) if exacta != None else 0
        err = max(abs(u - exacta(x))) if exacta is not None else 0
        # print("Error espacial cometido:",format(err))
        error = append(error, err)

        if i % 200 == 0 and dibujar and exacta is not None:
            plot(x, u, "b", x, exacta(x), "r")
            xlabel("x values")

            title(f"Aproximacion con N={Nx}")
            legend([f"Aproximacion en t={round(t, 2)}", f"Exacta en t={round(t, 2)}"])
            pause(0.1)
            draw()

        i += 1

    tf = time.time()
    if dibujar:
        show()

    return max(error)


def caso4_newton(
    x0,
    xf,
    Nx,
    t0,
    tf,
    Nt,
    v,
    ua,
    ub,
    u0,
    f,
    eps=10e-7,
    maxIter=500,
    exacta=None,
    dibujar=False,
):
    Nx = int(Nx)
    Nt = int(Nt)
    x0 = float(x0)
    t0 = float(t0)
    xf = float(xf)
    tf = float(tf)

    # Traslaccion del tiempo en el instante inicial al origen
    if t0 != 0:
        tf -= t0
        t0 = 0
    t = float(t0)

    # Se estudia el orden de convergencia del mas grande
    dx = (xf - x0) / float(Nx)
    dt = (tf - t0) / float(Nt)
    dx2 = dx * dx

    print(f"dx2={round(dx2, 7)}, dt={round(dt, 7)} \n")

    v = float(v)
    ua = float(ua)
    ub = float(ub)

    x = linspace(x0, xf, Nx + 1)
    u_n = u0(x)
    u = u_n.copy()

    D = lil_matrix((Nx + 1, Nx + 1), dtype="float64")
    Id = identity(Nx + 1, dtype="float64", format="csc")

    # Construimos la matriz D
    D.setdiag(2.0 * ones(Nx + 1), 0)
    D.setdiag(-1.0 * ones(Nx), 1)
    D.setdiag(-1.0 * ones(Nx), -1)

    D = D.tocsc()
    r = v * dt / dx2

    # Imponemos condicciones de extremo de tipo dirichlet directamente
    f_valores = f(x)

    # Contador para ir pintando la funcion cada un numero dados de iteracciones
    i = 0

    if dibujar:
        figure()

    error = []

    # Para que la primera iteraccion sea con t=0
    t = -dt
    while t < tf - dt / 2:
        t += dt
        iter = 0

        u = u_n.copy()

        while iter < maxIter:
            C = lil_matrix((Nx + 1, Nx + 1), dtype="float64")
            C.setdiag(u, 0)

            # Jacobiano
            J = Id + r * D - 2 * C * dt

            # Condicciones de extremo
            J = J.tolil()
            J[0, 1] = 0.0
            J[0, 0] = 1.0
            J[-1, -2] = 0.0
            J[-1, -1] = 1.0
            J = J.tocsc()

            LU = splu(J)

            # Construccion del resto
            G_u = (Id + r * D) * u - dt * (u * u) - (u_n + dt * f_valores)
            b = -G_u

            # Esta es la condiccion del extremo en este caso porque estamos calculando el incremento
            b[0] = ua - u[0]
            b[Nx] = ub - u[Nx]

            incremento = LU.solve(b)
            usol = u + incremento

            # error = max(abs(usol - u))
            errorIter = max(abs(incremento))

            u = usol

            if iter % 100 == 0 and dibujar:
                clf()
                plot(x, u, "b", x, exacta(x), "r")
                xlabel("x values")

                title(f"Aproximacion con N={Nx}")
                legend([f"Aproximacion en iter={round(iter, 2)}", "Exacta"])
                pause(0.01)

            if errorIter < eps:
                if dibujar:
                    clf()
                    plot(x, u, "b", x, exacta(x), "r")
                    xlabel("x values")

                    title(f"Aproximacion con N={Nx}")
                    legend([f"Aproximacion en iter={round(iter, 2)}, t={t}", "Exacta"])

                break

            iter += 1

        # Errores aproximacion
        # err = max(abs(u - exacta(x, t))) if exacta != None else 0
        err = max(abs(u - exacta(x))) if exacta is not None else 0
        error = append(error, err)

        if i % 100 == 0 and dibujar and exacta is not None:
            plot(x, u, "b", x, exacta(x), "r")
            xlabel("x values")

            title(f"Aproximacion con N={Nx}")
            legend([f"Aproximacion en t={round(t, 2)}", f"Exacta en t={round(t, 2)}"])
            pause(0.1)
            draw()

        i += 1
        u_n = u.copy()

    if dibujar:
        show()

    print(f"Error final newton es {max(error)}")
    return max(error)


########################################################################
########################################################################
########################################################################
########################################################################


def f(x):
    return cos(x) * (1 - cos(x))


def u(x):
    return cos(x)


def u0(x):
    return 1 - 2 * x / pi


def f_2(x):
    return -sin(pi * x)


def u_exact(x):
    return x**2 + x + 1


def f_manufactured(x):
    u_val = u_exact(x)
    u_xx_val = 2
    # v = 0.1
    return -0.1 * u_xx_val - u_val**2


error = 0
nuevo_error = 0

calculo = [100, 200, 400, 800]
# calculo = [50, 100, 200]

# Controla el caso a realizar
caso = 4

# fmt: off
if calculo.__len__() != 0:
    print(f"Caso {caso}")

    for i in calculo:
        # ---------- CASO 1 ---------------- #
        if caso == 1:
                nuevo_error = nolineal_v0(0, pi, i, 1, 1, -1, f, eps, 500, u, dibujar=False)

                # Causa overflow
                # nuevo_error = nolineal_v0(0, 1.0, i, 0.1, 1.0, 3.0, f_manufactured, eps, 500, u_exact, dibujar=False)

        # ---------- CASO 2 ---------------- #
        if caso == 2:
                nuevo_error = nolineal_v1(0, pi, i, 1, 1, -1, f, eps, 500, u, dibujar=False)
                # nuevo_errorc = nolineal_v1(0, 1.0, i, 0.1, 1.0, 3.0, f_manufactured, eps, 500, u_exact, dibujar=False)


        # ---------- CASO 3 ---------------- #
        if caso == 3:
            nuevo_error = newton(0, pi, i, 1, 1, -1, f, eps, 500, u, dibujar=False)

        # ---------- CASO 4 ---------------- #
        if caso == 4:
            # Error temporal
            # nuevo_error = caso4_v1(0, pi, 200, 0, 4, i, 1, 1, -1, u0, f, eps, 500, u, False)
            
            # Error espacial
            # nuevo_error = caso4_v1(0, pi, i, 0, 4, 1000, 1, 1, -1, u0, f, eps, 500, u, False)
            
            nuevo_error = caso4_newton(0, pi, i, 0, 1, 1000000, 1, 1, -1, u0, f, eps, 500, u, False)
            
        print("-------------------------")

        if error == 0:
            print("ERROR", nuevo_error)
        elif nuevo_error == 0:
            print("ZZZ")
        else:
            print("Cociente de errores (Orden): ", (error / nuevo_error))

        error = nuevo_error
        print("-------------------------")
# fmt: on
