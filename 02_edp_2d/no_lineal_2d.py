"""
Parabolica 2D no lineal - reaccion-difusion
Problema u_t - Deltau + u2 = f con Dirichlet. Esquemas Picard y Newton
por paso temporal.
Ver docs/enunciados_resumidos.md#02_edp_2d
"""

import time
from numpy import *
from scipy.sparse.linalg import splu
from scipy.sparse import lil_matrix, identity
from matplotlib.pyplot import *


style.use("dark_background")


# Usa algoritmo de Pircard (no es numericamente estable si dt no es suficientemente pequeno)
def ejer5_v1(
    xi,
    xf,
    Nx,
    yi,
    yf,
    Ny,
    t0,
    tf,
    Nt,
    u0,
    u1,
    u2,
    u3,
    estado_inicial,
    fuente,
    eps,
    exacta=None,
    dibujar=True,
):
    Bn = 10e20

    Nx = int(Nx)
    Ny = int(Ny)
    Nt = int(Nt)
    xi = float(xi)
    xf = float(xf)
    yi = float(yi)
    yf = float(yf)

    dx = (xf - xi) / float(Nx)
    dx2 = dx * dx
    dy = (yf - yi) / float(Ny)
    dy2 = dy * dy
    dt = (tf - t0) / float(Nt)
    N = (Nx + 1) * (Ny + 1)

    t = float(t0)
    tf = float(tf)

    x = linspace(xi, xf, Nx + 1)
    y = linspace(yi, yf, Ny + 1)
    X, Y = meshgrid(x, y)
    U = estado_inicial(X, Y)

    A = lil_matrix((N, N), dtype="float64")
    Mx_interior = lil_matrix((Nx + 1, Nx + 1), dtype="float64")
    Mx_exterior = lil_matrix((Nx + 1, Nx + 1), dtype="float64")
    My = lil_matrix((Nx + 1, Nx + 1), dtype="float64")

    Mx_interior.setdiag(1.0 + 2.0 * (1.0 / (dx2) + 1.0 / (dy2)) * dt * ones(Nx + 1), 0)
    Mx_interior.setdiag(-1.0 / (dx2) * dt * ones(Nx), 1)
    Mx_interior.setdiag(-1.0 / (dx2) * dt * ones(Nx), -1)

    Mx_exterior.setdiag(Bn * ones(Nx + 1), 0)

    My.setdiag(-1.0 / (dy2) * dt * ones(Nx + 1), 0)

    Mx_interior[-1, -1] = Bn
    # Mx_interior[-1, :] = 0

    Mx_interior[0, 0] = Bn

    # Linea siguiente es si fuera Neumann en 3
    # Mx_interior[0, 1] *= 2

    for i in range(1, Ny):
        A[i * (Nx + 1) : (i + 1) * (Nx + 1), i * (Nx + 1) : (i + 1) * (Nx + 1)] = (
            Mx_interior
        )
        A[i * (Nx + 1) : (i + 1) * (Nx + 1), (i - 1) * (Nx + 1) : i * (Nx + 1)] = My
        A[
            i * (Nx + 1) : (i + 1) * (Nx + 1), (i + 1) * (Nx + 1) : (i + 2) * (Nx + 1)
        ] = My

    # Condiccion de contorno en bordes inferior y superior
    A[0 : (Nx + 1), 0 : (Nx + 1)] = Mx_exterior
    A[Ny * (Nx + 1) : (Ny + 1) * (Nx + 1), Ny * (Nx + 1) : (Ny + 1) * (Nx + 1)] = (
        Mx_exterior
    )

    A = A.tocsc()
    LU = splu(A)

    # Contador para pintar las grafica cada 100 activaciones del bucle
    i = 0

    b = zeros((Ny + 1, Nx + 1))

    error = []

    while t < tf - dt / 2:
        t += dt

        b = fuente(X, Y, t) * dt + U

        # Linea siguiente es si fuera neumann en 3
        # b[:, 0] += 2.0 * dt / dx * g(x[0], y, t)

        b[0, :] = u0(x, y, t) * Bn
        b[:, Nx] = u1(x, y, t) * Bn
        b[Ny, :] = u2(x, y, t) * Bn
        b[:, 0] = u3(x[0], y, t) * Bn

        b = b.reshape(N)

        # Metodo de punto fijo para la resolucion del sistema no lineal
        error_fijo = eps
        ut = U.reshape(N)
        iter = 0
        while error_fijo >= eps and iter <= 500:
            bPuntoFijo = b - dt * ut * ut
            ut_new = LU.solve(bPuntoFijo)

            error_fijo = max(abs(ut_new - ut))
            # print(f"ERROR PUNTO FIJO: {error_fijo}")

            ut = ut_new
            iter += 1

        # El usol sera la nueva solucion de u en el instante de tiempo t + dt
        U = ut.reshape((Ny + 1, Nx + 1))

        # Errores
        # Pongo explicitamente np para que no se solape con max de python
        err = np.max(abs(U - exacta(X, Y, t)).reshape(N))  # type: ignore
        # print("Error espacial cometido:",format(err))

        error = append(error, err)

        if i % 75 == 0 and dibujar:
            usol = U
            cu = contourf(X, Y, usol, 20)
            colorbar(cu)
            cl = contour(X, Y, usol, 20, colors="k")
            clabel(cl, inline=1, fontsize=8)
            title(f"Problema contorno 2D, t={round(t, 3)}")

            show()

        print(f"Iteracion {i} | Error: {error[-1]}")

        i += 1

    if dibujar:
        usol = usol.reshape((Ny + 1, Nx + 1))
        cu = contourf(X, Y, usol, 20)
        colorbar(cu)
        cl = contour(X, Y, usol, 20, colors="k")
        clabel(cl, inline=1, fontsize=8)
        title(f"Problema contorno 2D, t={round(tf, 3)}")
        show()

    return max(error)


# Cambia el algoritmo de punto fijo empleado
#   El termino dt * u_{ij}^{n+1, l} unicamente se suma a la diagonal!!! (Piensa en la estructura de A)
def ejer5_v2(
    xi,
    xf,
    Nx,
    yi,
    yf,
    Ny,
    t0,
    tf,
    Nt,
    u0,
    u1,
    u2,
    u3,
    estado_inicial,
    fuente,
    eps,
    exacta=None,
    dibujar=True,
):
    Bn = 10e20

    Nx = int(Nx)
    Ny = int(Ny)
    xi = float(xi)
    xf = float(xf)
    yi = float(yi)
    yf = float(yf)

    dx = (xf - xi) / float(Nx)
    dx2 = dx * dx
    dy = (yf - yi) / float(Ny)
    dy2 = dy * dy
    dt = (tf - t0) / float(Nt)
    N = (Nx + 1) * (Ny + 1)

    t = float(t0)
    tf = float(tf)

    x = linspace(xi, xf, Nx + 1)
    y = linspace(yi, yf, Ny + 1)
    X, Y = meshgrid(x, y)
    U = estado_inicial(X, Y)

    A_base = lil_matrix((N, N), dtype="float64")
    Mx_interior = lil_matrix((Nx + 1, Nx + 1), dtype="float64")
    Mx_exterior = lil_matrix((Nx + 1, Nx + 1), dtype="float64")
    My = lil_matrix((Nx + 1, Nx + 1), dtype="float64")

    #   Comparar con lo que tengo en casa que esta bien
    Mx_interior.setdiag(1.0 + 2.0 * (1.0 / (dx2) + 1.0 / (dy2)) * dt * ones(Nx + 1), 0)
    Mx_interior.setdiag(-1.0 / (dx2) * dt * ones(Nx), 1)
    Mx_interior.setdiag(-1.0 / (dx2) * dt * ones(Nx), -1)

    Mx_exterior.setdiag(Bn * ones(Nx + 1), 0)

    My.setdiag(-1.0 / (dy2) * dt * ones(Nx + 1), 0)

    # Mx_interior[0, 1] *= 2

    Mx_interior[0, 0] = Bn
    Mx_interior[-1, -1] = Bn

    for i in range(1, Ny):
        A_base[i * (Nx + 1) : (i + 1) * (Nx + 1), i * (Nx + 1) : (i + 1) * (Nx + 1)] = (
            Mx_interior
        )
        A_base[i * (Nx + 1) : (i + 1) * (Nx + 1), (i - 1) * (Nx + 1) : i * (Nx + 1)] = (
            My
        )
        A_base[
            i * (Nx + 1) : (i + 1) * (Nx + 1), (i + 1) * (Nx + 1) : (i + 2) * (Nx + 1)
        ] = My

    # Condiccion de contorno en bordes inferior y superior
    A_base[0 : (Nx + 1), 0 : (Nx + 1)] = Mx_exterior
    A_base[Ny * (Nx + 1) : (Ny + 1) * (Nx + 1), Ny * (Nx + 1) : (Ny + 1) * (Nx + 1)] = (
        Mx_exterior
    )

    # Contador para pintar las grafica cada 100 activaciones del bucle
    j = 0

    b = zeros((Ny + 1, Nx + 1))

    error = array(0.0, dtype="float64")

    while t < tf - dt / 2:
        t += dt

        b = fuente(X, Y, t) * dt + U

        b[0, :] = u0(x, y, t) * Bn
        b[:, Nx] = u1(x, y, t) * Bn
        b[Ny, :] = u2(x, y, t) * Bn
        b[:, 0] = u3(x[0], y, t) * Bn

        b = b.reshape(N)

        # Metodo de punto fijo para la resolucion del sistema no lineal
        error_fijo = eps
        ul = U.reshape(N)
        iter = 0
        while error_fijo >= eps and iter <= 500:
            #! NECESITO RESETEAR LA MATRIX EN CADA ITERACCION
            A = A_base.copy()

            #! Estamos sumando u_{ij}, por la estrucutra de A, esto son los coeficientes asociados a la diagonal
            A.setdiag(A.diagonal() + 1 + dt * ul)

            #! La linea siguiente crashea el sistema
            # A += 1 + dt * ul
            A = A.tocsc()
            LU = splu(A)

            ul_new = LU.solve(b)

            error_fijo = max(abs(ul_new - ul))

            ul = ul_new
            iter += 1

        U = ul.reshape((Ny + 1, Nx + 1))

        err = np.max(abs(U - exacta(X, Y, t)).reshape(N))  # type: ignore
        # print("Error espacial cometido:",format(err))

        error = append(error, err)

        if j % 25 == 0 and dibujar:
            usol = U
            cu = contourf(X, Y, usol, 20)
            colorbar(cu)
            cl = contour(X, Y, usol, 20, colors="k")
            clabel(cl, inline=1, fontsize=8)
            title(f"Problema contorno 2D, t={round(t, 3)}")

            show()

        print("Error:", error[-1])

        j += 1

    if dibujar:
        usol = usol.reshape((Ny + 1, Nx + 1))
        cu = contourf(X, Y, usol, 20)
        colorbar(cu)
        cl = contour(X, Y, usol, 20, colors="k")
        clabel(cl, inline=1, fontsize=8)
        title(f"Problema contorno 2D, t={tf}")
        show()

    return max(error)


def ejer5_newton(
    xi,
    xf,
    Nx,
    yi,
    yf,
    Ny,
    t0,
    tf,
    Nt,
    u0,
    u1,
    u2,
    u3,
    estado_inicial,
    fuente,
    eps,
    exacta=None,
    dibujar=True,
):
    Bn = 10e20

    Nx = int(Nx)
    xi = float(xi)
    xf = float(xf)
    Ny = int(Ny)
    yi = float(yi)
    yf = float(yf)
    Nt = int(Nt)
    t = float(t0)
    tf = float(tf)

    N = (Nx + 1) * (Ny + 1)

    dx = (xf - xi) / float(Nx)
    dy = (yf - yi) / float(Ny)
    dt = (tf - t0) / float(Nt)

    dx2 = dx * dx
    dy2 = dy * dy

    x = linspace(xi, xf, Nx + 1)
    y = linspace(yi, yf, Ny + 1)
    X, Y = meshgrid(x, y)
    U = estado_inicial(X, Y)

    A = lil_matrix((N, N), dtype="float64")
    Mx = lil_matrix((Nx + 1, Nx + 1), dtype="float64")
    My = lil_matrix((Nx + 1, Nx + 1), dtype="float64")

    Mx_exterior = lil_matrix((Nx + 1, Nx + 1), dtype="float64")
    Mx_exterior.setdiag(Bn * ones(Nx + 1), 0)

    Mx.setdiag(1.0 + 2.0 * (1.0 / (dx2) + 1.0 / (dy2)) * dt, 0)
    Mx.setdiag(-dt / (dx2), 1)
    Mx.setdiag(-dt / (dx2), -1)

    My.setdiag(-dt / (dy2), 0)

    # Condiccion neumann en borde izq (Si fuera neumann)
    # Mx[0, 1] *= 2

    for i in range(1, Ny):
        A[i * (Nx + 1) : (i + 1) * (Nx + 1), i * (Nx + 1) : (i + 1) * (Nx + 1)] = Mx
        A[i * (Nx + 1) : (i + 1) * (Nx + 1), (i - 1) * (Nx + 1) : i * (Nx + 1)] = My
        A[
            i * (Nx + 1) : (i + 1) * (Nx + 1), (i + 1) * (Nx + 1) : (i + 2) * (Nx + 1)
        ] = My

    # Discretizacion finitas de bordes inferior y superior
    #! Necesitamos las matrices My, en este caso no estamos imponiendo penalizacion aqui,
    #   sino que las necesitamos para obtener el esquema de dif.finitas correcto
    A[0 : (Nx + 1), 0 : (Nx + 1)] = Mx
    A[0 : (Nx + 1), (Nx + 1) : 2 * (Nx + 1)] = My

    A[Ny * (Nx + 1) : (Ny + 1) * (Nx + 1), Ny * (Nx + 1) : (Ny + 1) * (Nx + 1)] = Mx
    A[Ny * (Nx + 1) : N, (Ny - 1) * (Nx + 1) : Ny * (Nx + 1)] = My

    A.tocsc()

    # Contador para pintar las grafica cada 100 activaciones del bucle
    i = 0

    b = zeros((Ny + 1, Nx + 1))

    # Listas son mas eficientes para almacenar errores
    error = []

    while t < tf - dt / 2:
        t += dt

        b = fuente(X, Y, t) * dt + U

        # Si fuera neumann a izquierda
        # En la esquina usamos dirichlet
        # No restamos porque a diferencia del caso del caso de dirichlet no estamos imponiendo el valor
        # b[1:Ny, 0] += 2.0 * dt / dx * u3(x[0], y[1:Ny], t)
        b = b.reshape(N)

        # Metodo de punto fijo para la resolucion del sistema no lineal por met.Newton
        error_fijo = eps
        ut = U.reshape(N)

        iter = 0
        while error_fijo >= eps and iter <= 500:
            J = A.copy()

            # Establecemos valores del jacobiano cuando derivamos respecto U_{ii}
            nueva_diagonal = 2.0 * dt * ut
            J.setdiag(J.diagonal() + nueva_diagonal)

            # Termino independiente
            G = -b + (A @ ut) + dt * ut * ut

            # ---- Condicciones Contorno ---- #
            # Preparamos jacobiano para imponer condicciones contorno

            # Borde inferior (y=yi, i=0..Nx)
            for k in range(Nx + 1):
                J[k, :] = 0
                J[k, k] = Bn
                # G[k] = Bn * (ut[k] - u0(x[k], y[0], t))

            # Borde superior (y=yf, i=Ny*(Nx+1)..N-1)
            for k in range(Ny * (Nx + 1), N):
                J[k, :] = 0
                J[k, k] = Bn
                # G[k] = Bn * (ut[k] - u2(x[k - Ny * (Nx + 1)], y[Ny], t))

            # Borde derecho (x=xf, nodos k = (i+1)*(Nx+1)-1 )
            for j in range(1, Ny):
                k_der = (j + 1) * (Nx + 1) - 1
                J[k_der, :] = 0
                J[k_der, k_der] = Bn
                # G[k] = Bn * (ut[k] - u1(x[Nx], y[i], t))

                # Bucle para el borde izquierdo
                k_izq = j * (Nx + 1)
                J[k_izq, :] = 0
                J[k_izq, k_izq] = Bn
                # F[k_izq] = Bn * (ut[k_izq] - condicion_borde_izquierdo(y[j], t))

            J = J.tocsc()
            G = G.reshape((Ny + 1, Nx + 1))
            ut = ut.reshape((Ny + 1, Nx + 1))

            # Restamos incremento para que se quede igual tras resolver
            # Primero ut ya que despues tomamos como termino independiente (-G)
            G[0, :] = (ut[0, :] - u0(x, y[0], t)) * Bn
            G[Ny, :] = (ut[Ny, :] - u2(x, y[-1], t)) * Bn
            G[:, Nx] = (ut[:, Nx] - u1(x[-1], y, t)) * Bn
            G[:, 0] = (ut[:, 0] - u3(x[0], y, t)) * Bn

            ut = ut.reshape(N)
            G = G.reshape(N)

            LU = splu(J)
            incremento = LU.solve(-G)

            error_fijo = max(abs(incremento))
            print(f"ERROR PUNTO FIJO: {error_fijo}")

            ut = ut + incremento
            iter += 1

        U = ut.reshape((Ny + 1, Nx + 1))

        err = np.max(abs(U - exacta(X, Y, t)).reshape(N))  # type: ignore
        error = append(error, err)

        if i % 25 == 0 and dibujar:
            usol = U
            cu = contourf(X, Y, usol, 20)
            colorbar(cu)
            cl = contour(X, Y, usol, 20, colors="k")
            clabel(cl, inline=1, fontsize=8)
            title(f"Problema contorno 2D, t={round(t, 3)}")

            show()

        print(f"Iteracion {i} | Error: {error[-1]}")

        i += 1

    if dibujar:
        usol = usol.reshape((Ny + 1, Nx + 1))
        cu = contourf(X, Y, usol, 20)
        colorbar(cu)
        cl = contour(X, Y, usol, 20, colors="k")
        clabel(cl, inline=1, fontsize=8)
        title(f"Problema contorno 2D, t={round(tf, 3)}")
        show()

    return max(error)


##############################################
##############################################
##############################################
##############################################


def f0(t, x, y):
    return cos(x) * sin(y) * (cos(t) + sin(t) * sin(t) * cos(x) * sin(y) + 2 * sin(t))


def borde_nulo(t, x, y):
    return 0 * x * t * y


def nula(x, y):
    return 0 * x * y


def u3(x, t, y):
    return sin(t) * sin(y)


def exacta(t, x, y):
    return cos(x) * sin(y) * sin(t)


# Comprobacion
error = 0

calculo = [100, 200, 400]
# calculo = []

if calculo.__len__() != 0:
    print("Caso 5")

    L = 2.0
    T = 1.0

    for i in calculo:
        # fmt: off
        nuevo_error = ejer5_newton(0.0, 1/2 * pi, i, 0.0, pi, i, 0.0, 2*pi, 1000, borde_nulo, borde_nulo, borde_nulo, u3, nula, f0, 10e-7, exacta, False)
        # fmt: on
        print("-------------------------")
        if error == 0:
            print("ERROR", nuevo_error)
        else:
            print("Cociente de error", (error / nuevo_error))
        error = nuevo_error
        print("-------------------------")

    pause(50000)
