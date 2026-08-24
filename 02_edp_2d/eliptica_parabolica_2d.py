"""
Eliptica y parabolica 2D - diferencias finitas
Problema u - nu Deltau = f con Dirichlet y Neumann (derivada normal)
en malla (Nx+1)x(Ny+1). Parabolica u_t - nu Deltau = f con Euler
implicito y theta-metodo. Orden 2 en espacio.
Ver docs/enunciados_resumidos.md#02_edp_2d
"""

# Derivada normal es  du / dn = gradiente(u) * normal (prod.escalar) (si es negativo es mas sencillo hacer el cambio en la funcion del contorno, i.e, considerar  (-g))
#   con normal un vector unitario en la direccion de la derivada

import time
from numpy import *
from scipy.sparse.linalg import splu
from scipy.sparse import lil_matrix, identity
from matplotlib.pyplot import *

style.use("dark_background")


def f0(x, y):
    # z = sin(x * y) * (1 + v*(x**2 + y**2))
    z = sin(x * y) * (1 + x**2 + y**2)
    return z


def u0(x):
    z = 0 * x
    return z


def u1(y):
    z = sin(2 * pi * y)
    return z


def u2(x):
    z = sin(2 * pi * x)
    return z


def u3(x):
    z = 0 * x
    return z


def un(y):
    return y * cos(2 * pi * y)


def exacta(x, y):
    z = sin(x * y)
    return z


def eliptico_2d_dirichlet_penalizacion(
    xi, xf, Nx, yi, yf, Ny, nu, u0, u1, u2, u3, fuente
):
    Bn = 1e20

    Nx = int(Nx)
    Ny = int(Ny)
    xi = float(xi)
    xf = float(xf)
    yi = float(yi)
    yf = float(yf)

    dx = (xf - xi) / float(Nx)
    dy = (yf - yi) / float(Ny)
    dx2 = dx * dx
    dy2 = dy * dy
    N = (Nx + 1) * (Ny + 1)

    x = linspace(xi, xf, Nx + 1)
    y = linspace(yi, yf, Ny + 1)
    X, Y = meshgrid(x, y)

    A = lil_matrix((N, N), dtype="float64")
    Mx = lil_matrix((Nx + 1, Nx + 1), dtype="float64")
    My = lil_matrix((Nx + 1, Nx + 1), dtype="float64")

    Mx.setdiag(1.0 + nu * (2.0 * (1.0 / dx2 + 1.0 / dy2) * ones(Nx + 1)), 0)
    Mx.setdiag(-nu / dx2 * ones(Nx + 1), 1)
    Mx.setdiag(-nu / dx2 * ones(Nx), -1)
    My.setdiag(-nu / dy2 * ones(Nx), 0)

    Mx[0, 0] = Bn
    Mx[Nx, Nx] = Bn

    for i in range(1, Ny):
        A[i * (Nx + 1) : (i + 1) * (Nx + 1), i * (Nx + 1) : (i + 1) * (Nx + 1)] = Mx
        A[i * (Nx + 1) : (i + 1) * (Nx + 1), (i - 1) * (Nx + 1) : i * (Nx + 1)] = My
        A[
            i * (Nx + 1) : (i + 1) * (Nx + 1), (i + 1) * (Nx + 1) : (i + 2) * (Nx + 1)
        ] = My

    # Imponemos los cambios en el primer y ultimo bloque de la diagonal para imponer las condicciones de contorno
    #   (penalizacion en este caso)
    Mx.setdiag(Bn * ones(Nx + 1), 0)
    A[0 : (Nx + 1), 0 : (Nx + 1)] = Mx
    A[Ny * (Nx + 1) : (Ny + 1) * (Nx + 1), Ny * (Nx + 1) : (Ny + 1) * (Nx + 1)] = Mx

    A = A.tocsc()

    b = zeros((Ny + 1, Nx + 1))
    b = fuente(X, Y)

    b[0, :] = Bn * u0(x)
    b[Ny, :] = Bn * u2(x)
    b[:, 0] = Bn * u3(y)
    b[:, Nx] = Bn * u1(y)
    b = b.reshape(N)

    LU = splu(A)
    usol = LU.solve(b)

    usol = usol.reshape((Ny + 1, Nx + 1))
    cu = contourf(X, Y, usol, 20)
    colorbar(cu)
    cl = contour(X, Y, usol, 20, colors="k")
    clabel(cl, inline=1, fontsize=8)
    show()
    error = max(abs(usol - exacta(X, Y)).reshape(N))
    print("Error:", error)


def eliptico_2d_dirichlet_simetrizacion(
    xi, xf, Nx, yi, yf, Ny, nu, u0, u1, u2, u3, fuente
):
    Nx = int(Nx)
    Ny = int(Ny)
    xi = float(xi)
    xf = float(xf)
    yi = float(yi)
    yf = float(yf)

    dx = (xf - xi) / float(Nx)
    dy = (yf - yi) / float(Ny)
    N = (Nx + 1) * (Ny + 1)

    x = linspace(xi, xf, Nx + 1)
    y = linspace(yi, yf, Ny + 1)
    X, Y = meshgrid(x, y)

    A = lil_matrix((N, N), dtype="float64")
    Mx = lil_matrix((Nx + 1, Nx + 1), dtype="float64")
    My = lil_matrix((Nx + 1, Nx + 1), dtype="float64")
    Id = identity(N, dtype="float64", format="csc")

    Mx.setdiag(2.0 * (1.0 / (dx**2) + 1.0 / (dy**2)) * ones(Nx + 1), 0)
    Mx.setdiag(-1.0 / (dx**2) * ones(Nx + 1), 1)
    Mx.setdiag(-1.0 / (dx**2) * ones(Nx), -1)
    My.setdiag(-1.0 / (dy**2) * ones(Nx), 0)

    Mx[0, 0] = 0.0
    Mx[0, 1] = 0.0
    Mx[Nx, Nx] = 0.0
    Mx[Nx, Nx - 1] = 0.0

    Mx[1, 0] = 0.0
    Mx[Nx - 1, Nx] = 0.0

    My[0, 0] = 0.0
    My[Nx, Nx] = 0.0

    for i in range(1, Ny):
        A[i * (Nx + 1) : (i + 1) * (Nx + 1), i * (Nx + 1) : (i + 1) * (Nx + 1)] = Mx
        if i > 1:
            A[i * (Nx + 1) : (i + 1) * (Nx + 1), (i - 1) * (Nx + 1) : i * (Nx + 1)] = My
        if i < Ny - 1:
            A[
                i * (Nx + 1) : (i + 1) * (Nx + 1),
                (i + 1) * (Nx + 1) : (i + 2) * (Nx + 1),
            ] = My

    A = Id + nu * A
    A = A.tocsc()

    b = zeros((Ny + 1, Nx + 1))
    b = fuente(X, Y)

    b[0, :] = u0(x)
    b[1, 1:Nx] += nu * u0(x[1:Nx]) / (dy * dy)

    b[Ny, :] = u2(x)
    b[(Ny - 1), 1:Nx] += nu * u2(x[1:Nx]) / (dy * dy)

    b[:, 0] = u3(y)
    b[1:Ny, 1] += nu * u3(y[1:Ny]) / (dx * dx)

    b[:, Nx] = u1(y)
    b[1:Ny, (Nx - 1)] += nu * u1(y[1:Ny]) / (dx * dx)

    b = b.reshape(N)

    LU = splu(A)
    usol = LU.solve(b)

    usol = usol.reshape((Ny + 1, Nx + 1))
    cu = contourf(X, Y, usol, 20)
    colorbar(cu)
    cl = contour(X, Y, usol, 20, colors="k")
    clabel(cl, inline=1, fontsize=8)
    show()
    error = max(abs(usol - exacta(X, Y)).reshape(N))
    print("Error:", error)

    return error


def eliptico_no_sim(xi, xf, Nx, yi, yf, Ny, nu, u0, u1, u2, u3, fuente):
    Nx = int(Nx)
    Ny = int(Ny)
    xi = float(xi)
    xf = float(xf)
    yi = float(yi)
    yf = float(yf)

    dx = (xf - xi) / float(Nx)
    dy = (yf - yi) / float(Ny)
    N = (Nx + 1) * (Ny + 1)

    x = linspace(xi, xf, Nx + 1)
    y = linspace(yi, yf, Ny + 1)
    X, Y = meshgrid(x, y)

    A = lil_matrix((N, N), dtype="float64")
    Mx = lil_matrix((Nx + 1, Nx + 1), dtype="float64")
    My = lil_matrix((Nx + 1, Nx + 1), dtype="float64")
    Id = identity(N, dtype="float64", format="csc")

    Mx.setdiag(2.0 * (1.0 / (dx**2) + 1.0 / (dy**2)) * ones(Nx + 1), 0)
    Mx.setdiag(-1.0 / (dx**2) * ones(Nx), 1)
    Mx.setdiag(-1.0 / (dx**2) * ones(Nx), -1)
    My.setdiag(-1.0 / (dy**2) * ones(Nx + 1), 0)

    Mx[0, 0] = 0.0
    Mx[0, 1] = 0.0
    Mx[Nx, Nx] = 0.0
    Mx[Nx, Nx - 1] = 0.0

    My[0, 0] = 0.0
    My[Nx, Nx] = 0.0

    for i in range(1, Ny):
        A[i * (Nx + 1) : (i + 1) * (Nx + 1), i * (Nx + 1) : (i + 1) * (Nx + 1)] = Mx
        A[i * (Nx + 1) : (i + 1) * (Nx + 1), (i - 1) * (Nx + 1) : i * (Nx + 1)] = My
        A[
            i * (Nx + 1) : (i + 1) * (Nx + 1), (i + 1) * (Nx + 1) : (i + 2) * (Nx + 1)
        ] = My

    A = Id + nu * A
    A = A.tocsc()

    b = zeros((Ny + 1, Nx + 1))
    b = fuente(X, Y)

    b[0, :] = u0(x)
    b[Ny, :] = u2(x)
    b[:, 0] = u3(y)
    b[:, Nx] = u1(y)
    b = b.reshape(N)

    LU = splu(A)
    usol = LU.solve(b)

    usol = usol.reshape((Ny + 1, Nx + 1))
    cu = contourf(X, Y, usol, 20)
    colorbar(cu)
    cl = contour(X, Y, usol, 20, colors="k")
    clabel(cl, inline=1, fontsize=8)
    show()
    error = max(abs(usol - exacta(X, Y)).reshape(N))
    print("Error:", error)


# METODO TIENE ORDEN 2 (diferencias finitas 2 orden)
# eliptico_2d_dirichlet_penalizacion(
#     0.0, 2 * pi, 200, 0.0, 2 * pi, 200, 1.0, u0, u1, u2, u3, f0
# )

# eliptico_2d_dirichlet_simetrizacion(0.0,2*pi,200,0.0,2*pi,200,1.0,u0,u1,u2,u3,f0)
# eliptico_2d_dirichlet_no_simetrico(0.0,2*pi,200,0.0,2*pi,200,1.0,u0,u1,u2,u3,f0)


#######################################
#######################################
#######################################
#######################################
# Esta aproximacion asegura que el error de truncamiento local es O(h_x^2) + O(h_y^2) en todo el dominio (incluyendo la frontera de Neumann),
#   lo que generalmente conduce a una convergencia global de segundo orden para la solucion numerica.
# Metodo nodo fantasma preserva el orden del metodo
# -> Principio de Precedencia: En la interseccion de fronteras con diferentes tipos de condiciones,
#       la condicion de Dirichlet siempre tiene precedencia sobre la de Neumann. (aporta menos error al metodo)


# Metodo orden 2 (correctamente implementado)
def eliptico_normal_penalizacion(
    xi, xf, Nx, yi, yf, Ny, nu, u0, g, u2, u3, fuente, exacta, dibujar=True
):
    Bn = 1e20

    Nx = int(Nx)
    Ny = int(Ny)
    xi = float(xi)
    xf = float(xf)
    yi = float(yi)
    yf = float(yf)

    dx = (xf - xi) / float(Nx)
    dy = (yf - yi) / float(Ny)
    dx2 = dx * dx
    dy2 = dy * dy
    N = (Nx + 1) * (Ny + 1)

    x = linspace(xi, xf, Nx + 1)
    y = linspace(yi, yf, Ny + 1)
    X, Y = meshgrid(x, y)

    A = lil_matrix((N, N), dtype="float64")

    Mx_interior = lil_matrix((Nx + 1, Nx + 1), dtype="float64")
    My = lil_matrix((Nx + 1, Nx + 1), dtype="float64")

    Mx_interior.setdiag(1.0 + nu * (2.0 / dx2 + 2.0 / dy2) * ones(Nx + 1), 0)
    Mx_interior.setdiag(-nu / dx2 * ones(Nx), 1)
    Mx_interior.setdiag(-nu / dx2 * ones(Nx), -1)
    My.setdiag(-nu / dy2 * ones(Nx + 1), 0)

    Mx_interior[0, 0] = Bn
    # Necesario para aislar la ecuacion de contorno
    Mx_interior[0, 1] = 0
    # Mx[Nx, Nx] = Bn

    # 2. Frontera derecha (Cond.contorno neumann)
    Mx_interior[-1, -2] *= 2

    # Construimos el interior de la matriz de coeficientes
    for i in range(1, Ny):
        A[i * (Nx + 1) : (i + 1) * (Nx + 1), i * (Nx + 1) : (i + 1) * (Nx + 1)] = (
            Mx_interior
        )
        A[i * (Nx + 1) : (i + 1) * (Nx + 1), (i - 1) * (Nx + 1) : i * (Nx + 1)] = My
        A[
            i * (Nx + 1) : (i + 1) * (Nx + 1), (i + 1) * (Nx + 1) : (i + 2) * (Nx + 1)
        ] = My

    # Imponer condiccion contorno (dirichlet en este caso) en los bordes inferiores y superiores
    Mx_dirichlet = lil_matrix((Nx + 1, Nx + 1), dtype="float64")
    Mx_dirichlet.setdiag(Bn * ones(Nx + 1), 0)

    A[0 : (Nx + 1), 0 : (Nx + 1)] = Mx_dirichlet
    A[Ny * (Nx + 1) : (Ny + 1) * (Nx + 1), Ny * (Nx + 1) : (Ny + 1) * (Nx + 1)] = (
        Mx_dirichlet
    )

    A = A.tocsc()

    b = zeros((Ny + 1, Nx + 1))
    b = fuente(X, Y)

    b[0, :] = Bn * u0(x)
    b[Ny, :] = Bn * u2(x)
    b[:, 0] = Bn * u3(y)

    # Condiccion tipo neumann en toda la frontera Gamma_1 (borde derecho)
    # Aplicado solo a los nodos interiores de la frontera
    b[1:Ny, Nx] += g(y[1:Ny]) * 2.0 * nu / dx

    b = b.reshape(N)

    LU = splu(A)
    usol = LU.solve(b)
    usol = usol.reshape((Ny + 1, Nx + 1))

    if dibujar:
        cu = contourf(X, Y, usol, 20)
        colorbar(cu)
        cl = contour(X, Y, usol, 20, colors="k")
        clabel(cl, inline=1, fontsize=8)
        title("Problema contorno 2D")
        show()

    error = max(abs(usol - exacta(X, Y)).reshape(N))
    # print("Error:", error)

    return error


# No da convergencia orden 2 en espacio
# Se ha impuesto la condiccion de frontera de neuamnn por medio de la tecnica de "medio volumen de control"
def eliptico_normal_simetrica(
    xi, xf, Nx, yi, yf, Ny, nu, u0, g, u2, u3, fuente, exacta, dibujar=True
):
    Nx = int(Nx)
    Ny = int(Ny)
    xi = float(xi)
    xf = float(xf)
    yi = float(yi)
    yf = float(yf)

    dx = (xf - xi) / float(Nx)
    dy = (yf - yi) / float(Ny)
    dx2 = dx * dx
    dy2 = dy * dy
    N = (Nx + 1) * (Ny + 1)

    x = linspace(xi, xf, Nx + 1)
    y = linspace(yi, yf, Ny + 1)
    X, Y = meshgrid(x, y)

    L = lil_matrix((N, N), dtype="float64")

    # --- Construccion de los bloques matriciales ---

    # 1. Bloque Mx: representa la discretizacion en X
    #    Este bloque sera el mismo para todas las filas interiores i=1,...,Ny-1
    Mx = lil_matrix((Nx + 1, Nx + 1), dtype="float64")
    # Diagonal principal para nodos interiores (j=1,...,Nx-1)
    Mx.setdiag(2.0 / dx2 * ones(Nx + 1), 0)
    # Diagonales secundarias
    Mx.setdiag(-1.0 / dx2 * ones(Nx), 1)
    Mx.setdiag(-1.0 / dx2 * ones(Nx), -1)

    # --- Cambio para  Simetria ---
    # En la frontera derecha (j=Nx), la contribucion de la derivada en x se escala por 1/2.
    Mx[Nx, Nx] = 1.0 / dx2

    # 2. Bloque My: representa la discretizacion en Y
    My = lil_matrix((Nx + 1, Nx + 1), dtype="float64")
    My.setdiag(-1.0 / dy2 * ones(Nx + 1), 0)

    # En la frontera derecha (j=Nx), la contribucion de la derivada en y se escala por 1/2.
    My[Nx, Nx] = -0.5 / dy2

    for i in range(1, Ny):
        # Bloque diagonal: Contiene la parte en X y la parte central de Y
        L_diag = Mx.copy()
        diag_y = 2.0 / dy2 * ones(Nx + 1)

        # La parte central de Y se escala por 1/2 en la frontera Neumann
        diag_y[Nx] *= 0.5
        L_diag.setdiag(L_diag.diagonal(0) + diag_y, 0)

        # Asignar el bloque diagonal
        L[i * (Nx + 1) : (i + 1) * (Nx + 1), i * (Nx + 1) : (i + 1) * (Nx + 1)] = L_diag

        # Bloques extra-diagonales
        L[i * (Nx + 1) : (i + 1) * (Nx + 1), (i - 1) * (Nx + 1) : i * (Nx + 1)] = My
        L[
            i * (Nx + 1) : (i + 1) * (Nx + 1), (i + 1) * (Nx + 1) : (i + 2) * (Nx + 1)
        ] = My

    A = identity(N, dtype="float64", format="lil") + nu * L
    A = A.tolil()

    # --- Forzar condiciones de Dirichlet en la matriz A ---
    # Para cualquier nodo k en una frontera Dirichlet, la ecuacion es simplemente u_k = valor.
    # Esto se logra poniendo un 1 en A[k,k] y 0 en el resto de la fila.
    for j in range(Nx + 1):  # Fronteras inferior y superior
        k_bottom = 0 * (Nx + 1) + j
        k_top = Ny * (Nx + 1) + j
        A[k_bottom, :] = 0
        A[k_bottom, k_bottom] = 1.0
        A[k_top, :] = 0
        A[k_top, k_top] = 1.0

    for i in range(1, Ny):  # Frontera izquierda
        k_left = i * (Nx + 1) + 0
        A[k_left, :] = 0
        A[k_left, k_left] = 1.0

    A = A.tocsc()

    b = fuente(X, Y)

    # MODIFICACION PARA NEUMANN SIMETRICO
    # En la frontera derecha, se anade el termino de Neumann escalado
    b[1:Ny, Nx] = 0.5 * b[1:Ny, Nx] + nu * g(y[1:Ny]) / dx

    # Ahora aplicamos las condiciones de contorno de Dirichlet a b
    b[0, :] = u0(x)
    b[Ny, :] = u2(x)
    b[:, 0] = u3(y)

    # Finalmente, movemos los terminos conocidos (fronteras de Dirichlet)
    # que afectan a las ecuaciones de los nodos interiores.
    b[1, 1 : Nx + 1] -= nu * (u0(x[1 : Nx + 1]) * (-1.0 / dy2))
    b[Ny - 1, 1 : Nx + 1] -= nu * (u2(x[1 : Nx + 1]) * (-1.0 / dy2))
    b[1:Ny, 1] -= nu * (u3(y[1:Ny]) * (-1.0 / dx2))
    # No hay termino para la frontera derecha porque es Neumann (incognita).

    b = b.reshape(N)

    A = A.tocsc()
    LU = splu(A)
    usol = LU.solve(b)
    usol = usol.reshape((Ny + 1, Nx + 1))

    if dibujar:
        cu = contourf(X, Y, usol, 20)
        colorbar(cu)
        cl = contour(X, Y, usol, 20, colors="k")
        clabel(cl, inline=1, fontsize=8)
        title("Problema contorno 2D (Esquema Simetrico)")
        show()

    error = max(abs(usol - exacta(X, Y)).flatten())
    print("Error (esquema simetrico):", error)

    return error


def eliptico_normal_no_sim(
    xi, xf, Nx, yi, yf, Ny, nu, u0, g, u2, u3, fuente, exacta, dibujar=True
):
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
    N = (Nx + 1) * (Ny + 1)

    x = linspace(xi, xf, Nx + 1)
    y = linspace(yi, yf, Ny + 1)
    X, Y = meshgrid(x, y)

    A = lil_matrix((N, N), dtype="float64")
    Mx_interior = lil_matrix((Nx + 1, Nx + 1), dtype="float64")
    My = lil_matrix((Nx + 1, Nx + 1), dtype="float64")
    Id = identity(Nx + 1, dtype="float64", format="csc")

    Mx_interior.setdiag(1 + 2.0 * (1.0 / dx2 + 1.0 / dy2) * ones(Nx + 1), 0)
    Mx_interior.setdiag(-1.0 / dx2 * ones(Nx), 1)
    Mx_interior.setdiag(-1.0 / dx2 * ones(Nx), -1)
    My.setdiag(-1.0 / dy2 * ones(Nx + 1), 0)

    Mx_interior[0, 0] = 1.0
    Mx_interior[0, 1] = 0.0

    # 2. Frontera derecha (Cond.contorno neumann)
    Mx_interior[-1, -2] *= 2

    My[0, 0] = 0.0
    My[Nx, Nx] = 0.0

    for i in range(1, Ny):
        A[i * (Nx + 1) : (i + 1) * (Nx + 1), i * (Nx + 1) : (i + 1) * (Nx + 1)] = (
            Mx_interior
        )
        A[i * (Nx + 1) : (i + 1) * (Nx + 1), (i - 1) * (Nx + 1) : i * (Nx + 1)] = My
        A[
            i * (Nx + 1) : (i + 1) * (Nx + 1), (i + 1) * (Nx + 1) : (i + 2) * (Nx + 1)
        ] = My

    A[0 : (Nx + 1), 0 : (Nx + 1)] = Id
    A[Ny * (Nx + 1) : (Ny + 1) * (Nx + 1), Ny * (Nx + 1) : (Ny + 1) * (Nx + 1)] = Id

    A = A.tocsc()

    b = zeros((Ny + 1, Nx + 1))
    b = fuente(X, Y)

    b[0, :] = u0(x)
    b[Ny, :] = u2(x)
    b[:, 0] = u3(y)
    b[1:Ny, Nx] += g(y[1:Ny]) * 2.0 * nu / dx
    b = b.reshape(N)

    LU = splu(A)
    usol = LU.solve(b)
    usol = usol.reshape((Ny + 1, Nx + 1))

    if dibujar:
        cu = contourf(X, Y, usol, 20)
        colorbar(cu)
        cl = contour(X, Y, usol, 20, colors="k")
        clabel(cl, inline=1, fontsize=8)
        show()

    error = max(abs(usol - exacta(X, Y)).reshape(N))
    print("Error:", error)

    return error


# Comprobacion
error = 0
eps = 10e-7

# calculo = [100, 200, 400]
calculo = []

if calculo.__len__() != 0:
    print("Contorno 2D ~ Derivada normal derecha")
    for i in calculo:
        print("----------------------")
        nuevo_error = eliptico_normal_simetrica(
            0.0, 2 * pi, i, 0.0, 2 * pi, 400, 1.0, u0, un, u2, u3, f0, exacta, False
        )
        print("----------------------")

        if error == 0:
            print("ERROR", nuevo_error)
        else:
            print("Cociente de error (Convergencia)", (error / nuevo_error))

        error = nuevo_error

    pause(50000)


# #####################################
# #####################################
# #####################################
# #####################################
# #####################################
# #####################################
# #####################################


# Condiccion normal (tipo neumann) en borde inferior
def eliptico_normal_penalizacion_v2(
    xi, xf, Nx, yi, yf, Ny, nu, u0, g, u2, u3, fuente, exacta, dibujar=True
):
    Bn = 1e20

    Nx = int(Nx)
    Ny = int(Ny)
    xi = float(xi)
    xf = float(xf)
    yi = float(yi)
    yf = float(yf)

    dx = (xf - xi) / float(Nx)
    dy = (yf - yi) / float(Ny)
    dx2 = dx * dx
    dy2 = dy * dy
    N = (Nx + 1) * (Ny + 1)

    x = linspace(xi, xf, Nx + 1)
    y = linspace(yi, yf, Ny + 1)
    X, Y = meshgrid(x, y)

    A = lil_matrix((N, N), dtype="float64")

    Mx = lil_matrix((Nx + 1, Nx + 1), dtype="float64")
    Mx.setdiag(1.0 + nu * (2.0 / dx2 + 2.0 / dy2) * ones(Nx + 1), 0)
    Mx.setdiag(-nu / dx2 * ones(Nx), 1)
    Mx.setdiag(-nu / dx2 * ones(Nx), -1)

    My = lil_matrix((Nx + 1, Nx + 1), dtype="float64")
    My.setdiag(-nu / dy2 * ones(Nx + 1), 0)

    Mx_interior = Mx.copy()
    Mx_interior[0, 0] = Bn
    Mx_interior[0, 1] = 0
    Mx_interior[Nx, Nx] = Bn
    Mx_interior[Nx, Nx - 1] = 0

    for i in range(1, Ny):
        A[i * (Nx + 1) : (i + 1) * (Nx + 1), i * (Nx + 1) : (i + 1) * (Nx + 1)] = (
            Mx_interior
        )
        A[i * (Nx + 1) : (i + 1) * (Nx + 1), (i - 1) * (Nx + 1) : i * (Nx + 1)] = My
        A[
            i * (Nx + 1) : (i + 1) * (Nx + 1), (i + 1) * (Nx + 1) : (i + 2) * (Nx + 1)
        ] = My

    A[0 : (Nx + 1), 0 : (Nx + 1)] = Mx_interior
    A[0 : (Nx + 1), (Nx + 1) : 2 * (Nx + 1)] = 2 * My

    Mx_dirichlet_superior = lil_matrix((Nx + 1, Nx + 1), dtype="float64")
    Mx_dirichlet_superior.setdiag(Bn * ones(Nx + 1), 0)
    A[Ny * (Nx + 1) : N, Ny * (Nx + 1) : N] = Mx_dirichlet_superior

    A = A.tocsc()

    b = fuente(X, Y)

    b[Ny, :] = Bn * u2(x)
    b[:, 0] = Bn * u3(y)
    b[:, Nx] = Bn * g(y)
    b[0, 1:Nx] += u0(x[1:Nx]) * 2.0 * nu / dy

    b = b.reshape(N)

    LU = splu(A)
    usol = LU.solve(b)
    usol = usol.reshape((Ny + 1, Nx + 1))

    if dibujar:
        figure(figsize=(10, 8))
        cu = contourf(X, Y, usol, 20, cmap="viridis")
        colorbar(cu, label="Solucion u(x,y)")
        cl = contour(X, Y, usol, 20, colors="k", linewidths=0.5)
        clabel(cl, inline=1, fontsize=8)
        title("Solucion Numerica del Problema Eliptico 2D")
        xlabel("x")
        ylabel("y")
        show()

    error = max(abs(usol - exacta(X, Y)).reshape(N))
    print(f"Error maximo absoluto: {error}")

    return error


# Comprobacion
error = 0
eps = 10e-7

# calculo = [100, 200, 400]
calculo = []

if calculo.__len__() != 0:
    print("Contorno 2D ~ Derivada normal lado inferior")
    for i in calculo:
        print("----------------------")
        #! Estas condicciones no estan adaptadas a condiccion neumann inferior
        nuevo_error = eliptico_normal_penalizacion_v2(
            0.0, 2 * pi, 200, 0.0, 2 * pi, 200, 1.0, u0, un, u2, u3, f0, exacta, True
        )
        # nuevo_error = eliptico_normal_penalizacion(
        #     0.0, 2 * pi, 200, 0.0, 2 * pi, 200, 1.0, u0, u1, u2, u3, f0
        # )
        print("----------------------")

        if error == 0:
            print("ERROR", nuevo_error)
        else:
            print("Cociente de error", (error / nuevo_error))

        error = nuevo_error

    pause(50000)


def eliptico_normal_temporal(
    xi,
    xf,
    Nx,
    yi,
    yf,
    Ny,
    t0,
    tf,
    Nt,
    nu,
    u0,
    g,
    u2,
    u3,
    estado_inicial,
    fuente,
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

    print(f"dx2: {dx2} | dy2: {dy2} | dt: {dt}")

    x = linspace(xi, xf, Nx + 1)
    y = linspace(yi, yf, Ny + 1)
    X, Y = meshgrid(x, y)
    U = estado_inicial(X, Y)

    A = lil_matrix((N, N), dtype="float64")
    Mx_interior = lil_matrix((Nx + 1, Nx + 1), dtype="float64")
    Mx_exterior = lil_matrix((Nx + 1, Nx + 1), dtype="float64")
    My = lil_matrix((Nx + 1, Nx + 1), dtype="float64")

    Mx_interior.setdiag(
        1.0 + 2.0 * (1.0 / (dx2) + 1.0 / (dy2)) * dt * ones(Nx + 1) * nu, 0
    )
    Mx_interior.setdiag(-nu / (dx2) * dt * ones(Nx), 1)
    Mx_interior.setdiag(-nu / (dx2) * dt * ones(Nx), -1)

    Mx_exterior.setdiag(Bn * ones(Nx + 1), 0)

    My.setdiag(-nu / (dy2) * dt * ones(Nx + 1), 0)

    Mx_interior[0, 0] = Bn
    Mx_interior[0, 1] = 0

    Mx_interior[-1, -2] *= 2

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

    usol = U
    error = array(0.0, dtype="float64")

    while t < tf - dt / 2:
        t += dt

        b = fuente(X, Y, t) * dt + U

        # En las esquinas Dirichlet > Neumann (respetamos mejor el orden del metodo)
        b[0, :] = u0(x, t) * Bn
        b[Ny, :] = u2(x, t) * Bn
        b[:, 0] = u3(y, t) * Bn

        # Condiccion tipo neumann en toda la frontera Gamma_1 (lado derecho)
        # En las esquinas, la condiccion de dirichlet tiene prioridad
        b[1:Ny, Nx] += g(y[1:Ny], t) * 2.0 * nu / dx * dt

        b = b.reshape(N)
        usol = LU.solve(b)

        # El usol sera la nueva solucion de u en el instante de tiempo t + dt
        U = usol.reshape((Ny + 1, Nx + 1))

        # Errores
        err = np.max(abs(U - exacta(X, Y, t)).reshape(N))  # type: ignore
        # print("Error espacial cometido:",format(err))

        # El error global E^k = U^k - u(t_k) en el paso k es una acumulacion de los errores locales de los pasos anteriores.
        error = append(error, err)

        if i % 250 == 0 and dibujar:
            usol = usol.reshape((Ny + 1, Nx + 1))
            cu = contourf(X, Y, usol, 20)
            colorbar(cu)
            cl = contour(X, Y, usol, 20, colors="k")
            clabel(cl, inline=1, fontsize=8)
            title(f"Problema contorno 2D, t={round(t, 3)}")

            show()

        # print("Error:", error[-1])

        i += 1

    if dibujar:
        usol = usol.reshape((Ny + 1, Nx + 1))
        cu = contourf(X, Y, usol, 20)
        colorbar(cu)
        cl = contour(X, Y, usol, 20, colors="k")
        clabel(cl, inline=1, fontsize=8)
        title(f"Problema contorno 2D, t={tf}")
        show()

    return max(error)


def f0_t(x, y, t):
    z = sin(x * y) * (1 + x**2 + y**2)
    return z


def exacta_t(x, y, t):
    return sin(x * y)


def estado(x, y):
    return 0 * x * y


def u0_t(x, t):
    z = 0 * x
    return z


def u1_t(y, t):
    z = sin(2 * pi * y)
    return z


def u2_t(x, t):
    z = sin(2 * pi * x)
    return z


def u3_t(x, t):
    z = 0 * x
    return z


def un_t(y, t):
    return y * cos(2 * pi * y)


###################################################################
###################################################################
###################################################################
###################################################################
###################################################################
###################################################################


def fuente(x, y, t):
    # return sin(x * y) * (nu * (x**2 + y**2) * cos(t) - sin(t))
    return sin(x * y) * (0.1 * (x**2 + y**2) * cos(t) - sin(t))


def u0_bc(x, t):  # Frontera y=0
    return zeros_like(x) * t


# def u0_bc(x, t):  # Frontera y=0
# return zeros_like(x) * t  # Multiplicar por t para que la forma sea correcta


def u2_bc(x, t):  # Frontera y=2*pi
    return sin(2 * pi * x) * cos(t)


def u3_bc(y, t):  # Frontera x=0
    return zeros_like(y) * t


def un_bc(y, t):  # Derivada normal en x=2*pi
    return y * cos(2 * pi * y) * cos(t)


def g0_ic(x, y):  # Condicion inicial en t=0
    return sin(x * y)


def exacta(x, y, t):
    return sin(x * y) * cos(t)


# Comprobacion
error = 0

# calculo = [10, 20, 40, 80, 160, 320]
calculo = [100, 200, 400]
# calculo = []

if calculo.__len__() != 0:
    print("2D Contorno ~ Temporal")

    L = 2.0
    T = 1.0

    for i in calculo:
        # fmt: off
        nuevo_error = eliptico_normal_temporal(0.0, 2 * pi, i, 0.0,2 * pi, i, 0.0, 1.0, 1000, 1.0, u0_t, un_t, u2_t, u3_t, estado, f0_t, exacta_t, False)

        # nuevo_error = eliptico_normal_temporal(
        # xi=0, xf=L, Nx=i,
        # yi=0, yf=L, Ny=i,
        # t0=0, tf=T,
        # nu=0.1,
        # u0=u0_bc, g=un_bc, u2=u2_bc, u3=u3_bc,
        # fuente=fuente, estado_inicial=g0_ic,
        # exacta=exacta,
        # dibujar=True
        # )
        # fmt: on
        print("-------------------------")
        if error == 0:
            print("ERROR", nuevo_error)
        else:
            print("Cociente de error", (error / nuevo_error))
        error = nuevo_error
        print("-------------------------")

    pause(50000)


# Para metodo condiccionalmente estable (obtenido por dif.finitas regresivas) la condiccion de estabilidad es:
#   v * dt * (1/dx2 + 1/dy2) <= 1 / 2

# Si 0T < 1/2 es condiccionalmente estable
#   v * dt * (1/dx2 + 1/dy2) <= 1 / (2 * (1 - 2 * 0T))


def eliptico_theta_metodo(
    xi,
    xf,
    Nx,
    yi,
    yf,
    Ny,
    t0,
    tf,
    nu,
    theta,
    u0,
    g,
    u2,
    u3,
    estado_inicial,
    fuente,
    exacta=None,
    dibujar=True,
):
    t_inicio = time.time()

    # --- Parametros y Discretizacion Espacial ---
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
    N = (Nx + 1) * (Ny + 1)

    # --- Discretizacion Temporal ---
    t = float(t0)
    tf = float(tf)

    # Condicion de estabilidad para metodos condicionalmente estables (theta < 0.5)
    if theta < 0.5:
        # La condicion CFL para 2D es dt <= 1 / (2*nu*(1-2*theta)*(1/dx^2 + 1/dy^2))
        dt_max = 1.0 / (2.0 * nu * (1.0 - 2.0 * theta) * (1.0 / dx2 + 1.0 / dy2))
        dt = 0.9 * dt_max  # Usamos un 90% del limite por seguridad
        Nt = int((tf - t0) / dt) + 1
        dt = (tf - t0) / Nt  # Reajustamos dt para llegar exactamente a tf
        print(f"Metodo explicito (theta<0.5). dt: {dt:.6f}")
    else:
        # Para metodos incondicionalmente estables, elegimos un dt razonable.
        Nt = 1000
        dt = (tf - t0) / float(Nt)

    x = linspace(xi, xf, Nx + 1)
    y = linspace(yi, yf, Ny + 1)
    X, Y = meshgrid(x, y)
    U = estado_inicial(X, Y)

    A = lil_matrix((N, N), dtype="float64")
    Mx_interior = lil_matrix((Nx + 1, Nx + 1), dtype="float64")
    Mx_exterior = lil_matrix((Nx + 1, Nx + 1), dtype="float64")
    My = lil_matrix((Nx + 1, Nx + 1), dtype="float64")

    Mx_interior.setdiag(1.0 + 2.0 * nu * dt * (1.0 / dx2 + 1.0 / dy2), 0)
    Mx_interior.setdiag(-nu * dt / dx2, 1)
    Mx_interior.setdiag(-nu * dt / dx2, -1)

    Mx_interior[-1, -2] *= 2

    # Contorno dirichlet
    Mx_exterior.setdiag(Bn, 0)
    My.setdiag(-nu * dt / dy2, 0)

    # Imponemos Dirichlet en la frontera izquierda (x=xi)
    Mx_interior[0, 0] = Bn
    Mx_interior[0, 1] = 0

    # Matriz global A por bloques
    for i in range(1, Ny):
        A[i * (Nx + 1) : (i + 1) * (Nx + 1), i * (Nx + 1) : (i + 1) * (Nx + 1)] = (
            Mx_interior
        )
        A[i * (Nx + 1) : (i + 1) * (Nx + 1), (i - 1) * (Nx + 1) : i * (Nx + 1)] = My
        A[
            i * (Nx + 1) : (i + 1) * (Nx + 1), (i + 1) * (Nx + 1) : (i + 2) * (Nx + 1)
        ] = My

    # Condiciones de contorno (dirichlet) en bordes inferior (y=yi) y superior (y=yf)
    A[0 : (Nx + 1), 0 : (Nx + 1)] = Mx_exterior
    A[Ny * (Nx + 1) : (Ny + 1) * (Nx + 1), Ny * (Nx + 1) : (Ny + 1) * (Nx + 1)] = (
        Mx_exterior
    )

    Id = identity(N, dtype="float64", format="csc")
    A = A.tocsc()

    # Matriz del lado izquierdo: LHS = (I - theta * dt * nu * L)
    LHS_matrix = (1.0 - theta) * Id + theta * A

    # Matriz para el lado derecho: RHS_op = (I + (1-theta) * dt * nu * Laplaciano)
    RHS_matrix_op = (2.0 - theta) * Id - (1.0 - theta) * A

    LU = splu(LHS_matrix)

    i = 0
    usol = U.reshape(N)
    error = [0.0]

    while t < tf - dt / 2:
        tnew = t + dt

        b = RHS_matrix_op * usol

        # 2. Anadir el termino fuente promediado
        fuente_theha = (1.0 - theta) * fuente(X, Y, t) + theta * fuente(X, Y, tnew)
        b += dt * fuente_theha.reshape(N)

        # 3. Imponer las condiciones de contorno en el vector b
        b = b.reshape((Ny + 1, Nx + 1))

        dirichlet = (1.0 - theta) + theta * Bn
        b[0, :] = u0(x, tnew) * dirichlet
        b[Ny, :] = u2(x, tnew) * dirichlet
        b[:, 0] = u3(y, tnew) * dirichlet

        neumann = g(y[1:Ny], t) * (1.0 - theta) + g(y[1:Ny], tnew) * theta
        b[1:Ny, Nx] += neumann * 2.0 * nu / dx * dt

        b = b.reshape(N)
        usol = LU.solve(b)

        U = usol.reshape((Ny + 1, Nx + 1))
        t = tnew

        if exacta:
            err = np.max(abs(U - exacta(X, Y, t)))  # type: ignore
            error.append(err)

        if i % 250 == 0 and dibujar:
            figure(figsize=(10, 8))
            cu = contourf(X, Y, U, 20, cmap="viridis")
            colorbar(cu, label="Valor de u")
            cl = contour(X, Y, U, 20, colors="k", linewidths=0.5)
            clabel(cl, inline=1, fontsize=8)
            title(f"Problema 2D con Theta-Metodo (theta={theta}), t={round(t, 3)}")
            xlabel("x")
            ylabel("y")
            show()

        i += 1

    t_fin = time.time()
    print(f"Tiempo total de ejecucion: {t_fin - t_inicio:.4f} segundos")

    if dibujar:
        figure(figsize=(10, 8))
        cu = contourf(X, Y, U, 20, cmap="viridis")
        colorbar(cu, label="Valor de u")
        cl = contour(X, Y, U, 20, colors="k", linewidths=0.5)
        clabel(cl, inline=1, fontsize=8)
        title(f"Solucion final con Theta-Metodo (theta={theta}), t={tf}")
        xlabel("x")
        ylabel("y")
        show()

    return max(error)


# Comprobacion
error = 0

calculo = [100, 200, 400]
# calculo = []

if calculo.__len__() != 0:
    print("2D Contorno ~ Temporal THEHTA-METODO")

    L = 2.0
    T = 1.0

    for i in calculo:
        # fmt: off
        nuevo_error = eliptico_theta_metodo(0.0, 2 * pi, 200, 0.0,2 * pi, 200, 0.0, 1.0, 1.0, 0.0, u0_t, un_t, u2_t, u3_t, estado, f0_t, exacta_t, True)

        # nuevo_error = eliptico_normal_temporal(
        # xi=0, xf=L, Nx=i,
        # yi=0, yf=L, Ny=i,
        # t0=0, tf=T,
        # nu=0.1,
        # u0=u0_bc, g=un_bc, u2=u2_bc, u3=u3_bc,
        # fuente=fuente, estado_inicial=g0_ic,
        # exacta=exacta,
        # dibujar=True
        # )
        # fmt: on
        print("-------------------------")
        if error == 0:
            print("ERROR", nuevo_error)
        else:
            print("Cociente de error", (error / nuevo_error))
        error = nuevo_error
        print("-------------------------")

    pause(50000)
