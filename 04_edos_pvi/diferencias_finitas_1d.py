"""
Diferencias finitas 1D - contorno lineal y no lineal
Dirichlet homogeneo, malla uniforme y no uniforme, caso -u''+q u=f,
extension no lineal con Newton.
Ver docs/enunciados_resumidos.md#04_edos_pvi
"""


# Resolucion del problema de contorno
# y''(x) = p(x)y'(x)+q(x)y(x) + r(x),
# y(a) = alpha, y(b) = beta,
# (condiciones de tipo Dirichlet)
# mediante el metodo de diferencias finitas.

from pylab import *
from time import perf_counter
from scipy.sparse.linalg import splu
from scipy.sparse import lil_matrix, identity


print("-----------Caso 1 -----------")


def exacta(x):
    return sin(2.0 * pi * x) / (1 + 4 * pi**2)


def funs(x):
    """Funciones p(x), q(x) y f(x)"""
    p = zeros(len(x))
    q = ones(len(x))
    r = -sin(2.0 * pi * x)
    return (p, q, r)


def diffin(a, b, alpha, beta, funs, N):
    h = (b - a) / (N + 1)  # paso de malla
    x = linspace(a, b, N + 2)  # discretizacion del intervalo [a, b]
    (p, q, r) = funs(x)  # coeficientes de la ecuacion
    A = lil_matrix(
        (N, N)
    )  # matriz del sistema en la clase lil (Linked List Matrix) de Scipy

    for i in range(N - 1):
        A[i, i] = 1.0 + 0.5 * q[i + 1] * h**2  # diagonal
        A[i, i + 1] = -0.5 * (1.0 - 0.5 * p[i + 1] * h)  # superdiagonal
        A[i + 1, i] = -0.5 * (1.0 + 0.5 * p[i + 2] * h)  # subdiagonal

    A[N - 1, N - 1] = 1.0 + 0.5 * q[N] * h**2  # ultimo elemento de la diagonal
    F = -(h**2) / 2 * r[1 : N + 1]

    # modificacion del termino independiente
    F[0] += alpha * 0.5 * (1.0 + 0.5 * p[1] * h)
    F[N - 1] += beta * 0.5 * (1.0 - 0.5 * p[N] * h)

    # resolucion del sistema
    A = A.tocsc()  # pone la matriz en formato csc (Compressed Sparse Column)
    LU = splu(A)  # descomposicion LU
    U = LU.solve(F)  # solucion del sistema lineal

    # condiciones de contorno
    U = append(alpha, U)
    U = append(U, beta)
    return x, U


# Datos del problema
a = 0.0  # extremo inferior del intervalo
b = 1.0  # extremo superior del intervalo
alpha = 0.0  # condicion de contorno en a
beta = 0.0  # condicion de contorno en b
N = 10  # numero de particiones (m=N-1)

tini = perf_counter()

x, U = diffin(a, b, alpha, beta, funs, N)

tfin = perf_counter()


# Make so it doennt print always
Ue = exacta(x)  # solucion exacta

error = max(abs(Ue - U))  # error cometido

# Resultados
print("-----")
print("Tiempo CPU: " + str(tfin - tini))
print("Error: " + str(error))
print("Paso de malla: " + str((b - a) / (N + 1)))
print("-----")

# solucion exacta con mayor resolucion
xx = linspace(a, b, 500)
Ue = exacta(xx)

plot(x, U, "b-o")  # dibuja la solucion aproximada
plot(xx, Ue, "r")  # dibuja la solucion exacta
title("Caso 1")
legend(["aproximada", "exacta"], loc="lower left")
grid(True)
show()

# Midamos el orden
errorAntiguo = 0

# calculo = []
calculo = [80, 160, 320, 640]
if calculo.__len__() != 0:
    print("Calculo experimental del orden")
    print("-------------------------")

    for i in calculo:
        x, U = diffin(a, b, alpha, beta, funs, i)
        Ue = exacta(x)

        print("-------------------------")
        nuevo_error = max(abs(Ue - U))  # error cometido

        print(f"Error cometido con N={i} es {nuevo_error} ")

        if errorAntiguo == 0:
            print("ERROR", nuevo_error)
            print("")
        else:
            # Cuando medimos el orden, si hacemos la malla el doble de grande, i.e, dx -> dx/2
            #   entonces por una cuenta inmediata, que ademas se vio explicitamente en clase,
            #   se tiene que si el metodo es de orden p, entonces
            #   el log2 del cociente de errores (el antiguo / partido nuevo) tiene que ser aproximadamente p.
            #   Como se comprueba cuando se ejecuta el codigo, el orden medido experimentalmente es 2 que coincide con los
            #   resultados teoricos.
            print(f"Orden aproximado (N={i}):", log2(errorAntiguo / nuevo_error))

        errorAntiguo = nuevo_error

    # pause(50000)


############ -> CASO 2
print("-----------Caso 2 -----------")

c1 = (
    (pi**4 + 4 * pi**2 + 2)
    / (pi**4 + 3 * pi**2 + 1)
    * (1 - exp(-1 - sqrt(5)))
    / (exp(-1 + sqrt(5)) - exp(-1 - sqrt(5)))
)
c2 = (
    (pi**4 + 4 * pi**2 + 2)
    / (pi**4 + 3 * pi**2 + 1)
    * (exp(-1 + sqrt(5)) - 1)
    / (exp(-1 + sqrt(5)) - exp(-1 - sqrt(5)))
)


def exacta2(x):
    a = (pi**2 + 1) / (pi**4 + 3 * pi**2 + 1)
    b = pi / (pi**4 + 3 * pi**2 + 1)

    return (
        c1 * exp((-1 + sqrt(5)) / 2 * x)
        + c2 * exp((-1 - sqrt(5)) / 2 * x)
        - a * cos(pi * x)
        + b * sin(pi * x)
    )


def funs(x):
    """Funciones p(x), q(x) y f(x)"""
    p = -1 * ones(len(x))
    q = ones(len(x))
    r = cos(pi * x)
    return (p, q, r)


# Datos del problema
a = 0.0  # extremo inferior del intervalo
b = 2.0  # extremo superior del intervalo
alpha = 1.0  # condicion de contorno en a
beta = 1.0  # condicion de contorno en b
N = 100  # numero de particiones (m=N-1)


x, U = diffin(a, b, alpha, beta, funs, N)
Ue = exacta2(x)  # solucion exacta

error = max(abs(Ue - U))  # error cometido

# solucion exacta con mayor resolucion
xx = linspace(a, b, 500)
Ue = exacta2(xx)
plot(x, U, "b-o")  # dibuja la solucion aproximada
plot(xx, Ue, "r")  # dibuja la solucion exacta
title("Caso 2")
show()
legend(["aproximada", "exacta"], loc="lower left")
grid(True)


############ -> CASO 3 | Nodo fantasma
# Nodo fantasma en x=a
print("-----------Caso 3 -----------")


def diffin_nfa(a, b, alpha, beta, funs, N):
    h = (b - a) / (N + 1)
    x = linspace(a, b, N + 2)
    (p, q, r) = funs(x)

    A = lil_matrix((N + 1, N + 1))

    for i in range(N):
        A[i, i] = 1.0 + 0.5 * q[i + 1] * h**2
        A[i, i + 1] = -0.5 * (1.0 - 0.5 * p[i + 1] * h)
        A[i + 1, i] = -0.5 * (1.0 + 0.5 * p[i + 2] * h)
    A[N, N] = 1.0 + 0.5 * q[N] * h**2
    F = -(h**2) / 2 * r[0 : N + 1]

    A[0, 0] = 1 + 1 / 2 * q[0] * h**2
    A[0, 1] = -1

    F[0] += -1 * alpha * h * (1 + p[0] / 2 * h)
    F[N] += beta * 0.5 * (1.0 - 0.5 * p[N] * h)

    # resolucion del sistema
    A = A.tocsc()  # pone la matriz en formato csc
    LU = splu(A)  # descomposicion LU
    U = LU.solve(F)  # solucion del sistema lineal

    # condiciones de contorno
    U = append(U, beta)
    return x, U


def exacta3(x):
    a1 = sin(2 * pi * x) / (1 + 4 * pi * pi)
    a2 = (1 - 2 * pi + 4 * pi * pi) / ((1 + 4 * pi * pi) * (1 + e * e))
    a3 = exp(x) - exp(2 - x)

    return a1 + a2 * a3


def funs(x):
    p = zeros(len(x))
    q = ones(len(x))
    r = -sin(2 * pi * x)
    return (p, q, r)


# Datos del problema
a = 0.0
b = 1.0
alpha = 1.0
beta = 0.0
N = 100

x, U = diffin_nfa(a, b, alpha, beta, funs, N)
Ue = exacta3(x)

error = max(abs(Ue - U))
# print(f"Error caso 3: {error}")

# solucion exacta con mayor resolucion
xx = linspace(a, b, 500)
Ue = exacta3(xx)
plot(x, U, "b-o")  # solucion aproximada
plot(xx, Ue, "r")  # solucion exacta
title("Caso 3")
show()
legend(["aproximada", "exacta"], loc="lower left")
grid(True)


# Midamos el orden
errorAntiguo = 0

# calculo = []
calculo = [80, 160, 320, 640]
if calculo.__len__() != 0:
    print("Caso 3 -> Calculo del orden")
    print("-------------------------")

    for i in calculo:
        x, U = diffin_nfa(a, b, alpha, beta, funs, i)
        Ue = exacta3(x)

        print("-------------------------")
        nuevo_error = max(abs(Ue - U))  # error cometido
        print(f"Error cometido con N={i} es {nuevo_error} ")

        if errorAntiguo == 0:
            print("ERROR", nuevo_error)
        else:
            # El comentario del orden es el mismo que para el caso (1).
            #   Efectivamente se comprueba que es de orden 2.
            print(f"Orden aproximado (N={i}):", log2(errorAntiguo / nuevo_error))

        errorAntiguo = nuevo_error

    # pause(50000)


############ -> CASO 4 | Nodo fantasma
# Nodo fantasma en x=a
print("-----------Caso 4 -----------")


def funs(x):
    p = ones(len(x)) * ((-2 * x) / (1 + x * x))
    q = ones(len(x)) * 1 / (1 + x * x)
    r = (x * x) / (1 + x * x)
    return (p, q, r)


# Datos del problema
a = 0.0
b = 1.0
alpha = 1.0
beta = 0.0
N = 100

x, U = diffin_nfa(a, b, alpha, beta, funs, N)

# error = max(abs(Ue - U))
# print(f"Error caso 3: {error}")

plot(x, U)  # solucion aproximada
title("Caso 4")
show()
legend(["aproximada"], loc="lower left")
grid(True)

# --- Malla no uniforme ---


# Resolucion del problema no lineal
# y''(x) = g(x,y),
# y(a) = alpha, y(b) = beta,
# (condiciones de tipo Dirichlet de contorno)
# mediante el metodo de diferencias finitas.

from pylab import *
from time import perf_counter
from scipy.sparse.linalg import splu
from scipy.sparse import lil_matrix, identity, spdiags


def diffinNoLineal(a, b, alpha, beta, g, DfCentral, Dfz, N, tol, nmax):
    # Mallado
    h = (b - a) / (N + 1)
    x = linspace(a, b, N + 2)
    A = lil_matrix((N, N))

    # Construimos matriz del metodo de diferencias finitas.
    # A = tridiag(-1, 2, -1)
    main_diag = 2 * np.ones(N)
    off_diag = -1 * np.ones(N)
    A = spdiags([off_diag, main_diag, off_diag], [-1, 0, 1], N, N).tocsc()

    # Preparar el bucle
    iter = 0
    usol = zeros((N, 1))  # Semilla inicial nula
    unext = zeros((N, 1))
    error = tol + 1

    G = zeros((N, 1))
    D = lil_matrix((N, N))

    while (iter < nmax) and abs(error) >= tol:
        # El sistema a resolver es DF(U_k) * V_k = F(U_k) para V_k
        print(f"Iter {iter}")

        # Realmente podriamos simplificar el siguiente codigo anadiendo a usol los valores de alfa y beta, y reordenando los indices
        #   del siguiente bucle para que en i=0 e i=N+1 automaticamente se incorporen los valores de contorno. Por supuesto despues
        #   tendriamos que tener cuidado a la hora de calcular el error de no hacerlo sobre los valores de contorno
        for i in range(1, N - 1):
            G[i] = (h**2) * g(x[i + 1], usol[i])

            y_prima = (usol[i + 1] - usol[i - 1]) / (2 * h)

            D[i, i] = 2 + h**2 * DfCentral(x[i + 1], usol[i], y_prima)[i]  # diagonal
            D[i, i + 1] = -1 + h / 2 * Dfz(x[i + 1], usol[i], y_prima)  # superdiagonal
            D[i + 1, i] = -1 - h / 2 * Dfz(x[i + 1], usol[i], y_prima)  # subdiagonal

        D[0, 0] = 2 + h**2 * DfCentral(x[1], usol[0], (usol[1] - alpha) / (2 * h))[0]
        D[0, 1] = -1 + h / 2 * Dfz(x[1], usol[0], (usol[1] - alpha) / (2 * h))
        D[1, 0] = -1 - h / 2 * Dfz(x[1], usol[0], (usol[1] - alpha) / (2 * h))

        D[N - 1, N - 1] = (
            2 + h**2 * DfCentral(x[N], usol[-1], (beta - usol[N - 1]) / (2 * h))[-1]
        )
        D[N - 1, N - 2] = -1 - h / 2 * Dfz(
            x[N], usol[-1], (beta - usol[N - 1]) / (2 * h)
        )

        G[0] = h**2 * g(x[1], usol[0]) - alpha
        G[-1] = h**2 * g(x[-2], usol[-1]) - beta

        F = A * usol + G
        D = D.tocsc()
        LU = splu(D)
        Y = LU.solve(F)
        unext = usol - Y

        error = np.max(abs(usol - unext))
        print(error)

        usol = unext.copy()
        iter += 1

    if iter == nmax:
        print("El metodo no converge desde la semilla inicial seleccionada")

    # condiciones de contorno
    usol = append(alpha, usol)
    usol = append(usol, beta)
    return x, usol


# DfCentral -> Matriz con derivadas f respecto u_i en la posicion i-esima
# Dfz -> Derivada de f respecto y' = z
def diffinNoLinealPrima(a, b, alpha, beta, g, DfCentral, Dfz, N, tol, nmax):
    # Mallado
    h = (b - a) / (N + 1)
    x = linspace(a, b, N + 2)
    A = lil_matrix((N, N))

    # Construimos matriz del metodo de diferencias finitas.
    # A = tridiag(-1, 2, -1)
    main_diag = 2 * np.ones(N)
    off_diag = -1 * np.ones(N)
    A = spdiags([off_diag, main_diag, off_diag], [-1, 0, 1], N, N).tocsc()

    # Preparar el bucle
    iter = 0
    usol = zeros((N + 2))  # Semilla inicial nula
    unext = zeros((N + 2))
    error = tol + 1

    # Imponer condicciones de contorno
    usol[0] = alpha
    usol[-1] = beta

    G = zeros((N))
    D = lil_matrix((N, N))

    while (iter < nmax) and abs(error) >= tol:
        # El sistema a resolver es DF(U_k) * V_k = F(U_k) para V_k
        # Construimos elementos del sistema
        print(f"--> Iter {iter}")

        # for i in range(N):
        #     G[i] = (h**2) * g(x[i + 1], usol[i + 1])
        #     y_prima = (usol[i + 2] - usol[i]) / (2 * h)

        #     D[i, i] = (
        #         2 + h**2 * DfCentral(x[i + 1], usol[i + 1], y_prima)[i]
        #     )  # diagonal

        #     if i != (N - 1):
        #         D[i, i + 1] = -1 + h / 2 * Dfz(
        #             x[i + 1], usol[i + 1], y_prima
        #         )  # superdiagonal
        #         D[i + 1, i] = -1 - h / 2 * Dfz(
        #             x[i + 1], usol[i + 1], y_prima
        #         )  # subdiagonal

        for i in range(N):
            G[i] = (h**2) * g(x[i + 1], usol[i + 1])

            y_prima = (usol[i + 2] - usol[i]) / (2 * h)

            D[i, i] = 2 + (h**2) * DfCentral(x[i + 1], usol[i + 1], y_prima)[i]

            if i < N - 1:
                D[i, i + 1] = -1 + (h / 2) * Dfz(x[i + 1], usol[i + 1], y_prima)

            if i > 0:
                D[i, i - 1] = -1 - (h / 2) * Dfz(x[i + 1], usol[i + 1], y_prima)

        G[0] += -alpha
        G[-1] += -beta

        F = A * usol[1:-1] + G
        LU = splu(D.tocsc())
        Y = LU.solve(F)

        unext = usol[1:-1] - Y
        unext = append(alpha, unext)
        unext = append(unext, beta)

        # No consideramos los puntos de contorno para el error, ya que los imponemos directamente
        error = np.max(abs(Y))
        print(f"error: {error}")

        usol = unext
        iter += 1

    if iter == nmax:
        print("El metodo no converge desde la semilla inicial seleccionada")

    return x, usol


print("-------------- CASO 2 --------------------")
# Datos ii)
N = 100
tol = 10e-6
nmax = 200
a = 0
b = 1
alpha = 0
beta = 0


def g1(x, y):
    return 3 * y + x**2 + 10 * y**3


def dgy(x, y, z):
    s = ones(N) * (3 + 20 * y**2)

    return s


def dgz(x, y, z):
    return 0


x, U = diffinNoLinealPrima(a, b, alpha, beta, g1, dgy, dgz, N, tol, nmax)

plot(x, U)
title("Caso 2")
show()


print("-------------- CASO 3 --------------------")
# Datos iii)
N = 150
tol = 10e-6
nmax = 200
a = 0
b = 2 * pi
alpha = 0.7
beta = 0.7


def g2(x, y):
    return -sin(y)


def dgy2(x, y, z):
    s = ones(N) * (-cos(y))

    return s


def dgz2(x, y, z):
    return 0


x, U = diffinNoLinealPrima(a, b, alpha, beta, g2, dgy2, dgz2, N, tol, nmax)
print(U[-2])
plot(x, U)
title("Caso 3")
show()

# --- Variante final ---


# Resolucion del problema no lineal
# y''(x) = g(x,y),
# y(a) = alpha, y(b) = beta,
# (condiciones de tipo Dirichlet de contorno)
# mediante el metodo de diferencias finitas.

from pylab import *
from time import perf_counter
from scipy.sparse.linalg import splu
from scipy.sparse import lil_matrix, identity, spdiags
import numpy as np


# Df_ui -> Derivada de f respecto u_i
# Dfz -> Derivada de f respecto y' = z
def diffinNoLineal(a, b, alpha, beta, g, Df_y, Dfz, N, tol, nmax):
    # Mallado
    h = (b - a) / (N + 1)
    x = linspace(a, b, N + 2)
    A = lil_matrix((N, N))

    # Construimos matriz del metodo de diferencias finitas.
    # A = tridiag(-1, 2, -1)
    main_diag = 2 * np.ones(N)
    off_diag = -1 * np.ones(N)
    A = spdiags([off_diag, main_diag, off_diag], [-1, 0, 1], N, N).tocsc()

    # Preparar el bucle
    iter = 0
    usol = zeros((N + 2))  # Semilla inicial nula
    unext = zeros((N + 2))
    error = tol + 1

    # Imponer contorno a la solucion
    usol[0] = alpha
    usol[-1] = beta

    G = zeros((N))
    D = lil_matrix((N, N))

    while (iter < nmax) and abs(error) >= tol:
        # El sistema a resolver es DF(U_k) * V_k = F(U_k) para V_k
        # Construimos elementos del sistema
        print(f"--> Iter {iter}")

        for i in range(N):
            # Termino independiente
            G[i] = (h**2) * g(x[i + 1], usol[i + 1])

            # Termino necesario para determinar el jacobiano en cada fila
            y_prima = (usol[i + 2] - usol[i]) / (2 * h)

            D[i, i] = 2 + (h**2) * Df_y(x[i + 1], usol[i + 1], y_prima)

            if i < N - 1:
                D[i, i + 1] = -1 + (h / 2) * Dfz(x[i + 1], usol[i + 1], y_prima)

            if i > 0:
                D[i, i - 1] = -1 - (h / 2) * Dfz(x[i + 1], usol[i + 1], y_prima)

        # Imponemos condiciones de contorno
        G[0] += -alpha
        G[-1] += -beta

        # Resolvemos el sistema lineal
        F = A * usol[1:-1] + G
        LU = splu(D.tocsc())
        Y = LU.solve(F)

        # Nos quedamos con nuestra condicion de contorno
        unext = usol[1:-1] - Y
        unext = append(alpha, unext)
        unext = append(unext, beta)

        # No consideramos los puntos de contorno para el error, ya que los imponemos directamente
        error = np.max(abs(Y))
        print(f"error: {error}")

        usol = unext
        iter += 1

    if iter == nmax:
        print("El metodo no converge desde la semilla inicial seleccionada")

    return x, usol


print("-------------- CASO 2 --------------------")
# Datos ii)
N = 100
tol = 10e-6
nmax = 200
a = 0
b = 1
alpha = 0
beta = 0


def g1(x, y):
    return 3 * y + x**2 + 10 * y**3


def dgy(x, y, z):
    s = 3 + 20 * y**2

    return s


def dgz(x, y, z):
    return 0


x, U = diffinNoLineal(a, b, alpha, beta, g1, dgy, dgz, N, tol, nmax)

plot(x, U)
title("Caso 2")
show()


print("-------------- CASO 3 --------------------")
# Datos iii)
N = 150
tol = 10e-6
nmax = 200
a = 0
b = 2 * pi
alpha = 0.7
beta = 0.7


def g2(x, y):
    return -sin(y)


def dgy2(x, y, z):
    s = -cos(y)

    return s


def dgz2(x, y, z):
    return 0


x, U = diffinNoLineal(a, b, alpha, beta, g2, dgy2, dgz2, N, tol, nmax)
print(U[-2])
plot(x, U)
title("Caso 3")
show()
