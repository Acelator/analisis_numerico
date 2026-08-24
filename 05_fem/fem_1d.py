"""
FEM 1D P1 - elementos finitos lineales
Problema -u'' + q u = f. Matrices de rigidez y masa con integracion
punto medio, malla no uniforme.
Ver docs/enunciados_resumidos.md#05_fem
"""

from pylab import *
from time import perf_counter
from scipy.sparse.linalg import splu
from scipy.sparse import lil_matrix, identity, spdiags, diags


def exacta(x):
    return sin(2.0 * pi * x) / (1 + 4 * pi**2)


def func(x):
    return sin(2.0 * pi * x)


# Datos del problema
a = 0.0  # extremo inferior del intervalo
b = 1.0  # extremo superior del intervalo
alpha = 0.0  # condicion de contorno en a
beta = 0.0  # condicion de contorno en b
N = 30  # numero de particiones (m=N-1)


# def diffin(a, b, alpha, beta, funs, N):


# No sabia (me lo debi de haber imaginado) que numpy tenia una funcion justo para esto
#   Uso la funcion de numpy en practicamente todas las veces que hago dicho calculo
def calcular_malla_particion(part):
    h = np.zeros(len(part) - 1)

    for i in range(0, len(h)):
        h[i] = part[i + 1] - part[i]

    return h


# Part es la particion del intervalo [a,b], no necesariamente uniforme
def fem_k1_homogeneo(a, b, part, alpha, beta, nfa, nfb, q, f):
    assert q > 0
    # Comprobar que al menos los extremos de la particion son los correctos, sino hemos pasado un particion incorrecta
    # Si tengo tiempo implementar checkeo para la particion entera, i.e., puntos crecientes.
    assert part[0] == a and part[-1] == b

    # Quitamos los extremos cuando no trabajemos con nodo fantasma
    hi = calcular_malla_particion(part)
    # hi = np.diff(part)
    # print(len(hi))
    N = len(hi) - 1

    # Construimos la matriz del MEF

    # diags asume no padding en los vectores que se le pasan a diferencia de spdiag
    R = diags(
        [-1 / hi[1:-1], 1 / hi[:-1] + 1 / hi[1:], -1 / hi[1:-1]],
        [-1, 0, 1],
        shape=(N, N),
    ).tocsc()
    M = diags(
        [hi[1:-1] / 6, 1 / 3 * (hi[:-1] + hi[1:]), hi[1:-1] / 6],
        [-1, 0, 1],
        shape=(N, N),
    ).tocsc()

    A = R + q * M

    LU = splu(A)

    # Termino independiente
    F = np.zeros(N)
    for i in range(0, N):
        # Formula punto medio
        F[i] = (
            1
            / 2
            * (
                hi[i] * f(1 / 2 * (part[i] + part[i + 1]))
                + hi[i + 1] * f(1 / 2 * (part[i + 1] + part[i + 2]))
            )
        )

        # Formula trapecio
        # F[i] = 1 / 2 * (hi[i] + hi[i + 1]) * f(part[i + 1])

    U = LU.solve(F)

    # # condiciones de contorno (Si son nulas)
    U = np.append(alpha, U)
    U = np.append(U, beta)
    return U


# x = linspace(a, b, N + 2)  # discretizacion uniforme del intervalo [a, b]

# U = fem_k1_homogeneo(a, b, x, alpha, beta, False, False, 1, func)

# plot(x, U, "*-")
# show()

# Ue = exacta(x)  # solucion exacta
# error = max(abs(Ue - U))  # error cometido
# print(error)


# Part es la particion del intervalo [a,b], no necesariamente uniforme
def fem_k1(a, b, part, alpha, beta, nfa, nfb, q, f):
    assert q > 0
    assert part[0] == a and part[-1] == b

    hi = calcular_malla_particion(part)
    N = len(hi) - 1 + 1 * nfa + 1 * nfb

    main_R = 1 / hi[:-1] + 1 / hi[1:]
    main_M = (hi[:-1] + hi[1:]) / 3

    if nfa:
        main_R = np.insert(main_R, 0, 1 / hi[0])
        main_M = np.insert(main_M, 0, hi[0] / 3)

    if nfb:
        main_R = np.append(main_R, 1 / hi[-1])
        main_M = np.append(main_M, hi[-1] / 3)

    off_R = -1 / hi
    off_M = hi / 6

    if not nfa:
        off_R = off_R[1:]
        off_M = off_M[1:]
    if not nfb:
        off_R = off_R[:-1]
        off_M = off_M[:-1]

    R = diags([off_R, main_R, off_R], [-1, 0, 1], shape=(N, N)).tocsc()
    M = diags([off_M, main_M, off_M], [-1, 0, 1], shape=(N, N)).tocsc()

    A = R + q * M
    LU = splu(A)

    # Termino independiente
    F = np.zeros(len(hi) - 1)
    for i in range(len(hi) - 1):
        # Formula del punto medio
        F[i] = (
            1
            / 2
            * (
                hi[i] * f(1 / 2 * (part[i] + part[i + 1]))
                + hi[i + 1] * f(1 / 2 * (part[i + 1] + part[i + 2]))
            )
        )

    if nfa:
        F = np.insert(F, 0, -alpha + 1 / 2 * hi[0] * f(1 / 2 * (part[0] + part[1])))
    else:
        A_n0_n1 = -1 / hi[0] + q * (hi[0] / 6)
        F[0] -= alpha * A_n0_n1

    if nfb:
        F = np.append(F, beta + 1 / 2 * hi[-1] * f(1 / 2 * (part[-2] + part[-1])))
    else:
        A_nhp1_nh = -1 / hi[-1] + q * (hi[-1] / 6)
        F[-1] -= beta * A_nhp1_nh

    U = LU.solve(F)

    if not nfa:
        U = np.insert(U, 0, alpha)
    if not nfb:
        U = np.append(U, beta)

    return U


def exacta3(x):
    a1 = sin(2 * pi * x) / (1 + 4 * pi * pi)
    a2 = (1 - 2 * pi + 4 * pi * pi) / ((1 + 4 * pi * pi) * (1 + e * e))
    a3 = exp(x) - exp(2 - x)

    return a1 + a2 * a3


# Midamos el orden
errorAntiguo = 0

# calculo = []
calculo = [10, 20, 40, 80, 160]
if calculo.__len__() != 0:
    print("Caso 1")
    print("-------------------------")

    for i in calculo:
        x = linspace(a, b, i + 2)
        U = fem_k1(a, b, x, alpha, beta, False, False, 1, func)
        Ue = exacta(x)

        nuevo_error = max(abs(Ue - U))  # error cometido
        print(f"error: {nuevo_error}")

        if errorAntiguo == 0:
            print("ERROR", nuevo_error)
        else:
            print("Orden aproximado", log2(errorAntiguo / nuevo_error))

        print("-------------------------")
        errorAntiguo = nuevo_error

    # pause(50000)


# alpha = 1
# N = 50
# x = linspace(a, b, N + 2)  # discretizacion uniforme del intervalo [a, b]

# U = fem_k1(a, b, x, alpha, beta, True, False, 1, func)
# plot(x, U)
# grid(True)
# show()

# alpha = 0
# U = fem_k1(a, b, x, alpha, beta, False, False, 1, func)
# plot(x, U, "*-")
# grid(True)
# show()

# Ue = exacta3(x)  # solucion exacta


# error = max(abs(Ue - U))  # error cometido
# print(error)

# /-----------------------------------------
# ------------------------------------------
# ------------------------------------------
#           Caso 2 | R3
# ------------------------------------------
# ------------------------------------------
# -----------------------------------------/


# Part es la particion del intervalo [a,b], no necesariamente uniforme
def ejer2(a, b, part, alpha, beta, nfa, nfb, q, p, f):
    assert q > 0
    # Comprobar que al menos los extremos de la particion son los correctos, sino hemos pasado un particion incorrecta
    # Si tengo tiempo implementar checkeo para la particion entera, i.e., puntos crecientes.
    assert part[0] == a and part[-1] == b

    # Quitamos los extremos cuando no trabajemos con nodo fantasma
    hi = calcular_malla_particion(part)
    # print(len(hi))
    N = len(hi) - 1

    # Diagonales Principales
    main_R = 1 / hi[:-1] + 1 / hi[1:]
    main_M = (hi[:-1] + hi[1:]) / 3

    # Diagonales Secundarias
    off_R = -1 / hi[:-1]
    off_M = hi[:-1] / 6

    R = diags([off_R, main_R, off_R], [-1, 0, 1], shape=(N, N)).tocsc()
    M = diags([off_M, main_M, off_M], [-1, 0, 1], shape=(N, N)).tocsc()

    # Calculamos x en el mallado
    x_mid = (part[:-1] + part[1:]) / 2.0

    # He calculado la matriz de D con la formula del punto medio. Como en nuestro caso p == 1 y lo que tenemos dentro de la integral
    #   es un polinomio de primer orden es exacto ya que la formula del punto medio es exacta para grado 1.
    # Para que el codigo funcione independientemente de la funcion p, evaluo en el punto medio
    # Si la funcion dista de ser lineal, entonces la aproximacion sera de mucha peor calidad.
    p_mid = p(x_mid)

    # La matriz D es antisimetrica
    D = diags(
        [
            -0.5 * p_mid[1:-1],  # Diagonal inferior
            0.5 * (p_mid[:-1] - p_mid[1:]),  # Diagonal principal
            0.5 * p_mid[1:-1],
        ],  # Diagonal superior
        [-1, 0, 1],
        shape=(N, N),
    ).tocsc()

    A = R + q * M + D
    LU = splu(A)

    # Termino independiente
    F = np.zeros(N)
    # Bucle para los nodos interiores
    for i in range(N):
        F[i] = (
            1
            / 2
            * (
                hi[i] * f(1 / 2 * (part[i] + part[i + 1]))
                + hi[i + 1] * f(1 / 2 * (part[i + 1] + part[i + 2]))
            )
        )

    RqM_n0_n1 = -1 / hi[0] + q * (hi[0] / 6)
    D_n0_n1 = -0.5 * p_mid[0]
    F[0] += -alpha * (RqM_n0_n1 + D_n0_n1)

    RqM_nhp1_nh = -1 / hi[-1] + q * (hi[-1] / 6)
    D_nhp1_nh = 0.5 * p_mid[-1]
    F[-1] += -beta * (RqM_nhp1_nh + D_nhp1_nh)

    U = LU.solve(F)

    # Imponemos condiciones de contorno (Si son nulas)
    U = append(alpha, U)
    U = append(U, beta)

    return U


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


def func(x):
    return -cos(pi * x)


# Datos del problema
a = 0.0  # extremo inferior del intervalo
b = 2.0  # extremo superior del intervalo
alpha = 1.0  # condicion de contorno en a
beta = 1.0  # condicion de contorno en b
N = 100  # numero de particiones (m=N-1)
x = linspace(a, b, N + 2)  # discretizacion uniforme del intervalo [a, b]

q = 1


def p(x):
    return -1 * ones_like(x)


# Midamos el orden
errorAntiguo = 0

# calculo = []
calculo = [10, 20, 40, 80, 160]
if calculo.__len__() != 0:
    print("Caso 2")
    print("-------------------------")

    for i in calculo:
        x = linspace(a, b, i + 2)
        U = ejer2(a, b, x, alpha, beta, False, False, q, p, func)
        Ue = exacta2(x)

        nuevo_error = max(abs(Ue - U))  # error cometido
        print(f"error: {nuevo_error}")

        if errorAntiguo == 0:
            print("ERROR", nuevo_error)
        else:
            print("Orden aproximado", log2(errorAntiguo / nuevo_error))

        print("-------------------------")
        errorAntiguo = nuevo_error

    # pause(50000)


# error = max(abs(Ue - U))  # error cometido
# print(f"Error caso2 {error}")

# # solucion exacta con mayor resolucion
# xx = linspace(a, b, 500)
# plot(x, U, "b-o")  # dibuja la solucion aproximada
# plot(x, Ue, "r")  # dibuja la solucion exacta
# title("Caso 2")
# legend(["aproximada", "exacta"], loc="lower left")
# grid(True)
# show()


# /-----------------------------------------
# ------------------------------------------
# ------------------------------------------
# ------------------------------------------
# -----------------------------------------/
# K2


def indice(i, j, deg):
    return i * deg + j


def fem_k2_dirichlet(a, b, part, alpha, beta, nfa, nfb, q, f):
    assert q >= 0
    assert part[0] == a and part[-1] == b

    hi = np.diff(part)
    Ne = len(hi)

    # Numero de nodos interiores: Total (2*Ne + 1) - 2 extremos = 2*Ne - 1
    N = 2 * Ne - 1

    # La matriz de los productos de los elementos de la base con todas las combinaciones posibles
    #   Representan nodos i i+1/2 i+1, en ese orden tanto por filas como por columnas.
    #   La calcule usando la formula de cuadratura de gauss para 3 puntos, ya que es exacta de polinomios de hasta grado 5, aunque aqui solo trabajamos como mucho de grado 4.
    Mi = np.array([[4, 2, -1], [2, 16, 2], [-1, 2, 4]], dtype=float)
    Ki = np.array([[7, -8, 1], [-8, 16, -8], [1, -8, 7]], dtype=float)

    # Matrices que nos construyen la matriz Ah del sistema
    M = lil_matrix((N, N))
    K = lil_matrix((N, N))
    F = np.zeros(shape=(N, 1), dtype=float)

    # Las matrices anteriores de Mi y Ki son para el intervalo [x_i, x_i+1]
    #   Cuando i=0, al ser condicciones dirichlet, el primer elemento de la base es \phi_{1/2}
    #   Por eso tenemos que "truncar" la matriz anterior por la submatriz 2x2 que representa dichos elementos
    h0 = hi[0]

    # Indices en la matriz completa del sistema donde queremos meter los valores
    J_glob = [0, 1]

    # Nos quedamos con el producto de las funciones phi_1/2 y phi_1, por eso suprimimos la primera fila y columna de tanto M como K.
    J_loc = [1, 2]

    # Inyectamos el bloque 2x2 inferior derecho de la matriz local
    K[np.ix_(J_glob, J_glob)] += Ki[np.ix_(J_loc, J_loc)] / (3 * h0)
    M[np.ix_(J_glob, J_glob)] += Mi[np.ix_(J_loc, J_loc)] * h0 / 30

    # Termino independiente
    #   Restamos parte correspondiente a no homogeneo. Es (- alpha * (a(0, 1/2) + a(0,1)))
    # Aproximamos la integral int(f * phi_i) en (a,b) por la formula de simpson de 3 puntos
    F[0, 0] += (4 * h0 / 6) * f((part[0] + part[1]) / 2.0) - alpha * (
        (q * (Mi[1, 0] * h0 / 30) + Ki[1, 0] / (3 * h0))
        + (q * (Mi[2, 0] * h0 / 30) + Ki[2, 0] / (3 * h0))
    )
    F[1, 0] += (h0 / 6) * f(part[1])

    for i in range(1, Ne - 1):
        h = hi[i]
        x_izq = part[i]
        x_der = part[i + 1]
        x_cen = (x_izq + x_der) / 2.0

        # Matriz de elementos que solapan
        #   Para realizar el metodo de ensamblado. Realmente solo se "intersecan" los elementos centrales para i real
        I = np.array([2 * i - 1, 2 * i, 2 * i + 1])

        K[np.ix_(I, I)] += Ki / (3 * h)
        M[np.ix_(I, I)] += Mi * h / 30

        F[I[0], 0] += (h / 6) * f(x_izq)
        F[I[1], 0] += (4 * h / 6) * f(x_cen)
        F[I[2], 0] += (h / 6) * f(x_der)

    # Igual que hicimos antes (para el primer phi_1/2), pero para i = Ne - 1
    i_last = Ne - 1
    hL = hi[i_last]

    J_glob_L = [2 * i_last - 1, 2 * i_last]
    J_loc_L = [0, 1]

    K[np.ix_(J_glob_L, J_glob_L)] += Ki[np.ix_(J_loc_L, J_loc_L)] / (3 * hL)
    M[np.ix_(J_glob_L, J_glob_L)] += Mi[np.ix_(J_loc_L, J_loc_L)] * hL / 30

    F[J_glob_L[0], 0] += (hL / 6) * f(part[i_last])
    F[J_glob_L[1], 0] += (4 * hL / 6) * f(
        (part[i_last] + part[i_last + 1]) / 2.0
    ) - beta * (
        (Ki[1, 2] / (3 * hL) + q * Mi[1, 2] * hL / 30)  # Corresponde a (a(N - 1/2, N))
        + (
            Ki[0, 2] / (3 * hL) + q * Mi[0, 2] * hL / 30
        )  # Corresponde a (a(N - 1,   N))
    )

    A = (K + q * M).tocsc()
    LU = splu(A)

    # Resolvemos el sistema interior
    U = LU.solve(F)

    # Convertimos a 1D
    U = U.flatten()

    # Anadimos las fronteras dirichlet
    U = np.append(alpha, U)
    U = np.append(U, beta)

    return U


def fem_k2(a, b, part, alpha, beta, nfa, nfb, q, f):
    assert q >= 0
    assert part[0] == a and part[-1] == b

    hi = np.diff(part)
    Ne = len(hi)

    # Base (interiores): 2*Ne - 1
    # mas uno por cada condicion neumann por tener que calcular dicho valor de contorno
    N = 2 * Ne - 1 + 1 * nfa + 1 * nfb

    Mi = np.array([[4, 2, -1], [2, 16, 2], [-1, 2, 4]], dtype=float)
    Ki = np.array([[7, -8, 1], [-8, 16, -8], [1, -8, 7]], dtype=float)

    # Matrices que nos construyen la matriz Ah del sistema
    M = lil_matrix((N, N))
    K = lil_matrix((N, N))
    F = np.zeros(shape=(N, 1), dtype=float)

    # Controla si el primer nodo interior empareja con la fila 0 de la matriz global o con la fila 1
    shift = 0 if nfa else -1

    for i in range(Ne):
        h = hi[i]
        x_izq = part[i]
        x_der = part[i + 1]
        x_cen = (x_izq + x_der) / 2.0

        K_loc = Ki / (3 * h)
        M_loc = Mi * h / 30

        J_loc = [0, 1, 2]
        J_glob = [2 * i + shift, 2 * i + 1 + shift, 2 * i + 2 + shift]

        # Interpolamos f(x) con P2 y lo multiplicamos por la matriz de masa exacta
        f_vec = np.array([f(x_izq), f(x_cen), f(x_der)])
        M_loc_pura = Mi * h / 30
        F_loc = M_loc_pura @ f_vec

        # # Aproximamos la integral int(f * phi_i) por la formula de Simpson
        # F_loc = np.zeros(3)
        # F_loc[0] = (h / 6) * f(x_izq)
        # F_loc[1] = (4 * h / 6) * f(x_cen)
        # F_loc[2] = (h / 6) * f(x_der)

        # Imposicion de condiciones de contorno en x=a
        if i == 0:
            if nfa:
                # Sumamos la condicion no homogenea
                F_loc[0] -= alpha
            else:
                # Nos queda en el desarrollo -alpha * (a(0, 1/2) + a(0, 1))
                F_loc[1] -= alpha * (K_loc[1, 0] + q * M_loc[1, 0])
                F_loc[2] -= alpha * (K_loc[2, 0] + q * M_loc[2, 0])

                J_loc = J_loc[1:]
                J_glob = J_glob[1:]

        # Imposicion de condiciones de contorno en x=b
        if i == Ne - 1:
            if nfb:
                F_loc[2] += beta
            else:
                # Nos queda en el desarrollo -alpha * (a(0, 1/2) + a(0, 1))
                F_loc[0] -= beta * (K_loc[0, 2] + q * M_loc[0, 2])
                F_loc[1] -= beta * (K_loc[1, 2] + q * M_loc[1, 2])

                J_loc = J_loc[:-1]
                J_glob = J_glob[:-1]

        # Inyectamos submatriz local
        K[np.ix_(J_glob, J_glob)] += K_loc[np.ix_(J_loc, J_loc)]
        M[np.ix_(J_glob, J_glob)] += M_loc[np.ix_(J_loc, J_loc)]

        for k_idx, j_act in enumerate(J_loc):
            F[J_glob[k_idx], 0] += F_loc[j_act]

    A = (K + q * M).tocsc()
    LU = splu(A)
    U = LU.solve(F)

    # Convertimos a 1D
    U = U.flatten()

    # Anadimos las fronteras si son Dirichlet
    if not nfa:
        U = np.insert(U, 0, alpha)
    if not nfb:
        U = np.append(U, beta)

    return U


N = 100
a = 0
alpha = 0
b = 1
beta = 0


# b = 8 / pi
# beta = sin(16) / (1 + 4 * pi**2)

x = linspace(a, b, N + 2)
U = fem_k2_dirichlet(a, b, x, alpha, beta, False, False, 1, func)
# Hacer que ya la propia funcion lo devuelva correctamente
U = np.ravel(U)


# Pintar todos los nodos, incluyendo puntos medios
def refinar_con_puntos_medios(x):
    x = np.asarray(x)
    xf = np.empty(2 * len(x) - 1)

    xf[0::2] = x
    xf[1::2] = 0.5 * (x[:-1] + x[1:])

    return xf


xf = refinar_con_puntos_medios(x)


errorAntiguo = 0

# calculo = []
calculo = [10, 20, 40, 80, 160]
if calculo.__len__() != 0:
    print("Caso 1 / P2")
    print("-------------------------")

    # Datos del problema
    a = 0.0
    b = 1.0
    alpha = 0.0
    beta = 0.0

    for i in calculo:
        x = linspace(a, b, i + 2)
        U = fem_k2(a, b, x, alpha, beta, False, False, 1, func)

        idx = np.arange(0, len(U), 2)

        # xf = refinar_con_puntos_medios(x)
        Ue = exacta(x)

        nuevo_error = max(abs(Ue - U[idx]))  # error cometido
        print(f"error: {nuevo_error}")

        # ! DA muy mal lo arreglare cuando pueda y vea que problema hay
        if errorAntiguo == 0:
            print("ERROR", nuevo_error)
        else:
            print("Orden aproximado", log2(errorAntiguo / nuevo_error))

        print("-------------------------")
        errorAntiguo = nuevo_error

    # pause(50000)


# figure()
# plt.plot(x, U[idx], "o-")
# plt.grid(True)
# plt.xlabel("x")
# plt.ylabel("U (1,3,5...)")
# plt.show()


# Pintar solo nodos originales
idx = np.arange(0, len(U), 2)

# Calculo error
Ue = exacta(x)  # solucion exacta
error = max(abs(Ue - U[idx]))  # error cometido
print(error)


plt.figure()
plt.title("K2 | Dirichlet homogeneo")
plt.plot(x, U[idx], "o-")
plt.grid(True)
plt.show()


plt.figure()
plt.plot(xf, U, "o-")
plt.title("K2 | Dirichlet homogeneo | Todos los nodos")
plt.grid(True)
plt.xlabel("x")
plt.ylabel("U")
plt.show()
