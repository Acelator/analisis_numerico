"""
Algebra lineal numerica - metodos directos e iterativos
Normas, condicionamiento, Gauss con pivote, LU/Cholesky,
Jacobi, Gauss-Seidel, SOR y metodo de la potencia.
Ver docs/enunciados_resumidos.md#07_algebra_lineal
"""

# Requisitos previos

# -*- coding: utf-8 -*-

from numpy import *
from numpy.linalg import *
from numpy import abs, sum, max, min


# Matriz de tipo predefinido
# B = array([1,2], [3,4], dtype=complex)

# Matriz identidad
# I           = eye(4)        # Size 4x4
# IPrime      = eye(4,2)      # Size 4x2 con 1 en la diagonal
# IPrimePrime = eye(4, k=1)   # Size 4x4 con 1 en la fila k-esima por encima de la diagonal (si k < 0 entonces por debajo)

# linespace permite crear un vector por interpolacion: linspace(0,1,5) crea el vector [0., 1., 2., 3., 4.]
# rehshape crea matrices, tomando como primer parametro un vector, y segundo el tamano de la matriz a generar

# Para acceder a los elementos de un array podemos usar la notacion a:b:c considerando todos los indices entre a y b con el primero incluido y el segundo excluido con salta de c.
#   La ausencia de a implica tomar todos los elementos desde el primer, mientras que la ausencia de b que se consideran los elementos desde a hasta el ultimo.
#   La presencia de esta notacion en ambos indices genera una matriz, mientras que si unicamente es empleada en uno de los indices, se generara un vector.


# Para conjugar una matriz es necesario emplear las funciones conjugate y transpose simultaneamente, ya que conjugate unicamente conjuga los elementos. Tambien podemos usar la funcion conjugada de este archivo
# Para multiplicar matrices utilizamos el operador "@"
# Las funciones tril() y triu() extraen la parte triangular inferior y superior, respectivamente, de la matriz que se da como argumento; admitiendo un argumento opcional k, que permite anadir o suprimir extradiagonales segun signo.

# La funcion eig() toma como valor de entrada una matriz y tiene por salida dos argumentos, el primero un vector conteniendo a los autovalores de la matriz en un orden arbitrario, y una matriz cuadradada cuya columnas son autovectores
#   asociado a los autovalores en el mismo orden
# La funcion svd() toma una matriz y devuelve tres argumentos, el segundo un vector con los valores singulares de la matriz de entrada, y los otros dos argumentos de salida son la descomposicion en valores singulares de la matriz original en matrices ortogonales-unitarias
# matriz_rank() nos proporciona el rango de la matriz
# La funcion qr() toma una matriz y devuelve en el primero de sus dos argumentos, la ortonormalizacion de G-S de los vectores columnas de la matriz dada

# axis=0 por columnas | axis=1 por filas

# Calcular la inversa de una matriz se resume a resolver el sistema lineal AX=I con I la matriz identidad correspondiente (equiv a usar funcion inv() )

# `time` permite medir tiempos de ejecucion (se debe importar del modulo time). Llamandola devuelve la hora, con lo que no es mas que realizar una resta para obtener el tiempo de ejecucion.
# `max` debe usarse tal que max([]) con los elementos a considerar dentro de los corchetes

# Funciones


# Seccion 1
def conjugada(A):
    if ndim(A) == 1:
        A = array([A])
    return conjugate(transpose(A))


# Seccion 2
def norma_vec(X, p):
    X = array(X, dtype=complex)

    # Comprobar si X es un vector
    if ndim(X) != 1:
        if shape(X)[1] != 1:
            return "Parametro no es un vector"

    normaInf = max(abs(X))

    if p == inf:
        return normaInf
    elif p >= 1:
        return normaInf * sum(abs(X) ** p) ** (1 / p)
    else:
        return "Error norma_vec: p no valido"


def conv_norma_vec(X):
    print("Vector: X = ", X)
    normainf = norma_vec(X, inf)

    # Si la norma infinito es 0 es claro que todas las componentes son 0. Luego toda norma p valdra 0
    if normainf == 0 or normainf <= e - 100:
        print("La norma converge a 0")
        return

    print("||X||_inf = ", normainf)
    error = 1.0
    p = 0
    while error >= 1e-10 and p < 200:
        p = p + 1
        normap = norma_vec(X, p)
        error = abs((normap - normainf) / normainf)
        print("p = ", p, " ||X||_p = ", normap, " Error relativo = ", error)
    if error < 1e-10:
        print("Convergencia numerica alcanzada.")
    else:
        print("Numero maximo de iteraciones alcanzado.")


def norma_mat(A, p):
    if ndim(A) == 2:
        (m, n) = shape(A)
    if ndim(A) != 2 or m != n:
        return "Error norma_mat: Matriz entrada no compatible."
    A = array(A, dtype=complex)
    if p == inf:
        return max(sum(abs(A), axis=1))
    elif p == 1:
        return max(sum(abs(A), axis=0))
    elif p == 2:
        return max((svd(A)[1]))
    elif p == "fro":
        return sqrt(sum(abs(A) ** 2))
    else:
        return "Error norma_mat: valor de p."


# Condicionamiento de una cierta matriz A respecto la norma matricial subordinada ||.||_p
def cond(A, p):
    return norma_mat(A, p) * norma_mat(inv(A), p)


# Seccion 4 - Resolucion sistemas de ecuaciones
def descenso(A, B):
    m, n = shape(A)
    p, q = shape(B)
    if m != n or n != p or q != 1:
        return False, "Error descenso: error en las dimensiones."
    if min(abs(diag(A))) < 1e-200:
        return False, "Error descenso: matriz singular."
    if A.dtype == complex or B.dtype == complex:
        X = zeros((n, 1), dtype=complex)
    else:
        X = zeros((n, 1), dtype=float)
    for i in range(n):
        X[i, 0] = B[i, 0]
        if i != 0:
            X[i, 0] -= A[i, :i] @ X[:i, 0]
        X[i, 0] = X[i, 0] / A[i, i]
    return True, X


# Version mejorada, permite resolver simultaneamente varios sistemas lineales, AX=B todos con la misma matriz y diferentes segundos miembros
def descenso(A, B):
    m, n = shape(A)
    p, q = shape(B)
    if m != n or n != p or q < 1:
        return False, "Descenso - error : error en las dimensiones."
    if min(abs(diag(A))) < 1e-200:
        return False, "Descenso - error : matriz singular."
    if A.dtype == complex or B.dtype == complex:
        X = zeros((m, q), dtype=complex)
    else:
        X = zeros((m, q), dtype=float)
    for i in range(n):
        X[i, :] = B[i, :]
        if i != 0:
            X[i, :] -= A[i, :i] @ X[:i, :]
        X[i, :] = X[i, :] / A[i, i]
    return True, X


# Version mejorada, permite resolver simultaneamente varios sistemas lineales
def remonte(A, B):
    m, n = shape(A)
    p, q = shape(B)

    if m != n or n != p or q < 1:
        return False, "Error remonte: error en las dimensiones."

    if min(abs(diag(A))) < 1e-200:
        return False, "Error remonte: matriz singular."

    if A.dtype == complex or B.dtype == complex:
        X = zeros((n, q), dtype=complex)
    else:
        X = zeros((n, q), dtype=float)

    for i in range(n - 1, -1, -1):
        X[i, :] = B[i, :]
        if i != n - 1:
            X[i, :] -= A[i, i + 1 :] @ X[i + 1 :, :]
        X[i, :] = X[i, :] / A[i, i]

    return True, X


# Ambas suponen que los elementos diagonales son 1 (No se comprueba, unicamente se asume)
def descenso1(A, B):
    m, n = shape(A)
    p, q = shape(B)

    if m != n or n != p or q < 1:
        return False, "Error descenso: error en las dimensiones."
    if A.dtype == complex or B.dtype == complex:
        X = zeros((m, q), dtype=complex)
    else:
        X = zeros((m, q), dtype=float)
    for i in range(n):
        X[i, :] = B[i, :]
        if i != 0:
            X[i, :] -= A[i, :i] @ X[:i, :]
    return True, X


def remonte1(A, B):
    m, n = shape(A)
    p, q = shape(B)

    if m != n or n != p or q < 1:
        return False, "Error remonte: error en las dimensiones."
    if A.dtype == complex or B.dtype == complex:
        X = zeros((m, q), dtype=complex)
    else:
        X = zeros((m, q), dtype=float)
    for i in range(n - 1, -1, -1):
        X[i, :] = B[i, :]
        if i != n - 1:
            X[i, :] -= A[i, i + 1 :] @ X[i + 1 :, :]
    return True, X


# Se asumen que la matriz A es tridiagonal, i.e unicamente la diagonal y una fila por arriba o por abajo sera no nula (dependiendo del tipo de matriz triangular)
def descenso_1diag(A, B):
    m, n = shape(A)
    p, q = shape(B)

    if m != n or n != p or q < 1:
        return False, "Error descenso: error en las dimensiones."

    if min(abs(diag(A))) < 1e-200:
        return False, "Error descenso: matriz singular."
    if A.dtype == complex or B.dtype == complex:
        X = zeros((m, q), dtype=complex)
    else:
        X = zeros((m, q), dtype=float)

    for i in range(r + 1):
        X[i, :] = B[i, :]
        if i != 0:
            X[i, :] -= A[i, i - 1] * X[i - 1, :]
        X[i, :] = X[i, :] / A[i, i]
    return True, X


def remonte_1diag(A, B):
    m, n = shape(A)
    p, q = shape(B)

    if m != n or n != p or q < 1:
        return False, "Error remonte: error en las dimensiones."
    if min(abs(diag(A))) < 1e-200:
        return False, "Error remonte: matriz singular."

    if A.dtype == complex or B.dtype == complex:
        X = zeros((m, q), dtype=complex)
    else:
        X = zeros((m, q), dtype=float)

    for i in range(n - 1, -1, -1):
        X[i, :] = B[i, :]
        if i != n - 1:
            X[i, :] -= A[i, i + 1] * X[i + 1, :]
        X[i, :] = X[i, :] / A[i, i]
    return True, X


# r es el parametro que representa el tamano de la semi-anchura de la banda
def descenso_rdiag(A, B, r):
    m, n = shape(A)
    p, q = shape(B)

    if m != n or n != p or q < 1 or r >= n:
        return False, "Descenso - err : Dimensiones no compatibles."
    if min(abs(diag(A))) < 1e-200:
        return False, "Descenso - error : matriz singular."

    if A.dtype == complex or B.dtype == complex:
        X = zeros((m, q), dtype=complex)
    else:
        X = zeros((m, q), dtype=float)

    for i in range(n):
        X[i, :] = B[i, :]
        if i != 0:
            X[i, :] -= A[i, max([i - r, 0]) : i] @ X[max([i - r, 0]) : i, :]
        X[i, :] = X[i, :] / A[i, i]
    return True, X


def remonte_rdiag(A, B, r):
    m, n = shape(A)
    p, q = shape(B)

    if m != n or n != p or q < 1:
        return False, "Error remonte: error en las dimensiones."
    if min(abs(diag(A))) < 1e-200:
        return False, "Error remonte: matriz singular."

    if A.dtype == complex or B.dtype == complex:
        X = zeros((n, q), dtype=complex)
    else:
        X = zeros((n, q), dtype=float)

    for i in range(n - 1, -1, -1):
        X[i, :] = B[i, :]
        if i != n - 1:
            X[i, :] -= (
                A[i, i + 1 : min([i + r + 1, n])] @ X[i + 1 : min([i + r + 1, n]), :]
            )
        X[i, :] = X[i, :] / A[i, i]
    return True, X


# Es habitual que para el almacenamiento de la misma en la memoria del ordenador, no se utilice una matriz de tamano nxn (cuyos elementos serian en su mayoria 0), sino que se utilice una matriz optimizada de tamano nx2, en cuya segunda columna se almacenan los elementos de la diagonal principal conservando la fila, y en cuya primera columna se almacenan los elementos de la sub-diagonal tambien conservando la fila (el valor almacenada en la primera fila y primera columna de esta matriz optimizada no tiene ninguna utilidad en este caso). En el caso de matrices triangulares superiores se almacena la diagonal principal en la primera columna, mientras que en la segunda columna se almacenan los elementos de la supra-diagonal, siempre conservando la fila
def descenso_1diag_vacia(A, B):
    m, n = shape(A)
    p, q = shape(B)

    if m != n or n != p or q < 1:
        return False, "Error descenso: error en las dimensiones."
    if min(abs(diag(A))) < 1e-200:
        return False, "Error descenso: matriz singular."

    if A.dtype == complex or B.dtype == complex:
        X = zeros((n, q), dtype=complex)
        Y = zeros((n, 2), dtype=complex)
    else:
        X = zeros((n, q), dtype=float)
        Y = zeros((n, 2), dtype=float)

    for i in range(n):
        if i != 0:
            Y[i, 0] = A[i, i - 1]
            Y[i, 1] = A[i, i]
        else:
            Y[i, 1] = A[i, i]

    for i in range(n):
        X[i, :] = B[i, :]
        if i != 0:
            X[i, :] -= Y[i, 0] * X[i - 1, :]
        X[i, :] = X[i, :] / Y[i, 1]
    return True, X


def remonte_1diag_vacia(A, B):
    m, n = shape(A)
    p, q = shape(B)

    if m != n or n != p or q < 1:
        return False, "Error remonte: error en las dimensiones."
    if min(abs(diag(A))) < 1e-200:
        return False, "Error remonte: matriz singular."

    if A.dtype == complex or B.dtype == complex:
        X = zeros((n, q), dtype=complex)
        Y = zeros((n, 2), dtype=complex)
    else:
        X = zeros((n, q), dtype=float)
        Y = zeros((n, 2), dtype=float)

    for i in range(n):
        if i != n - 1:
            Y[i, 1] = A[i, i + 1]
            Y[i, 0] = A[i, i]
        else:
            Y[i, 0] = A[i, i]

    for i in range(n - 1, -1, -1):
        X[i, :] = B[i, :]
        if i != n - 1:
            X[i, :] -= Y[i, 1] * X[i + 1, :]
        X[i, :] = X[i, :] / Y[i, 0]
    return True, X


# Seccion 5 - Continuacion resolucion sistemas lineales


# Algoritmo de Gauss - Estrategia de pivot parcial
def gauss_pp(A, B):
    m, n = shape(A)
    p, q = shape(B)

    if m != n or n != p or q < 1:
        return False, "Error gauss_pp: error en las dimensiones."

    if A.dtype == complex or B.dtype == complex:
        gaussA = array(A, dtype=complex)
        gaussB = array(B, dtype=complex)
    else:
        gaussA = array(A, dtype=float)
        gaussB = array(B, dtype=float)

    for k in range(n - 1):
        # Returns the indices of the maximum values
        pos = argmax(abs(gaussA[k:, k]))
        ik = pos + k

        # Si ik = k el pivot se encuentra incialmente en la fila correcta
        if ik != k:
            # Hacemos la permutacion oportuna ya que el pivot escogido no es el que se encuentra inicialmente
            gaussA[[ik, k], :] = gaussA[[k, ik], :]
            gaussB[[ik, k], :] = gaussB[[k, ik], :]

        # Si no se verifica el if, toda la columna es nula (practicamente)
        if abs(gaussA[k, k]) >= 1e-200:
            for i in range(k + 1, n):
                # Operamos elemento diagonal
                gaussA[i, k] = gaussA[i, k] / gaussA[k, k]

                # Operamos todo lo que queda por debajo de la diagonal en la columna oportuna
                gaussA[i, k + 1 :] -= gaussA[i, k] * gaussA[k, k + 1 :]

                # Operamos identicamente en la matriz B (en este caso incluyendo la diagonal)
                gaussB[i, :] -= gaussA[i, k] * gaussB[k, :]

    # Resolvemos el sistema con las matrices diagonales superiores
    exito, X = remonte(gaussA, gaussB)
    return exito, X


# Algoritmo Gauss-jordan - Estrategia de pivot parcial
def gaussjordan_pp(A, B):
    m, n = shape(A)
    p, q = shape(B)
    if m != n or n != p or q < 1:
        return False, "Error gauss_jordan_pp: error en las dimensiones."

    if A.dtype == complex or B.dtype == complex:
        gaussA = array(A, dtype=complex)
        gaussB = array(B, dtype=complex)
    else:
        gaussA = array(A, dtype=float)
        gaussB = array(B, dtype=float)

    for k in range(n):
        pos = argmax(abs(gaussA[k:, k]))
        ik = pos + k
        if ik != k:
            gaussA[[ik, k], :] = gaussA[[k, ik], :]
            gaussB[[ik, k], :] = gaussB[[k, ik], :]

        if abs(gaussA[k, k]) >= 1e-200:
            for i in range(k):
                gaussA[i, k] = gaussA[i, k] / gaussA[k, k]
                gaussA[i, k + 1 :] -= gaussA[i, k] * gaussA[k, k + 1 :]
                gaussB[i, :] -= gaussA[i, k] * gaussB[k, :]

            for i in range(k + 1, n):
                gaussA[i, k] = gaussA[i, k] / gaussA[k, k]
                gaussA[i, k + 1 :] -= gaussA[i, k] * gaussA[k, k + 1 :]
                gaussB[i, :] -= gaussA[i, k] * gaussB[k, :]

    # if min(abs(diag(A))) < 1e-200:
    #    return False, "Error: matriz singular."
    # if B.dtype == complex:
    #    X = array(gaussB, dtype=complex)
    # else:
    #    X = array(gaussB, dtype=float)

    X = copy(gaussB)

    d = diag(gaussA)
    d = reshape(d, (n, 1))
    X = X / d

    return True, X


def gauss_verbose_pp(A, B):
    m, n = shape(A)
    p, q = shape(B)
    if m != n or n != p or q < 1:
        return False, "Error gauss_pp: error en las dimensiones."
    if A.dtype == complex or B.dtype == complex:
        gaussA = array(A, dtype=complex)
        gaussB = array(B, dtype=complex)
    else:
        gaussA = array(A, dtype=float)
        gaussB = array(B, dtype=float)
    for k in range(n - 1):
        pos = argmax(abs(gaussA[k:, k]))
        ik = pos + k
        if ik != k:
            gaussA[[ik, k], :] = gaussA[[k, ik], :]
            gaussB[[ik, k], :] = gaussB[[k, ik], :]
        if abs(gaussA[k, k]) >= 1e-200:
            for i in range(k + 1, n):
                gaussA[i, k] = gaussA[i, k] / gaussA[k, k]
                gaussA[i, k + 1 :] -= gaussA[i, k] * gaussA[k, k + 1 :]
                gaussB[i, :] -= gaussA[i, k] * gaussB[k, :]
    exito, X = remonte(gaussA, gaussB)

    print("Matriz MA: \n", triu(gaussA))
    print("Matriz MB: \n", gaussB)

    return exito, X


def gaussjordan_verbose_pp(A, B):
    m, n = shape(A)
    p, q = shape(B)

    if m != n or n != p or q < 1:
        return False, "Error gauss_jordan_pp: error en las dimensiones."

    if A.dtype == complex or B.dtype == complex:
        gaussA = array(A, dtype=complex)
        gaussB = array(B, dtype=complex)
    else:
        gaussA = array(A, dtype=float)
        gaussB = array(B, dtype=float)

    for k in range(n):
        pos = argmax(abs(gaussA[k:, k]))
        ik = pos + k

        if ik != k:
            gaussA[[ik, k], :] = gaussA[[k, ik], :]
            gaussB[[ik, k], :] = gaussB[[k, ik], :]

        if abs(gaussA[k, k]) >= 1e-200:
            for i in range(k):
                gaussA[i, k] = gaussA[i, k] / gaussA[k, k]
                gaussA[i, k + 1 :] -= gaussA[i, k] * gaussA[k, k + 1 :]
                gaussB[i, :] -= gaussA[i, k] * gaussB[k, :]

            for i in range(k + 1, n):
                gaussA[i, k] = gaussA[i, k] / gaussA[k, k]
                gaussA[i, k + 1 :] -= gaussA[i, k] * gaussA[k, k + 1 :]
                gaussB[i, :] -= gaussA[i, k] * gaussB[k, :]

    #    if min(abs(diag(A))) < 1e-200:
    #         return False, "Error: matriz singular."
    # if B.dtype == complex:
    #     X = array(gaussB, dtype=complex)
    #    else:
    #        X = array(gaussB, dtype=float)

    X = copy(gaussB)

    d = diag(gaussA)
    d = reshape(d, (n, 1))
    X = X / d

    print("Matriz MA: \n", diag(gaussA))
    print("Matriz MB: \n", gaussB)

    return True, X


# Metodo de Gauss - Estrategia primer pivot no nulo
def gauss_1p(A, B):
    m, n = shape(A)
    p, q = shape(B)

    if m != n or n != p or q < 1:
        return False, "Error gauss_pp: error en las dimensiones."

    if A.dtype == complex or B.dtype == complex:
        gaussA = array(A, dtype=complex)
        gaussB = array(B, dtype=complex)
    else:
        gaussA = array(A, dtype=float)
        gaussB = array(B, dtype=float)

    for k in range(n - 1):
        for ik in range(k, n - 1):
            if abs(gaussA[ik, k]) > 1e-200:
                break

        if ik != k:
            gaussA[[ik, k], :] = gaussA[[k, ik], :]
            gaussB[[ik, k], :] = gaussB[[k, ik], :]

        if abs(gaussA[k, k]) >= 1e-200:
            for i in range(k + 1, n):
                gaussA[i, k] = gaussA[i, k] / gaussA[k, k]
                gaussA[i, k + 1 :] -= gaussA[i, k] * gaussA[k, k + 1 :]
                gaussB[i, :] -= gaussA[i, k] * gaussB[k, :]
    exito, X = remonte(gaussA, gaussB)
    return exito, X


# Metodo Gauss-Jordan - Estrategia primer pivot no nulo
def gaussjordan_1p(A, B):
    m, n = shape(A)
    p, q = shape(B)

    if m != n or n != p or q < 1:
        return False, "Error gauss_jordan_pp: error en las dimensiones."

    if A.dtype == complex or B.dtype == complex:
        gaussA = array(A, dtype=complex)
        gaussB = array(B, dtype=complex)
    else:
        gaussA = array(A, dtype=float)
        gaussB = array(B, dtype=float)

    for k in range(n):
        for ik in range(n - 1, k - 1, -1):
            if abs(gaussA[ik, k]) > 1e-200:
                break

        if ik != k:
            gaussA[[ik, k], :] = gaussA[[k, ik], :]
            gaussB[[ik, k], :] = gaussB[[k, ik], :]

        if abs(gaussA[k, k]) >= 1e-200:
            for i in range(k):
                gaussA[i, k] = gaussA[i, k] / gaussA[k, k]
                gaussA[i, k + 1 :] -= gaussA[i, k] * gaussA[k, k + 1 :]
                gaussB[i, :] -= gaussA[i, k] * gaussB[k, :]

            for i in range(k + 1, n):
                gaussA[i, k] = gaussA[i, k] / gaussA[k, k]
                gaussA[i, k + 1 :] -= gaussA[i, k] * gaussA[k, k + 1 :]
                gaussB[i, :] -= gaussA[i, k] * gaussB[k, :]

    # if min(abs(diag(A))) < 1e-200:
    #    return False, "Error: matriz singular."
    # if B.dtype == complex:
    #    X = array(gaussB, dtype=complex)
    # else:
    #    X = array(gaussB, dtype=float)

    X = copy(gaussB)

    d = diag(gaussA)
    d = reshape(d, (n, 1))
    X = X / d

    return True, X


# Seccion 6


# Factorizacion LU (L triang.inf con 1 diag / U triang.super / ambas inversibles)
def facto_lu(A):
    m, n = shape(A)

    if m != n:
        return False, "Error Factorizacion LU: error en las dimensiones."

    if A.dtype == complex:
        lu = array(A, dtype=complex)
    else:
        lu = array(A, dtype=float)

    for k in range(n - 1):
        if abs(lu[k, k]) < 1e-200:
            return (
                False,
                "Error facto_lu: no existe la factorizacion, no posible division.",
            )
        else:
            for i in range(k + 1, n):
                lu[i, k] = lu[i, k] / lu[k, k]
                lu[i, k + 1 :] -= lu[i, k] * lu[k, k + 1 :]

    return True, lu


# Resolucion sistema lineal AX=B mediante factorizacion LU
def metodo_lu(A, B):
    m, n = shape(A)
    p, q = shape(B)

    if m != n or n != p or q < 1:
        return False, "Error metodo LU: Dimensiones no comptabiles."

    exito, lu = facto_lu(A)
    if exito:
        e, Y = descenso1(lu, B)
        t, X = remonte(lu, Y)
        return True, X
    else:
        return False, "Error metodos LU: error en la resolucion."


# Factorizacion - Cholesky (Matriz def.pos) -> C triang.inf con elem.diag positivos
def facto_cholesky(A):
    m, n = shape(A)

    if m != n:
        return False, "Error facto_cholesky: Dimensiones no compatibles."

    if A.dtype == complex:
        return False, "Error facto_cholesky: matriz compleja."
    else:
        # Matriz Cholesky
        c = array(A, dtype=float)

    for i in range(n):
        c[i, i] -= sum(power(c[i, 0:i], 2))

        if c[i, i] >= 1e-100:
            c[i, i] = sqrt(c[i, i])
        else:
            return False, "Error facto_cholesky: no se factoriza la matriz"

        # Se sigue desarrollo de los apuntes
        for j in range(i + 1, n):
            c[j, i] -= sum(c[i, 0:i] * c[j, 0:i])
            c[j, i] = c[j, i] / c[i, i]
            c[i, j] = c[j, i]  # Por simetria

    return True, c


# Resolucion sistema lineal AX=B mediante factorizacion cholesky
def metodo_cholesky(A, B):
    m, n = shape(A)
    p, q = shape(B)

    if m != n or n != p or q < 1:
        return False, "Error metodo Cholesky: Dimensiones no compatibles."

    d, chol = facto_cholesky(A)
    if exito:
        t, Y = descenso(chol, B)
        p, X = remonte(chol, Y)
        return True, X
    else:
        return False, "Error metodo Cholesky: error en la resolucion."


# Seccion 8 - Metodos clasicos


# Jacobi -> A = D - E - F, con diagonal, triang.inf, triang.inf con M = D, N = E - F
# PARAM: Matrices las matrices A y B del sistema lineal AX=B a resolver, XOLD una semilla inicial para empezar el calculo del metodo iterativo
#       asi como itermax, numero maximo de iteracciones, tol el valor de exactitud que deseemos
def jacobi(A, B, XOLD, itermax, tol):
    m, n = shape(A)
    p, q = shape(B)
    r, s = shape(XOLD)

    if m != n or n != p or q != 1 or n != r or s != 1 or min(abs(diag(A))) < 1e-200:
        return False, "ERROR jacobi: no se resuelve el sistema."

    if A.dtype == complex or B.dtype == complex or XOLD.dtype == complex:
        tipo = "complex"
    else:
        tipo = "float"

    k = 0
    error = tol

    while k < itermax and error >= tol:
        k = k + 1
        XNEW = array(B, dtype=tipo)

        for i in range(n):
            if i != 0:
                XNEW[i, 0] -= A[i, :i] @ XOLD[:i, 0]

            if i != n - 1:
                XNEW[i, 0] -= A[i, i + 1 :] @ XOLD[i + 1 :, 0]

            XNEW[i, 0] = XNEW[i, 0] / A[i, i]

        error = norma_vec(XNEW - XOLD, inf)
        XOLD = array(XNEW)

    print("Iteracion: k = ", k)
    print("Error absoluto: error = ", error)

    if k == itermax and error >= tol:
        return False, "ERROR jacobi: no se alcanza convergencia."
    else:
        print("Convergencia numerica alcanzada por metodo jacobi.")
        return True, XNEW


# Metodo Gauss-Seidel (relajacion w=1) A = D - E- F con M = D - E, N = F
def gauss_seidel(A, B, XOLD, itermax, tol):
    m, n = shape(A)
    p, q = shape(B)
    r, s = shape(XOLD)

    if m != n or n != p or q != 1 or n != r or s != 1 or min(abs(diag(A))) <= 1e-200:
        return False, "ERROR gauss_seidel: no se resuelve el sistema."

    k = 0
    error = 1.0

    while k < itermax and error >= tol:
        k = k + 1
        XNEW = array(B)

        for i in range(n):
            if i != 0:
                XNEW[i, 0] -= A[i, :i] @ XNEW[:i, 0]
            if i != n - 1:
                XNEW[i, 0] -= A[i, i + 1 :] @ XOLD[i + 1 :, 0]
            XNEW[i, 0] = XNEW[i, 0] / A[i, i]

        error = norma_vec(XNEW - XOLD, inf)
        XOLD = array(XNEW)

    print("Iteracion: k = ", k)
    print("Error absoluto: error = ", error)

    if k == itermax and error >= tol:
        return False, "ERROR gauss_seidel: no se alcanza convergencia."
    else:
        print("Convergencia numerica alcanzada: gauss_seidel.")
        return True, XNEW


# Metodo relajacion -> A = 1/w D - (1-w)/w D - E - F con M= 1/w D - E, N = (1-w)/w D + F
def relajacion(A, B, XOLD, omega, itermax, tol):
    m, n = shape(A)
    p, q = shape(B)
    r, s = shape(XOLD)

    if m != n or n != p or q != 1 or n != r or s != 1 or min(abs(diag(A))) < 1e-200:
        return False, "ERROR relajacion: no se resuelve el sistema."

    k = 0
    error = 1.0

    while k < itermax and error >= tol:
        k = k + 1
        XNEW = array(B)

        for i in range(n):
            if i != 0:
                XNEW[i, 0] -= A[i, :i] @ XNEW[:i, 0]
            if i != n - 1:
                XNEW[i, 0] -= A[i, i + 1 :] @ XOLD[i + 1 :, 0]

            XNEW[i, 0] += ((1 - omega) / omega) * A[i, i] * XOLD[i, 0]
            XNEW[i, 0] = omega * XNEW[i, 0] / A[i, i]

        error = norma_vec(XNEW - XOLD, inf)
        XOLD = array(XNEW)

    print("Iteracion: k = ", k)
    print("Error absoluto: error = ", error)

    if k == itermax and error >= tol:
        return False, "ERROR relajacion: no se alcanza convergencia."
    else:
        print("Convergencia numerica alcanzada: relajacion.")
        return True, XNEW


# Seccion 9 - Calculo autovalores


# X debe ser unitario
def potencia(A, X, norma, itermax, tol):
    m, n = shape(A)
    r, s = shape(X)

    if m != n or n != r or s != 1:
        return False, "ERROR potencia: no se ejecuta el programa.", 0, 0

    k = 0
    error = tol
    normaold = 0.0

    if A.dtype == complex or X.dtype == complex:
        lambdas = zeros(n, dtype=complex)
    else:
        lambdas = zeros(n, dtype=float)

    while k < itermax and error >= tol:
        k = k + 1
        Y = A @ X
        normanew = norma_vec(Y, norma)
        error = abs(normanew - normaold)

        for i in range(n):
            if abs(X[i, 0]) >= 1e-100:
                lambdas[i] = Y[i, 0] / X[i, 0]
            else:
                lambdas[i] = 0.0

        X = Y / normanew

        print("Iteracion: k = ", k)
        print("Norma: ||A*X_k|| = ", normanew)
        print("Lambdas: lambdas_k = \n", lambdas)
        print("Vectores: X_k = \n", transpose(X))
        normaold = normanew

    if k == itermax and error >= tol:
        return False, "ERROR potencia: no se alcanza convergencia.", 0, 0
    else:
        print("Metodo de la potencia: convergencia numerica alcanzada.")
        return True, normanew, lambdas, X


def potenciainv(A, X, norma, itermax, tol):
    m, n = shape(A)
    r, s = shape(X)

    if m != n or n != r or s != 1:
        return False, "ERROR potenciainv: no se ejecuta el programa.", 0, 0

    e, LU = facto_lu(A)
    if not e:
        return False, "ERROR potenciainv: sin factorizacion LU.", 0, 0

    k = 0
    error = 1.0
    normaold = tol

    if A.dtype == complex or X.dtype == complex:
        lambdas = zeros(n, dtype=complex)
    else:
        lambdas = zeros(n, dtype=float)

    while k < itermax and error >= tol:
        k = k + 1
        exito, Y = descenso1(LU, X)
        exito, Y = remonte(LU, Y)

        if not exito:
            return False, "ERROR potenciainv: sin factorizacion LU.", 0, 0

        normanew = norma_vec(Y, norma)
        error = abs(normanew - normaold)

        for i in range(n):
            if abs(X[i, 0]) >= 1e-100:
                lambdas[i] = Y[i, 0] / X[i, 0]
            else:
                lambdas[i] = 0.0
        X = Y / normanew

        print("Iteracion: k = ", k)
        print("Norma: ||A-1*X_k|| = ", normanew)
        print("Lambdas: lambdas_k = ", lambdas)
        print("Vectores: X_k = ", transpose(X))
        normaold = normanew

    if k == itermax and error >= tol:
        return False, "ERROR potenciainv: no se alcanza convergencia.", 0, 0
    else:
        print("Metodo de la potencia inversa: convergencia numerica alcanzada.")
        return True, normanew, lambdas, X


# Potencia desplazada
def potenciades(A, X, des, norma, itermax, tol):
    m, n = shape(A)
    r, s = shape(X)
    if m != n or n != r or s != 1:
        return False, "ERROR potenciades: no se ejecuta el programa.", 0, 0
    B = A - des * eye(n)
    exito, normanew, lambdas, X = potencia(B, X, norma, itermax, tol)
    return exito, normanew, lambdas, X


# Potencia desplazada respecto matrix inversa
def potenciadesinv(A, X, des, norma, itermax, tol):
    m, n = shape(A)
    r, s = shape(X)
    if m != n or n != r or s != 1:
        return False, "ERROR potenciadesinv: no se ejecuta el programa.", 0, 0
    B = A - des * eye(n)
    exito, normanew, lambdas, X = potenciainv(B, X, norma, itermax, tol)
    return exito, normanew, lambdas, X
