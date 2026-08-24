"""
Eliptica y parabolica 1D - diferencias finitas 2o orden
Problema estacionario u - alfa u'' = f con Dirichlet (directa/penalizacion/simetria)
y Neumann (nodo fantasma). Parabolica u_t - alfa u_xx = f por metodo de lineas
(explicito CFL, implicito incondicional, theta-metodo).
Detalles: docs/enunciados_resumidos.md#01_edp_1d
Dependencias: numpy, scipy.sparse, matplotlib
"""

# PARA MEDIR ORDEN EN EL ESPACIO NECESITO DIVIDIR DT POR DOS EN CADA OCASION (SUPONIENDO NT -> 2 N = * NT_new)
# Si A no simetrica no podemos usar cholesky (Doble de eficiente)
import time

from numpy import *  # type: ignore
from scipy.sparse.linalg import splu
from scipy.sparse import lil_matrix, identity
from scipy.linalg import lu_factor, lu_solve, cho_factor, cho_solve
from matplotlib.pyplot import *  # type: ignore


def f0_1(x):
    y = 2.0 * sin(x)
    return y


def uexacta_1(x):
    y = sin(x)
    return y


# Resuelve el problema de contorno
# u-alfa*u''=f
# u(x0)=u0
# u(xf)=uL
# usamos matrices "vacias"


# Imposicion directa de las condicciones dirichlet de contorno
def contorno_directo_sparse_v0(x0, xf, N, alfa, ua, ub, fuente, exacta, dibujar=False):
    t1 = time.time()

    N = int(N)
    x0 = float(x0)
    xf = float(xf)
    alfa = float(alfa)
    ua = float(ua)
    ub = float(ub)

    dx = (xf - x0) / float(N)
    dx2 = dx * dx

    x = linspace(x0, xf, N + 1)
    xp = x[1:N]
    D = lil_matrix((N - 1, N - 1), dtype="float64")
    Id = identity(N - 1, dtype="float64", format="csc")

    D.setdiag(2.0 * ones(N - 1), 0)
    D.setdiag(-1.0 * ones(N - 2), 1)
    D.setdiag(-1.0 * ones(N - 2), -1)

    D = D.tocsc()
    A = Id + alfa / dx2 * D
    LU = splu(A)

    # Construccion del vector b
    b = fuente(xp)

    # En la primera y ultima ecuacion del metodo, despejamos u0 y uN (valores ya conocidos)
    b[0] += ua * alfa / dx2
    b[N - 2] += ub * alfa / dx2

    usol = LU.solve(b)

    tf = time.time()
    print("Tiempo de ejecucion:", format(tf - t1))

    uplot = x * 0.0
    uplot[1:N] = usol

    # Imponemos directamente las condicciones de contorno
    uplot[0] = ua
    uplot[N] = ub

    if dibujar:
        plot(x, uplot, "b", x, exacta(x), "r")
        show()

    err = max(abs(uplot - exacta(x)))
    print("Error cometido:", format(err))

    return err


# Ahora incluimos todos los puntos como incognitas (imponemos directamente contorno dirichlet)
def contorno_directo_sparse_v1(x0, xf, N, alfa, ua, ub, fuente, exacta, dibujar=False):
    t1 = time.time()

    N = int(N)
    x0 = float(x0)
    xf = float(xf)
    alfa = float(alfa)
    ua = float(ua)
    ub = float(ub)

    dx = (xf - x0) / float(N)
    dx2 = dx * dx

    x = linspace(x0, xf, N + 1)
    D = lil_matrix((N + 1, N + 1), dtype="float64")
    Id = identity(N + 1, dtype="float64", format="csc")

    # Construimos la matriz D, sin tener en cuenta la primera y la ultima ecuacion
    D.setdiag(2.0 * ones(N + 1), 0)
    D.setdiag(-1.0 * ones(N), 1)
    D.setdiag(-1.0 * ones(N), -1)

    # Modificamos D para imponer condicciones de contorno
    D[0, 0] = 0.0
    D[0, 1] = 0.0

    D[N, N] = 1.0
    D[N, N - 1] = 0.0

    D = D.tocsc()
    # la matriz A no es simetrica
    A = Id + alfa / dx2 * D

    LU = splu(A)

    b = fuente(x)

    # Condicciones de contorno
    b[0] = ua
    b[N] = ub

    usol = LU.solve(b)

    tf = time.time()
    print("Tiempo de ejecucion:", format(tf - t1))
    if dibujar:
        plot(x, usol, "b", x, exacta(x), "r")
        show()

    err = max(abs(usol - exacta(x)))
    print("Error cometido:", format(err))

    return err


# Ahora incluimos todos los puntos como incognitas, pero operamos para
#   que la matriz resultante siga siendo simetrica
def contorno_directo_sparse_v2(x0, xf, N, alfa, ua, ub, fuente, exacta, dibujar=False):
    # Penalizacion
    M = 1e20

    t1 = time.time()

    N = int(N)
    x0 = float(x0)
    xf = float(xf)
    alfa = float(alfa)
    ua = float(ua)
    ub = float(ub)

    dx = (xf - x0) / float(N)
    dx2 = dx * dx

    x = linspace(x0, xf, N + 1)
    D = lil_matrix((N + 1, N + 1), dtype="float64")
    Id = identity(N + 1, dtype="float64", format="csc")

    # construimos la matriz D
    D.setdiag(2.0 * ones(N + 1), 0)
    D.setdiag(-1.0 * ones(N), 1)
    D.setdiag(-1.0 * ones(N), -1)

    D = D.tocsc()
    A = Id + alfa / dx2 * D

    # Preparamos A para penalizacion
    A[0, 0] = M
    A[N, N] = M

    # la matriz A es simetrica
    LU = splu(A)

    b = fuente(x)

    # Contorno dirichlet por penalizacion
    b[0] = M * ua
    b[N] = M * ub

    usol = LU.solve(b)

    tf = time.time()
    print("Tiempo de ejecucion:", format(tf - t1))

    if dibujar:
        plot(x, usol, "b", x, exacta(x), "r")
        show()

    err = max(abs(usol - exacta(x)))
    print("Error cometido:", format(err))

    return err


# resuelve el problema de contorno
# -nu*u'' +alfa*u=f
# u(x0)=u0
# u(xf)=uL
# usando matrices densas


# MUCHO MAS LENTO al no aprovechar la estrucutura sparse
def contorno_directo_densa(x0, xf, N, alfa, ua, ub, fuente, exacta, dibujar=False):
    t1 = time.time()
    M = 1e30

    N = int(N)
    x0 = float(x0)
    xf = float(xf)
    dx = (xf - x0) / float(N)
    dx2 = dx * dx
    alfa = float(alfa)
    ua = float(ua)

    ub = float(ub)
    x = linspace(x0, xf, N + 1)
    D = lil_matrix((N + 1, N + 1), dtype="float64")
    Id = identity(N + 1, dtype="float64", format="csc")

    D.setdiag(2.0 * ones(N + 1), 0)
    D.setdiag(-1.0 * ones(N), 1)
    D.setdiag(-1.0 * ones(N), -1)

    D = D.tocsc()
    A = Id + alfa / dx2 * D

    # modificamos A para imponer la primera y la ultima ecuacion
    A[0, 0] = M
    A[N, N] = M

    # la matriz A es simetrica
    # usamos la matriz completa, incluyendo los ceros
    A = A.todense()
    CA = cho_factor(A)

    b = fuente(x)

    # Contorno
    b[0] = M * ua
    b[N] = M * ub

    usol = cho_solve(CA, b)

    tf = time.time()
    print("Tiempo de ejecucion:", format(tf - t1))

    if dibujar:
        plot(x, usol, "b", x, exacta(x), "r")
        show()

    err = max(abs(usol - exacta(x)))
    print("Error cometido:", format(err))


# ---------------- CASO 1 ----------------#
#   NODO FANTASMA ~ Condiccion contorno de tipo neumann
#       -> Anadimos una ecuacion extra para resolver el problema
#       En este caso ub sera la aproximacion de la derivada en u'(b)
def caso1(x0, xf, N, alfa, ua, g, fuente, exacta, dibujar=False):
    t1 = time.time()

    N = int(N)
    x0 = float(x0)
    xf = float(xf)

    dx = (xf - x0) / float(N)
    dx2 = dx * dx

    alfa = float(alfa)
    ua = float(ua)
    g = float(g)

    x = linspace(x0, xf, N + 1)
    D = lil_matrix((N, N), dtype="float64")
    Id = identity(N, dtype="float64", format="csc")

    D.setdiag(2.0 * ones(N), 0)
    D.setdiag(-1.0 * ones(N - 1), 1)
    D.setdiag(-1.0 * ones(N - 1), -1)

    # Cambiamos del formato lil al formato csc
    D = D.tocsc()
    A = Id + alfa / dx2 * D

    # Condiccion tipo neumann en b
    A[N - 1, N - 1] = 1 + 2 * alfa / dx2
    A[N - 1, N - 2] = -2 * alfa / dx2

    LU = splu(A)

    # Construccion del vector b
    b = fuente(x[1:])
    b[0] += ua * alfa / dx2
    b[N - 1] += 2 * g * alfa / dx

    usol = LU.solve(b)

    usol = concatenate(([ua], usol))

    tf = time.time()
    print("Tiempo de ejecucion:", format(tf - t1))

    # print(f"Valor en a es {usol[0]}")
    # print(f"Valor en b es {usol[-1:]}")
    # print(f"ERROR IN value {argmax(abs(usol - uexacta(x)))}")

    if dibujar:
        plot(x, usol, "b", x, exacta(x), "r")
        suptitle(f"Aproximacion con tecnica nodo fantasma N={N}")
        legend(["Aproximacion", "Exacta"])
        show()

    err = max(abs(usol - exacta(x)))
    print("Error cometido:", format(err))

    return err


# Incluimos primer prunto tambien como incognitas
def caso1_v1(x0, xf, N, alfa, ua, g, fuente, exacta, dibujar=False):
    t1 = time.time()

    N = int(N)
    x0 = float(x0)
    xf = float(xf)

    dx = (xf - x0) / float(N)
    dx2 = dx * dx

    alfa = float(alfa)
    ua = float(ua)
    g = float(g)

    x = linspace(x0, xf, N + 1)
    D = lil_matrix((N + 1, N + 1), dtype="float64")
    Id = identity(N + 1, dtype="float64", format="csc")

    # Construimos la matriz D, sin tener en cuenta la primera y la ultima ecuacion
    D.setdiag(2.0 * ones(N + 1), 0)
    D.setdiag(-1.0 * ones(N), 1)
    D.setdiag(-1.0 * ones(N), -1)

    # Modificamos D para imponer la primera y la segunda ecuacion contorno
    D[0, 0] = 0.0
    D[0, 1] = 0.0

    D = D.tocsc()
    A = Id + alfa / dx2 * D

    # Condiccion tipo neumann en b
    A[N, N] = 1 + 2 * alfa / dx2
    A[N, N - 1] = -2 * alfa / dx2

    LU = splu(A)

    b = fuente(x)

    # Condiccion de contorno
    b[0] = ua
    b[N] += 2 * g * alfa / dx

    usol = LU.solve(b)

    tf = time.time()
    print("Tiempo de ejecucion:", format(tf - t1))

    if dibujar:
        plot(x, usol, "b", x, exacta(x), "r")
        title(f"Aproximacion con tecnica nodo fantasma N={N}")
        legend(["aproximacion", "exacta"])
        show()

    err = max(abs(usol - exacta(x)))
    print("Error cometido:", format(err))

    return err


# Incluimos todos los puntos como incognitas, pero operamos para seguir teniendo una matriz simetrica
def caso1_v2(x0, xf, N, alfa, ua, ub, fuente, exacta, dibujar=False):
    # Penalizacion para imponer frontera (mantener simetria)
    M = 1e20

    t1 = time.time()

    N = int(N)
    x0 = float(x0)
    xf = float(xf)
    dx = (xf - x0) / float(N)
    dx2 = dx * dx

    alfa = float(alfa)
    ua = float(ua)
    ub = float(ub)

    x = linspace(x0, xf, N + 1)
    D = lil_matrix((N + 1, N + 1), dtype="float64")
    Id = identity(N + 1, dtype="float64", format="csc")

    D.setdiag(2.0 * ones(N + 1), 0)
    D.setdiag(-1.0 * ones(N), 1)
    D.setdiag(-1.0 * ones(N), -1)

    D = D.tocsc()
    A = Id + alfa / dx2 * D

    b = fuente(x)

    # Aplicar penalizacion Dirichlet en a (izquierda)
    # Imponemos penalizacion para hacer la matriz simetrica
    A[0, 0] = M

    # Condiccion tipo neumann en b
    # Imponemos penalizacion para hacer la matriz simetrica
    A[N, N] = (1 / 2) * (1 + 2 * alfa / dx2)
    A[N, N - 1] = -alfa / dx2

    b[0] = M * ua
    b[N] *= 1 / 2
    b[N] += alfa / dx * ub

    # Como A es simetrica utilizamos una matriz cholesky
    # La matriz de cholesky se puede computar en la mitad de tiempo (aprox)
    # Tiene que pasarse a densa para poder computar cholesky
    A = A.todense()
    C = cho_factor(A)
    usol = cho_solve(C, b)

    tf = time.time()
    print("Tiempo de ejecucion:", format(tf - t1))

    if dibujar:
        plot(x, usol, "b", x, exacta(x), "r")
        title(f"Aproximacion con tecnica nodo fantasma con penalizacion N={N}")
        legend(["aproximacion", "exacta"])
        show()

    err = max(abs(usol - exacta(x)))

    print("Error cometido:", format(err))
    show()

    return err


# ---------------- CASO 2 ----------------#
# Tomamos variables como boolean para saber si se nos imponen condicciones del tipo dirichelt o del tipo neumann y
#   una vez que lo aclaramos no hacemos mas que seguir las implementaciones anteriores
def caso2(
    x0,
    xf,
    N,
    alfa,
    ua,
    ub,
    fuente,
    dirirchet_a=True,
    dirichet_b=True,
    exacta=None,
    dibujar=False,
):
    t1 = time.time()

    N = int(N)
    x0 = float(x0)
    xf = float(xf)

    dx = (xf - x0) / float(N)
    dx2 = dx * dx

    alfa = float(alfa)
    ua = float(ua)
    ub = float(ub)

    # Incluimos todos los puntos como incognitas (necesario si tenemos que usar nodos fantasmas)
    x = linspace(x0, xf, N + 1)

    # Como hemos visto en teoria la construccion de la matriz A es identica en todos los casos, unicamente cambiarian
    #   la primera y ultima fila en funcion del tipo de contorno.
    D = lil_matrix((N + 1, N + 1), dtype="float64")
    Id = identity(N + 1, dtype="float64", format="csc")

    # construimos la matriz D
    D.setdiag(2.0 * ones(N + 1), 0)
    D.setdiag(-1.0 * ones(N), 1)
    D.setdiag(-1.0 * ones(N), -1)

    D = D.tocsc()

    # Modificamos D para imponer las condicciones de contornos requeridas
    if not dirichet_b:
        D[N, N] = 2
        D[N, N - 1] = -2
    else:
        D[N, N] = 0.0
        D[N, N - 1] = 0.0

    if not dirirchet_a:
        D[0, 0] = 2
        D[0, 1] = -2
    else:
        D[0, 0] = 0.0
        D[0, 1] = 0.0

    A = Id + alfa / dx2 * D
    LU = splu(A)

    b = fuente(x)

    if not dirichet_b:
        b[N] += 2 * alfa / dx * ub
    else:
        b[N] = ub

    if not dirirchet_a:
        b[0] += -2 * alfa / dx * ua
    else:
        b[0] = ua

    usol = LU.solve(b)

    tf = time.time()
    print("Tiempo de ejecucion:", format(tf - t1))

    if dibujar:
        figure()
        plot(x, usol, "b", x, exacta(x), "r")
        title(f"Nodo fantasma con penalizacion N={N}, contorno arbitrario")
        legend(["Aproximacion", "Exacta"])
        show()

    err = max(abs(usol - exacta(x)))
    print("Error cometido:", format(err))
    show()

    return err


# ---------------- CASO 3 ~ Metodos de lineas ----------------#


# El error en un tiempo T es la suma acumulada de los errores locales cometidos en cada paso anterior, mas el error propio de ese ultimo paso


# Incondiccionalmente estable
# Segundo orden en espacio, primero en tiempo
def caso3_implicito(
    x0, xf, Nx, t0, tf, Nt, alfa, ua, ub, u0, fuente, uexacta, dibujar=False
):
    ti = time.time()

    Nx = int(Nx)
    Nt = int(Nt)
    x0 = float(x0)
    t0 = float(t0)
    xf = float(xf)
    tf = float(tf)
    alfa = float(alfa)
    ua = float(ua)
    ub = float(ub)

    # Traslaccion temporal dedsde el instante inicial al origen
    if t0 != 0:
        tf -= t0
        t0 = 0

    # Marca donde nos encontramos del mallado discreto del tiempo
    t = float(0)

    dx = (xf - x0) / float(Nx)
    dx2 = dx * dx

    #  SE MIRA EL MAS GRANDE <--------------
    # Queremos ver error en el espacio, por lo tanto dx2 < dt
    # Vemos error en tiempo si dt > dx2, sino en el espacio porque esencialmente se lo come
    dt = (tf - t0) / float(Nt)

    print(f"dx2={round(dx2, 7)}, dt={round(dt, 7)}")

    x = linspace(x0, xf, Nx + 1)
    u = u0(x)

    D = lil_matrix((Nx + 1, Nx + 1), dtype="float64")
    Id = identity(Nx + 1, dtype="float64", format="csc")

    D.setdiag(2.0 * ones(Nx + 1), 0)
    D.setdiag(-1.0 * ones(Nx), 1)
    D.setdiag(-1.0 * ones(Nx), -1)

    D[0, 0] = 0
    D[0, 1] = 0
    D[Nx, Nx - 1] = 0
    D[Nx, Nx - 1] = 0

    D = D.tocsc()
    A = Id + alfa * dt / dx2 * D

    LU = splu(A)

    # Contador para dibujar la grafica
    i = 0
    t_dibujo = 0.0

    #  Inicializamos mathplotlib
    if dibujar:
        figure()
        # crear lineas vacias una sola vez
        # (ln_approx,) = plot(x, u, "b", label="Aproximacion")
        # (ln_exact,) = plot(x, uexacta(x, t), "r", label="Exacta")
        # legend(["Aproximacion", "Exacta"])
        # draw()

    # Inicializacion calculo del error
    error = array(0.0, dtype="float64")

    while t < tf - dt / 2:
        # print(f"Iteraccion {round(t,2)}")
        t += dt

        b = fuente(x, t) * dt + u

        #! ESTABA MAL HECHO EN LA SECCION ENTREGADA
        # Imposicion de condicciones de contorno
        b[0] = ua
        # b[1] += ua * alfa * dt / dx2

        # b[N - 1] += ub * alfa * dt / dx2
        b[Nx] = ub

        usol = LU.solve(b)

        # El usol sera la nueva solucion de u en el instante de tiempo t + dt
        u = usol

        # Errores
        err = max(abs(u - uexacta(x, t)))
        # print("Error espacial cometido:",format(err))
        error = append(error, err)

        if i % 100 == 0 and dibujar:
            tiempo_dibujo_inicial = time.time()

            clf()
            plot(x, u, "b", label="Aproximacion")
            plot(x, uexacta(x, t), "r", label="Exacta")
            # ln_approx.set_ydata(u)
            # ln_exact.set_ydata(uexacta(x, t))

            title(f"Aproximacion con N={Nx} en instante t={round(t, 3)}")
            draw()
            pause(0.04)
            # show()

            t_dibujo += time.time() - tiempo_dibujo_inicial

        i += 1

    tf = time.time()
    print(f"Tiempo ejecucion: {format(tf - ti - t_dibujo)}")
    # print(f"Error: {max(error)}")

    if dibujar:
        show()

    return max(error)


# Tomamos la condiccion de neumann como una funcion de t
# NECESITO TOMAR NT PARA PODER FIJAR NX PARA VER ERROR TEMPORAL
def caso3_implicito_neumann_b(
    x0, xf, Nx, t0, tf, Nt, alfa, ua, g, u0, f, uexacta, dibujar=False
):
    time.time()

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
    t = float(0)

    dx = (xf - x0) / float(Nx)
    dx2 = dx * dx

    # Medimos el error del valor que es mas grande
    # Queremos ver error en el espacio, por lo tanto dx2 < dt
    # Vemos error en tiempo si dt > dx2, sino en el espacio porque esencialmente se lo come
    dt = (tf - t0) / float(Nt)

    print(f"dx2={round(dx2, 8)}, dt={round(dt, 8)}")

    alfa = float(alfa)
    ua = float(ua)
    # g = float(g)

    x = linspace(x0, xf, Nx + 1)
    u = u0(x)

    D = lil_matrix((Nx + 1, Nx + 1), dtype="float64")
    Id = identity(Nx + 1, dtype="float64", format="csc")

    D.setdiag(2.0 * ones(Nx + 1), 0)
    D.setdiag(-1.0 * ones(Nx), 1)
    D.setdiag(-1.0 * ones(Nx), -1)

    # Imposicion condicciones de contorno en x=a
    D[0, 0] = 0
    D[0, 1] = 0

    D = D.tocsc()
    A = Id + alfa * dt / dx2 * D

    # Condiccion contorno neumann
    A[Nx, Nx - 1] = -2 * alfa * dt / dx2

    A.tocsc()
    LU = splu(A)

    # Contador para pintar la funcion
    i = 0

    #  Inicializamos mathplotlib
    if dibujar:
        figure()

    # Inicializacion calculo error
    error = array(0.0, dtype="float64")

    while t < tf - dt / 2:
        if t > tf - dt / 2:
            t = tf - dt

        t += dt
        b = f(x, t) * dt + u

        # Dirichlet en x=a
        b[0] = ua

        # Neumann en x=b
        #! ESTABA EVALUANDO F EN EL PUNTO N (NUMERO DE PARTICIONES) NO EN LA COORDENADA X[N]
        b[Nx] += alfa * dt / dx * g(t) * 2

        usol = LU.solve(b)

        # El usol sera la nueva solucion de u en el instante de tiempo t + dt
        u = usol

        # Errores
        err = max(abs(u - uexacta(x, t)))
        # print("Error cometido:", format(err))
        error = append(error, err)

        if i % 100 == 0 and dibujar:
            clf()
            plot(x, u, "b", label="Aproximacion")
            plot(x, uexacta(x, t), "r", label="Exacta")

            title(f"Aproximacion con N={Nx} en instante t={round(t, 3)}")
            legend(["Aproximacion", "Exacta"])
            draw()
            pause(0.04)

        i += 1

    tf = time.time()

    if dibujar:
        show()

    return max(error)


#####################################


# Condiccionalmente estable -> Condicciona el valor de dt
# Permite obtener el valor tras el siguiente paso de tiempo sin necesidad de resolver un sistema
# Segundo orden en espacio, primero en tiempo
# Pero se lo come, al final seran el mismo
def caso3_explicito(x0, xf, N, t0, tf, alfa, ua, ub, u0, f, uexacta, dibujar=False):
    time.time()

    N = int(N)
    x0 = float(x0)
    t0 = float(t0)
    xf = float(xf)
    tf = float(tf)
    alfa = float(alfa)
    ua = float(ua)
    ub = float(ub)

    dx = (xf - x0) / float(N)
    dx2 = dx * dx

    # Queremos ver error en el espacio, por lo tanto dx2 > dt
    t = float(t0)

    # ------- Condiccion de estabilidad -------------- #

    # Eleccion por CFL, tomamos CFL = 1/2
    # CFL = 1/4
    # dt = dx2 * 1/ (alfa * 2) * CFL

    # Otra opcion
    M = ceil(tf * 2 * alfa / dx2)
    dt = tf / M

    print(f"dx2={round(dx2, 7)}, dt={round(dt, 7)}")

    x = linspace(x0, xf, N + 1)
    u = u0(x)

    D = lil_matrix((N + 1, N + 1), dtype="float64")
    Id = identity(N + 1, dtype="float64", format="csc")

    D.setdiag(2.0 * ones(N + 1), 0)
    D.setdiag(-1.0 * ones(N), 1)
    D.setdiag(-1.0 * ones(N), -1)

    # Cambiamos del formato lil al formato csc
    D = D.tocsc()
    A = Id - alfa * dt / dx2 * D
    # Modificamos D para imponer la primera y la segunda ecuacion de contorno

    # Contador para ir pintando la funcion cada cierto numero dado de iteracciones
    i = 0

    error = array(0.0, dtype="float64")

    if dibujar:
        figure()

    while t < tf - dt:
        # print(f"Iteraccion {round(t,2)}")
        if t > tf - dt:
            tnew = tf
        else:
            tnew = t + dt

        # Resolvemos el sistema lineal (explicito)
        parte_principal = (A * u)[1:-1] + dt * f(x[1:-1], t)

        # Imponemos el contorno
        usol = concatenate((array([ua]), parte_principal, array([ub])))

        # Si tuvieramos condiccion de tipo neumann en x=a. (r = alfa * dt / dx2)
        # u_0^(n+1) = (1 - 2r) * u_0^n + 2r * u_1^n + dt * f_0^n - (2alfadt / dx) * h^n
        # usol[0] = ua
        # usol[N] = ub

        # El usol sera la nueva solucion de u en el instante de tiempo t + dt
        u = usol

        # Errores
        err = max(abs(u - uexacta(x, tnew)))
        # print("Error espacial cometido:",format(err))
        error = append(error, err)

        if i % 200 == 0 and dibujar:
            clf()
            plot(x, u, "b", x, uexacta(x, tnew), "r")
            xlabel("x values")

            # tvalues = linspace(t0,tf, 100)
            # plot(tvalues,usol,'b', tvalues,uexacta(1, tvalues),'r')
            # xlabel("x values")

            title(f"Aproximacion con N={N}")
            legend([f"Aproximacion en t={round(t, 2)}", f"Exacta en t={round(t, 2)}"])
            draw()
            pause(0.5)
            # show()

        i += 1
        t = tnew

    tf = time.time()
    if dibujar:
        show()

    return max(error)


# Segundo orden en espacio, segundo en tiempo
# Theta = 0 -> Explicito
# Theta = 1 -> Implicito
# Para imponer condiccion contorno de tipo neumann podemos utilizar esquema implicito e implicito con la tecnica del nodo fantasma
#   sumar ambos de forma ponderada y alterar nuestra primera o ultima ecuacion en este metodo para imponer neumann.
def caso3_theta_metodo(
    x0, xf, Nx, t0, tf, Nt, alfa, ua, ub, u0, f, theta, uexacta, dibujar
):
    Nx = int(Nx)
    Nt = int(Nt)
    x0 = float(x0)
    t0 = float(t0)
    xf = float(xf)
    tf = float(tf)
    alfa = float(alfa)
    ua = float(ua)
    ub = float(ub)

    dx = (xf - x0) / float(Nx)
    dx2 = dx * dx

    t = float(t0)

    # Condicion de estabilidad para los metodos condicionalmente estables (theta < 0.5)
    # Para theta >= 0.5, el metodo es incondicionalmente estable y podemos elegir dt.
    if theta < 0.5:
        # La condicion CFL es dt <= dx^2 / (2 * alfa * (1 - 2*theta))
        # Se toma un dt un 90% del limite para seguridad.
        dt_max = dx2 / (2 * alfa * (1 - 2 * theta))
        dt = 0.9 * dt_max
    else:
        dt = (tf - t0) / float(Nt)

    x = linspace(x0, xf, Nx + 1)
    u = u0(x)

    D = lil_matrix((Nx + 1, Nx + 1), dtype="float64")
    Id = identity(Nx + 1, dtype="float64", format="csc")

    # Matriz de diferencias finitas para la segunda derivada (sin condiciones de contorno)
    D.setdiag(2.0 * ones(Nx + 1), 0)
    D.setdiag(-1.0 * ones(Nx), 1)
    D.setdiag(-1.0 * ones(Nx), -1)

    # Imponer condiciones de contorno tipo Dirichlet en D
    D[0, 0] = 0.0
    D[0, 1] = 0.0

    D[Nx, Nx] = 0.0
    D[Nx, Nx - 1] = 0.0

    D = D.tocsc()

    # Matriz del sistema implicito completo
    AI = Id + alfa * dt / dx2 * D

    # Matriz del sistema explicito completo
    AE = Id - alfa * dt / dx2 * D

    # Matriz del lado izquierdo (LHS) de la ecuacion:
    LHS_matrix = (1 - theta) * Id + theta * AI
    LU = splu(LHS_matrix)

    i = 0
    if dibujar:
        figure()

    error_list = [0.0]

    while t < tf - dt / 2:
        tnew = t + dt

        # Construccion del vector del lado derecho (RHS)
        b = ((1 - theta) * AE + theta * Id) * u + dt * (
            (1 - theta) * f(x, t) + theta * f(x, tnew)
        )

        # Imponer condiciones de contorno
        b[0] = ua
        b[Nx] = ub

        usol = LU.solve(b)

        u = usol

        # --- Calculo de error y visualizacion ---
        err = max(abs(u - uexacta(x, tnew)))
        error_list.append(err)

        if i % 100 == 0 and dibujar:
            clf()
            plot(x, u, "b.-", label=f"Aproximacion en t={round(tnew, 2)}")
            plot(x, uexacta(x, tnew), "r-", label=f"Exacta en t={round(tnew, 2)}")
            xlabel("Posicion x")
            title(f"Aproximacion con N={Nx}, dt={dt:.4f}, theta={theta}")
            legend()

            draw()
            pause(0.01)

        i += 1
        t = tnew

    if dibujar:
        show()

    return max(error_list)


def caso3_lineas_condiciones(
    x0,
    xf,
    N,
    t0,
    tf,
    alfa,
    ua,
    ub,
    u0,
    f,
    fexacta=None,
    metodo="implicito",
    dirichlet_a=True,
    dirichlet_b=True,
    dt=None,
    dibujar=True,
):
    t1 = time.time()

    if metodo.lower() not in ("implicito", "explicito"):
        raise ValueError(
            "Seleccione 'implicito' o 'explicito' para el parametro metodo."
        )

    N = int(N)
    if N < 2:
        raise ValueError("Se requiere N >= 2 para discretizar el intervalo espacial.")

    x0 = float(x0)
    xf = float(xf)
    t0 = float(t0)
    tf = float(tf)
    alfa = float(alfa)
    ua = float(ua)
    ub = float(ub)

    dx = (xf - x0) / float(N)
    dx2 = dx * dx

    x = linspace(x0, xf, N + 1)
    u = array(u0(x), dtype="float64")

    def _imponer_frontera(u_vec):
        if dirichlet_a:
            u_vec[0] = ua
        else:
            u_vec[0] = u_vec[1] - dx * ua
        if dirichlet_b:
            u_vec[-1] = ub
        else:
            u_vec[-1] = u_vec[-2] + dx * ub

    _imponer_frontera(u)

    # Si t0 > tf realizar traslaccion
    intervalo = tf - t0

    if dt is None:
        dt_base = intervalo / float(4 * N)
        if metodo == "explicito" and alfa > 0.0:
            dt_cfl = dx2 / (2.0 * alfa)
            dt = min(dt_base, 0.9 * dt_cfl)
        else:
            dt = dt_base

    pasos = max(1, int(ceil(intervalo / dt)))
    dt = intervalo / pasos
    sigma = alfa * dt / dx2

    if metodo == "implicito":
        M = N - 1
        A = lil_matrix((M, M), dtype="float64")
        A.setdiag((1.0 + 2.0 * sigma) * ones(M))
        if M > 1:
            A.setdiag((-sigma) * ones(M - 1), 1)
            A.setdiag((-sigma) * ones(M - 1), -1)
        if not dirichlet_a:
            A[0, 0] = 1.0 + sigma
        if not dirichlet_b:
            A[M - 1, M - 1] = 1.0 + sigma
        A = A.tocsc()
        LU = splu(A)

    if dibujar:
        figure()

    errores = []
    t = t0

    for k in range(pasos):
        t_siguiente = t0 + (k + 1) * dt

        if metodo == "implicito":
            rhs = u[1:-1] + dt * f(x[1:-1], t_siguiente)
            if rhs.size:
                if dirichlet_a:
                    rhs[0] += sigma * ua
                else:
                    rhs[0] += sigma * dx * ua
                if dirichlet_b:
                    rhs[-1] += sigma * ub
                else:
                    rhs[-1] += -sigma * dx * ub
                u[1:-1] = LU.solve(rhs)
            _imponer_frontera(u)

        else:
            _imponer_frontera(u)
            lap = u[2:] - 2.0 * u[1:-1] + u[:-2]
            u[1:-1] = u[1:-1] + sigma * lap + dt * f(x[1:-1], t)
            _imponer_frontera(u)

        exacta_actual = fexacta(x, t_siguiente) if fexacta is not None else None
        if exacta_actual is not None:
            errores.append(max(abs(u - exacta_actual)))

        if dibujar and (k % 25 == 0 or k == pasos - 1):
            clf()
            plot(x, u, "b", label="Aproximacion")
            if exacta_actual is not None:
                plot(x, exacta_actual, "r", label="Exacta")
            title(f"t = {round(t_siguiente, 3)}, numero iteracciones N={N}")
            legend(["Aproximacion", "Exacta"])
            xlabel("x")
            pause(0.05)
            draw()

        t = t_siguiente

    print("Tiempo de ejecucion:", format(time.time() - t1))

    if dibujar:
        show()

    return max(errores) if errores else None


###########################################################################################
###########################################################################################
###########################################################################################
###########################################################################################
###########################################################################################
###########################################################################################
###########################################################################################
###########################################################################################
###########################################################################################
###########################################################################################
###########################################################################################
###########################################################################################
###########################################################################################
###########################################################################################
###########################################################################################
###########################################################################################
###########################################################################################
###########################################################################################


def f0_2(x, t):
    return sin(x) + t * sin(x)


def exacta_2(x, t):
    return t * sin(x)


def u0(x):
    return 0.0 * x


def fnula(x, t):
    return 0.0 * x


def sin0(x):
    return sin(x)


def f1_exacta(x, t):
    return sin(x) * exp(-t)


def neumann_b(t):
    return t


def exactaP(x, t):
    return 1 + sin(pi * x) * exp(-t)


def fuenteP(x, t):
    return (pi**2 - 1) * sin(pi * x) * exp(-t)


def uoP(x):
    return 1 + sin(pi * x)


def ubP(t):
    return -pi * exp(-t)


def poly_f1(x):
    return pow(x, 5) - 20 * pow(x, 3)


def exacta_poly(x):
    return pow(x, 5)


def u0_ejemplo(x):
    return sin(pi * x)


def f_ejemplo(x, t):
    return zeros_like(x)


def fexacta_ejemplo(x, t):
    return t * sin(x)


error = 0
nuevo_error = 0

calculo = [100, 200, 400]
# calculo = [50, 100, 200]

# Controla el caso a realizar
caso = 3

# fmt: off
if calculo.__len__() != 0:
    print(f"Caso {caso}")

    for i in calculo:
        # ---------- CASO 0 ---------------- #
        if caso == 0:
            # contorno_directo_sparse_v0(0, pi, i, 1.0, 0, 0, f0_1, uexacta_1)
            # nuevo_error = contorno_directo_sparse_v1(
            #     0, pi, i, 1.0, 0, 0, f0_1, uexacta_1
            # )
            nuevo_error = contorno_directo_sparse_v2(0, pi, i, 1.0, 0, 0, f0_1, uexacta_1)
            # contorno_directo_densa(0, pi, 400, 1.0, 0, 0, f0)

        # ---------- CASO 1 ---------------- #
        if caso == 1:
            # nuevo_error = caso1(0, pi, i, 1.0, 0, -1.0, f0_1, uexacta_1, True)
            # nuevo_error = caso1_v1(0, 1.0, i, 1.0, 0.0, 5.0, poly_f1, exacta_poly, False)
            nuevo_error = caso1_v2(0, 1.0, i, 1.0, 0.0, 5.0, poly_f1, exacta_poly, False)


        # ---------- CASO 2 ---------------- #
        if caso == 2:
            # nuevo_error = caso2(0, pi, i, 1.0, 1.0, -1.0, f0_1, False, False, uexacta_1, False)
            nuevo_error = caso2(0, 1.0, i, 1.0, 0.0, 5.0, poly_f1, True, False, exacta_poly, False)

        # ---------- CASO 3 ---------------- #
        if caso == 3:
            
            # ---------- Implicito ---------------- #
            # Error en el espacio
            # nuevo_error = caso3_implicito(0, pi, i, 0, 4, 1000, 1.0, 0.0, 0.0, sin0, fnula, f1_exacta, False)
            # Error en el tiempo
            # nuevo_error = caso3_implicito(0, pi, 400, 0, 4, i, 1.0, 0.0, 0.0, sin0, fnula, f1_exacta, False)
            
            # nuevo_error = caso3_implicito(0, pi, i, 0, 4, i, 1.0, 0.0, 0.0, sin0, fnula, f1_exacta, False)

            # Error en el tiempo
            # nuevo_error = caso3_implicito_neumann_b(0, 1.0, 200, 0, 4, i, 1.0, 1.0, ubP, uoP, fuenteP, exactaP, False)
            
            # Error en el espacio
            nuevo_error = caso3_implicito_neumann_b(0, 1.0, i, 0, 4, 200000, 1.0, 1.0, ubP, uoP, fuenteP, exactaP, False)
            
            # nuevo_error = caso3_implicito_neumann_b(0, 2*pi, i, 0, 4.0, 1000, 1.0, 0.0, neumann_b, u0, f0_2, exacta_2, False)

            # ---------- Explicito ---------------- #
            # nuevo_error = caso3_explicito(0, pi, i, 0, 4, 1.0, 0, 0, sin0, fnula, f1_exacta, False)

            # ---------- 0-metodo ---------------- #
            cte = 1
            # nuevo_error = caso3_theta_metodo(0, 2 * pi, i, 0, 4, 4000, 1.0, 0, 0, u0, f0_2, cte, exacta_2, False)
           
            # nuevo_error = caso3_lineas_condiciones(0, pi, i, 0, 4, 1.0, 0.0, 0.0, u0, f0_2, exacta_2, "implicito", True, True)

            
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


# El orden que obtengo en el desarrollo del metodo de diferencias finitas es un maximo teorico, i.e,
#   la velocidad de convergencia del metodo cuando la malla tiende hacia el propio intervalo no puede ser mayor que
#   el resultado teorico. Lo que pasa con los polinomios (polinomios muy simples, tipo x^2 y productos ) que como tienen
#      un error tan pequeno, realmente ya ha convergido practicamente en su totalidad, luego el error decrece mas lentamente.
