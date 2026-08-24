"""
Visualizacion 2D para resultados de transporte
Lee mallas 2D de salida y genera mapas de calor / animaciones.
"""

import matplotlib
import sys, getopt
import ast
import time
from numpy import *

# from pylab import *
from matplotlib.pyplot import *

ion()


def calcular_rango(fichero):
    """
    Funcion auxiliar para calcular los maximos y minimos para ajustar colorbar al pintar

    Parameters
    ----------
    fichero : str
        nombre el fichero.

    Returns
    -------
    rango: array
        valores [xmin,xmax,ymin,ymax,zmin,zmax]

    """
    print("Calculando rango de ventana...")
    xmin = 1.0e30
    xmax = -1.0e30
    ymin = 1.0e30
    ymax = -1.0e30
    zmin = 1.0e30
    zmax = -1.0e30

    with open(fichero) as f:
        ln = f.readline()
        x = [float(x0) for x0 in ln.split()]
        xmin = x[0]
        xmax = x[1]
        ymin = x[3]
        ymax = x[4]
        for ln in f:
            tt = [float(x0) for x0 in ln.split()]
            t0 = tt[0]
            z = array(tt[1::])
            zmin = min([zmin, min(z)])
            zmax = max([zmax, max(z)])

    if ymax > ymin:
        espacio = 0.1 * (ymax - ymin)
    else:
        espacio = 0.01

    ymin = ymin - espacio
    ymax = ymax + espacio

    return [xmin, xmax, ymin, ymax, zmin, zmax]


def pintar(fichero, np=1):
    """
    pintar(fichero,np)

    Parameters
    ----------
    fichero : str
        nombre del fichero.
    np : int
        cada cuantas lineas del fichero se pintaran. Por defecto todas

    Returns
    -------
    None.

    """

    print("Abriendo fichero %s" % fichero)
    rango = calcular_rango(fichero)
    print("Pintando cada %i lineas" % np)
    zmin = rango[4]
    zmax = rango[5]

    niveles = linspace(zmin, zmax, 21)
    fig1, ax2 = subplots(constrained_layout=True)

    with open(fichero) as f:
        ln = f.readline()
        x = [float(x0) for x0 in ln.split()]
        xmin = x[0]
        xmax = x[1]
        npx = int(x[2])
        ymin = x[3]
        ymax = x[4]
        npy = int(x[5])
        x = linspace(xmin, xmax, npx + 1)
        y = linspace(ymin, ymax, npy + 1)
        X, Y = meshgrid(x, y)
        iter = 0
        for ln in f:
            if (iter == 0) | ((np > 0) & (iter >= np)) | (np <= 0):
                tt = [float(x0) for x0 in ln.split()]
                t0 = tt[0]
                Z = array(tt[1::])
                Z = Z.reshape((npy + 1, npx + 1))
                type(Z)
                size(Z)
                cla()
                img = contourf(X, Y, Z, levels=niveles)
                if iter == 0:
                    colorbar(img)
                cad = "Tiempo:" + str(t0)
                title(cad)
                draw()
                pause(0.1)
                iter = 0
            iter = iter + 1


def main(argv):
    try:
        opts, args = getopt.getopt(argv, "h:f:n:")
    except getopt.GetoptError:
        print("pinta2d.py -f <fichero> -n <salto>")
        sys.exit(2)
    np = 1
    for opt, arg in opts:
        if opt == "-h":
            print("pinta2d.py -f <fichero> -n <salto>")
            sys.exit()
        elif opt == "-f":
            fichero = arg.strip(" ")
        elif opt in ("-n"):
            aux = arg.strip(" ")
            np = ast.literal_eval(aux)
    # fichero=raw_input('Nombre del fichero:')
    # np=raw_input('Visualizar cada: ')
    # np=ast.literal_eval(np)

    print("Abriendo fichero %s" % fichero)
    rango = calcular_rango(fichero)
    print("Pintando cada %i lineas" % np)
    zmin = rango[4]
    zmax = rango[5]

    niveles = linspace(zmin, zmax, 21)
    fig1, ax2 = subplots(constrained_layout=True)

    with open(fichero) as f:
        ln = f.readline()
        x = [float(x0) for x0 in ln.split()]
        xmin = x[0]
        xmax = x[1]
        npx = int(x[2])
        ymin = x[3]
        ymax = x[4]
        npy = int(x[5])
        x = linspace(xmin, xmax, npx + 1)
        y = linspace(ymin, ymax, npy + 1)
        X, Y = meshgrid(x, y)
        iter = 0
        for ln in f:
            if (iter == 0) | ((np > 0) & (iter >= np)) | (np <= 0):
                tt = [float(x0) for x0 in ln.split()]
                t0 = tt[0]
                Z = array(tt[1::])
                Z = Z.reshape((npy + 1, npx + 1))
                type(Z)
                size(Z)
                cla()
                img = contourf(X, Y, Z, 20)
                if iter == 0:
                    colorbar(img)
                cad = "Tiempo:" + str(t0)
                title(cad)
                draw()
                pause(0.1)
                iter = 0
            iter = iter + 1
    input("Pulse intro para finalizar")
    sys.exit()


if __name__ == "__main__":
    try:
        opts, args = getopt.getopt(sys.argv[1:], "h:f:n:")
    except:
        opts = 0

    if len(opts) == 0:
        # Opcion para ejecutarlo desde Spyder
        fichero = "prueba.txt"
        np = 1
        pintar(fichero, np)
    else:
        # Opcion para ejecutarlo desde un terminal pasando los argumentos
        # pinta1d.py -f <fichero> -n <salto>
        main(sys.argv[1:])
