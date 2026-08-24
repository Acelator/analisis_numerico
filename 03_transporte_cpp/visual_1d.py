"""
Visualizacion 1D para resultados de transporte
Lee ficheros de salida del simulador C++ y genera graficas con matplotlib.
"""

import matplotlib
import sys, getopt
import ast
import time
from numpy import *

# from pylab import *
from matplotlib.pyplot import *


def calcular_rango(fichero):
    """
    Funcion auxiliar para calcular los maximos y minimos para ajustar la ventana al pintar

    Parameters
    ----------
    fichero : str
        nombre el fichero.

    Returns
    -------
    rango: array
        valores [xmin,xmax,ymin,ymax]

    """
    print("Calculando rango de ventana...")
    ymin = 1.0e30
    ymax = -1.0e30

    with open(fichero) as f:
        ln = f.readline()
        x = [float(x0) for x0 in ln.split()]
        xmin = x[0]
        xmax = x[-1]
        for ln in f:
            tt = [float(x0) for x0 in ln.split()]
            t0 = tt[0]
            y = tt[1::]
            ymin = min([ymin, min(y)])
            ymax = max([ymax, max(y)])

    if ymax > ymin:
        espacio = 0.1 * (ymax - ymin)
    else:
        espacio = 0.01

    ymin = ymin - espacio
    ymax = ymax + espacio

    return [xmin, xmax, ymin, ymax]


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
    print()
    with open(fichero) as f:
        ln = f.readline()
        x = [float(x0) for x0 in ln.split()]
        iter = 0
        for ln in f:
            if (iter == 0) | ((np > 0) & (iter >= np)) | (np <= 0):
                tt = [float(x0) for x0 in ln.split()]
                t0 = tt[0]
                y = tt[1::]
                clf()
                plot(x, y)
                axis(rango)
                cad = "Tiempo:" + str(t0)
                title(cad)
                pause(0.05)
                iter = 0
            iter = iter + 1


def main(argv):
    try:
        opts, args = getopt.getopt(argv, "h:f:n:")
    except getopt.GetoptError:
        print("pinta1d.py -f <fichero> -n <salto>")
        sys.exit(2)
    np = 1
    if len(opts) == 0:
        print("pinta1d.py -f <fichero> -n <salto>")
        sys.exit(2)
    for opt, arg in opts:
        if opt == "-h":
            print("pinta1d.py -f <inputfile> -n <outputfile>")
            sys.exit()
        elif opt == "-f":
            fichero = arg.strip(" ")
        elif opt in ("-n"):
            aux = arg.strip(" ")
            np = ast.literal_eval(aux)
    print("Abriendo fichero %s" % fichero)
    rango = calcular_rango(fichero)
    print("Pintando cada %i lineas" % np)
    print()
    with open(fichero) as f:
        ln = f.readline()
        x = [float(x0) for x0 in ln.split()]
        iter = 0
        for ln in f:
            if (iter == 0) | ((np > 0) & (iter >= np)) | (np <= 0):
                tt = [float(x0) for x0 in ln.split()]
                t0 = tt[0]
                y = tt[1::]
                clf()
                plot(x, y)
                axis(rango)
                cad = "Tiempo:" + str(t0)
                title(cad)
                pause(0.05)
                iter = 0
            iter = iter + 1

    try:
        show(block=True)
    except TypeError:
        show()

    input("Pulse intro para finalizar")
    sys.exit()


if __name__ == "__main__":
    try:
        opts, args = getopt.getopt(sys.argv[1:], "h:f:n:")
    except:
        opts = 0

    if len(opts) == 0:
        # Opcion para ejecutarlo desde Spyder
        fichero = "data.txt"
        np = 1
        pintar(fichero, np)
    else:
        # Opcion para ejecutarlo desde un terminal pasando los argumentos
        # pinta1d.py -f <fichero> -n <salto>
        main(sys.argv[1:])
