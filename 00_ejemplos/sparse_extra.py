"""
Ejemplo sparse - operaciones adicionales
"""


import numpy as np
from scipy.sparse import coo_matrix

filas = np.array([0,1,1,3,4])
columnas = np.array([0,2,4,3,4])
datos = np.array([1,2,3,4,5],dtype=float)

mat_coo = coo_matrix((datos,(filas,columnas)))
print(mat_coo)
print('------------')
print(mat_coo.tocsc())