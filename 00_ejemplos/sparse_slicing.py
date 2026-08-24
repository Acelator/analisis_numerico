"""
Ejemplo sparse - slicing con lil_matrix y setdiag
Operaciones basicas con matrices dispersas.
"""

from scipy.sparse import lil_matrix
from scipy.sparse.linalg import spsolve
from numpy.linalg import solve, norm
from numpy.random import rand

A = lil_matrix((1000, 1000))
A[0, :100] = rand(100)
A[1, 100:200] = A[0, :100]

A.setdiag(rand(1000))
print(A[0:5,0:5])

B = A[0:5,0:5]
print(B.toarray())

data_size = A.data.nbytes/(1024**2)
print('El tamano de la matriz completa es %f Mb' % data_size)
