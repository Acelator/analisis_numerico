"""
Ejemplo sparse - matriz dispersa CSR 3x3
Demo basica de scipy.sparse.csr_matrix.
"""

import numpy as np
from scipy.sparse import * # type: ignore

A = csr_matrix([[1, 2, 0], [0, 0, 3], [4, 0, 5]])
# A = csr_matrix([[1, 2, 0], [0, 0, 3], [4, 0, 5]], copy=True)
Ac = np.array([[1, 2, 0], [0, 0, 3], [4, 0, 5]])

print(A)
print('------------------------')
print(Ac)

# Para pasarlo a np array 
completa = A.toarray()
