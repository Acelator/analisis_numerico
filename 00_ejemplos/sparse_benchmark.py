"""
Ejemplo sparse - benchmark dispersa vs densa 1000x1000
Comparacion de memoria y tiempo con scipy.sparse.
"""

import numpy as np
from scipy.sparse import *
from scipy.stats import uniform

np.random.seed(seed=42)
data = uniform.rvs(size=1000000, loc=0, scale=2)
data = np.reshape(data,(1000,1000))

data[data<1] = 0
print(data[0:5,0:5])

#print('-------------------')
data_size = data.nbytes/(1024**2)
print('El tamano de la matriz completa es %f Mb' % data_size)

data_csr = csr_matrix(data)
#print(data_csr)
data_size_csr = data_csr.data.size/(1024**2)
print('El tamano de la matriz sparse es %f Mb' % data_size_csr)
