# 01 - EDP 1D

Diferencias finitas de segundo orden para problemas 1D.

- `eliptica_parabolica_1d.py` - Eliptica `u - alfa u'' = f` con Dirichlet (directa, penalizacion, simetria) y Neumann (nodo fantasma); parabolica `u_t - alfa u_xx = f` por metodo de lineas (explicito, implicito, theta-metodo). Orden 2 verificado.
- `no_lineal_1d.py` - No lineal `-nu u'' - u2 = f` con punto fijo `A U^{l+1}=F+(U2)^l`, matriz dependiente y Newton `J=nu/dx2 D -2diag(u)`. Evolucion `u_t - nu u_xx = f+u2`.
- `schwarz_1d.py` - Schwarz alternante con solape `l`, variantes Dirichlet-Dirichlet / Dirichlet-Neumann y extension a N subdominios.

Dependencias: `numpy`, `scipy.sparse`, `matplotlib`. Detalles: `../docs/enunciados_resumidos.md#01_edp_1d`
