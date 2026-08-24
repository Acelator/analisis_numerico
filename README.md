# Analisis Numerico - Coleccion de Metodos Numericos y EDPs

Coleccion de implementaciones propias desarrolladas durante el Grado en Matematicas (Universidad de Malaga) para asignaturas de Analisis Numerico y Calculo Cientifico. El repositorio reune metodos numericos para EDOs, EDPs y algebra lineal, con enfasis en diferencias finitas, elementos finitos y esquemas hiperbolicos.

## Estructura

| Carpeta | Contenido | Lenguaje | Ficheros principales |
|---|---|---|---|
| `01_edp_1d/` | EDP eliptica/parabolica 1D, no lineal y Schwarz | Python | `eliptica_parabolica_1d.py` (Dirichlet/Neumann, nodo fantasma, theta-metodo), `no_lineal_1d.py` (punto fijo, Newton), `schwarz_1d.py` (descomposicion de dominios) |
| `02_edp_2d/` | EDP 2D eliptica/parabolica y reaccion-difusion | Python | `eliptica_parabolica_2d.py` (Dirichlet/Neumann 2D), `no_lineal_2d.py` (Picard/Newton) |
| `03_transporte_cpp/` | Transporte hiperbolico y aguas someras 1D/2D | C++17 + OpenMP | `calor_transporte_1d.cpp`, `calor_transporte_2d.cpp` (Burgers, Saint-Venant), `visual_1d.py`, `visual_2d.py` |
| `04_edos_pvi/` | EDOs - metodos unipaso y multipaso | Python | `metodos_pvi.py` (Euler, Heun, RK4), `sistemas_pvi.py` (Lotka-Volterra, cohete), `multipaso.py` (AB/AM), `tiro.py` (metodo del tiro), `diferencias_finitas_1d.py`, `calor_1d.py` |
| `05_fem/` | Elementos finitos 1D y 2D | Python + FreeFem++ | `fem_1d.py` (P1), `freefem/laplace.edp`, `calor.edp`, `prac8/*.edp` |
| `06_raices_interpolacion/` | Raices e interpolacion | Python | `raices_interpolacion.py` (biseccion, secante, Newton, Chebyshev, trozos) |
| `07_algebra_lineal/` | Algebra lineal numerica | Python + Jupyter | `algebra.py` (Gauss, LU, Jacobi, SOR, potencia), `01_normas.ipynb` ... `10_descomposiciones.ipynb` (practicas UMA, Prof. M.J. Castro / F.J. Palma) |
| `00_ejemplos/` | Demos de matrices dispersas | Python | `sparse_demo.py`, `sparse_benchmark.py` |
| `docs/` | Resumenes anonimizados de problemas | Markdown | `enunciados_resumidos.md` |

Ver `README.md` en cada subcarpeta para detalle de metodos y uso.

## Requisitos

- Python >=3.10, `numpy`, `scipy`, `matplotlib`
- C++17 con OpenMP (`g++ -fopenmp`), opcional `FreeFem++ >=4.12` para `.edp`
- Jupyter para notebooks (`jupyter lab`)

## Uso rapido

```bash
# EDP 1D
python 01_edp_1d/eliptica_parabolica_1d.py
python 01_edp_1d/schwarz_1d.py

# Transporte 1D (paralelo)
g++ -O2 -fopenmp 03_transporte_cpp/calor_transporte_1d.cpp -o /tmp/calor1d
/tmp/calor1d 1  # 1=transporte lineal, 2=gaussiana, 3=Burgers, 41/42=aguas someras
python 03_transporte_cpp/visual_1d.py

# EDOs
python 04_edos_pvi/metodos_pvi.py
python 04_edos_pvi/sistemas_pvi.py

# FEM 1D
python 05_fem/fem_1d.py
# FEM 2D (FreeFem++)
FreeFem++ 05_fem/freefem/laplace.edp
```

## Verificacion de orden

Los ficheros incluyen calculo de error frente a solucion exacta y estimacion de orden por refinamiento (`log2(e_old/e)` o cociente `e_N/e_2N`). Ver comentarios en cabecera y `docs/enunciados_resumidos.md` para detalles de cada caso.

## Licencia

MIT - ver `LICENSE`.
