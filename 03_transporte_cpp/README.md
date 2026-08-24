# 03 - Transporte hiperbolico (C++ OpenMP)

- `calor_transporte_1d.cpp` - Transporte `u_t + c u_x =0` (upwind), Burgers `u_t+u u_x=alfa u_xx`, aguas someras 1D Saint-Venant `h_t+q_x=0`. Paralelizado con OpenMP.
- `calor_transporte_2d.cpp` - Extension 2D.
- `visual_1d.py`, `visual_2d.py` - Visualizacion de salidas.

Compilacion: `g++ -O2 -fopenmp calor_transporte_1d.cpp -o calor1d && ./calor1d 1`

Detalles: `../docs/enunciados_resumidos.md#03_transporte_cpp`
