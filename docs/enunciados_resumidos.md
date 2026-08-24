# Resumenes de problemas

## 01_edp_1d - Problemas de contorno y evolucion 1D

### Eliptica 1D Dirichlet (`eliptica_1d.py`)
Problema estacionario: `u(x) - alfa u''(x) = f(x)` en `(a,b)`, con `u(a)=ua`, `u(b)=ub`. Caso de referencia `a=0, b=pi, alfa=1, ua=ub=0, f=2 sin(x)` con solucion exacta `u=sin(x)`. Discretizacion por diferencias finitas centradas de segundo orden en malla uniforme. Estudio de diferentes tratamientos de contorno Dirichlet (imposicion directa, penalizacion con parametro grande, simetrizacion) y comparacion de matrices dispersas vs densas. Verificacion numerica de orden 2 por refinamiento de malla.

Variante Neumann: `u(a)=ua , u'(b)=u'b` tratada con tecnica de nodo fantasma simetrico. Orden 2 verificado experimentalmente. Extension a algoritmo general que selecciona tipo Dirichlet/Neumann en cada extremo mediante flags.

### Parabolica 1D (`parabolica_1d.py`)
Problema de evolucion: `u_t - alfa u_xx = f(x,t)` en `(a,b)x(0,T]`, con condiciones Dirichlet o Neumann y dato inicial `u(x,0)=u0(x)`. Semidiscretizacion espacial con diferencias finitas de segundo orden (metodo de lineas). Discretizacion temporal: Euler explicito (condicionalmente estable, CFL `dt <= dx2/(2alfa)`), Euler implicito (incondicionalmente estable) y theta-metodo (estable si `theta>=0.5`, condicional en caso contrario). Medicion experimental de orden espacial y temporal con solucion manufaturada.

### No lineal 1D (`no_lineal_1d.py`)
Problema estacionario no lineal: `-nu u'' - u2 = f` en `(a,b)`, `a=0, b=pi, nu=1, f=cos(x)(1-cos(x))` con `u(a)=1, u(b)=-1`, solucion `u=cos(x)`. Metodos iterativos: (1) punto fijo clasico `A U^{l+1}=F+(U2)^l`, (2) punto fijo con matriz dependiente `A(U^l)U^{l+1}=F` donde `A(U)=1/dx2 M - diag(U)` y `M` tridiagonal `2,-1,-1`, (3) Newton con Jacobiano `J=nu/dx2 D -2 diag(U)`. Penalizacion para Dirichlet y criterio `||U^{l+1}-U^l||inf < 1e-7` o 500 iteraciones.

Problema de evolucion no lineal: `u_t - nu u_xx = f + u2` con Euler implicito en tiempo y resolucion no lineal por paso temporal mediante punto fijo/Newton.

### Schwarz 1D (`schwarz_1d.py`)
Descomposicion de dominios de Schwarz para `u - nu u'' = f` en `(0,L)`, `L=1, nu=2, f=5 exp(-(x-0.5)2)`. Dos subdominios solapados `I1unionI2=[0,1]`, `I1interseccionI2=Ic, |Ic||>0` con solape de `l` nodos (`l=ceil(N/40)` en pruebas). Algoritmo iterativo paralelo: cada subdominio resuelve su problema con condicion Dirichlet interpolada del vecino en la interfaz. Criterio `max|u1-u2|<epsilon` en solape. Variantes: (a) Dirichlet-Dirichlet, (b) Dirichlet-Neumann en `x=L`, (c) tres subdominios, (d) N subdominios arbitrarios. Analisis de velocidad de convergencia en funcion de `l`.

## 02_edp_2d

### Eliptica 2D (`eliptica_2d.py`)
Problema: `u - nu Deltau = f` en `Omega=(0,Lx)x(0,Ly)` con cuatro tipos de contorno: Dirichlet en todo el borde, o mixto con Neumann `du/dn` en `x=Lx` (y variante con Neumann en `y=0`). Solucion exacta manufaturada para validacion. Discretizacion por diferencias finitas centradas en malla tensorial `(Nx+1)x(Ny+1)`, sistema `(Nx+1)(Ny+1)` con tratamiento de esquinas. Implementaciones para Neumann mediante penalizacion y simetrizacion. Orden 2 verificado.

### Parabolica 2D (`parabolica_2d.py`)
Evolucion: `u_t - nu Deltau = f(x,y,t)` con condiciones mixtas Dirichlet/Neumann y dato inicial `g0`. Metodo de lineas: diferencias finitas centradas en espacio + Euler implicito y theta-metodo en tiempo. CFL 2D `nu dt(1/dx2+1/dy2) <= 1/(2(1-2theta))`.

### No lineal 2D (`no_lineal_2d.py`)
Reaccion-difusion: `u_t - Deltau + u2 = f` en dominio rectangular. Tres esquemas por paso temporal: Picard clasico, Picard con correccion diagonal `diag+dt*u`, y Newton con Jacobiano completo, todos con Dirichlet penalizado. Comparacion de convergencia.

## 03_transporte_cpp - Ecuaciones hiperbolicas (C++)

### Transporte 1D (`calor_transporte_1d.cpp`)
Ecuacion de transporte lineal `u_t + c u_x = 0` con difusion artificial upwind, condicion inicial senoidal/gaussiana y frontera periodica o Dirichlet. CFL `dt = phi dx/|c|`. Ecuacion de Burgers no lineal `u_t + u u_x = alfa u_xx` con `dt = gamma dx/max|u|` adaptativo. Sistema de aguas someras (Saint-Venant) 1D: `h_t+q_x=0`, `q_t+(q2/h+g h2/2)_x=0, g=9.81`, con CFL `dt=gamma dx / max(|q/h|+sqrt(g h))`, casos periodico y frontera abierta por extrapolacion. Implementacion paralelizada con OpenMP (`parallel for`, `reduction(max)`).

### Transporte 2D (`calor_transporte_2d.cpp`)
Extension 2D de transporte y aguas someras con fondo plano en malla cartesiana, analogo 1D.

## 04_edos_pvi - Problemas de valor inicial (EDOs)

### Metodos unipaso (`metodos_pvi.py`)
Biblioteca de metodos para `y'=f(t,y), y(a)=y0`: Euler explicito, Taylor 2 y 3, Heun, punto medio, RK4 (escalar y sistemas), Euler implicito. Casos de prueba: `y'=0.5(t2-y)` en `[0,10]` exacta `t2-4t+8-7e^{-t/2}`, problema de deposito `S'=6-S/10`, sistemas Lotka-Volterra, oscilador rigido `x''+20x'+101x=0` exacta `e^{-10t}cos t`, modelo SIR y cohete con empuje `T` y masa combustible `mf`, friccion `C v|v|`. Verificacion de orden por cocientes `e_N/e_{2N}`.

### Multipaso (`multipaso.py`)
Adams-Bashforth AB2/AB3/AB4, Adams-Moulton AM3/AM4, predictor-corrector ABM. Arranque con metodos unipaso, estudio de estabilidad absoluta, frontera de estabilidad (`locfron`), orden y control de paso. Problema de referencia `y'=-y+2 sin t` y sistemas 3x3.

### Tiro (`tiro.py`)
Metodo del tiro para problemas de contorno de segundo orden: reformulacion como PVI con parametro y resolucion de ecuacion no lineal `G(theta0)=x` mediante biseccion/secante. Caso cohete sin motor `x=50, v0=25, M=7.5, C=0.03`, RK4 `h=0.01`, secante `theta0=pi/16, pi/4`. Tambien tiro para EDO de segundo orden generica con `exacta sin(2pix)/(1+4pi2)`.

### Diferencias finitas 1D (`diferencias_finitas_1d.py`)
Contorno `-u''+q u = f` con malla no uniforme, integracion por diferencias finitas y FEM. Calculo de error y orden.

### Calor 1D (`calor_1d.py`)
Ecuacion del calor `u_t - c u_xx =0` Dirichlet con metodos explicito (`dt<=dx2/(2c)`), implicito y Crank-Nicolson. Extensiones: adveccion-difusion `u_t -c u_xx + v u_x=0` (`v=0,0.01,1,5,10`) con upwind si `v>=0`, y ecuacion de ondas `u_tt=c2 u_xx` explicita `k=h/c`. Comparaciones graficas y animacion.

## 05_fem - Elementos finitos

### FEM 1D (`fem_1d.py`)
FEM P1 para `-u''+q u = f` con matrices de rigidez `R` y masa `M` (`R = diags(-1/h, 1/h_{i-1}+1/h_i, -1/h)`, `M = diags(h/6, (h_{i-1}+h_i)/3, h/6)`), integracion punto medio, malla no uniforme. Caso `q>0`, condiciones Dirichlet homogeneas y no homogeneas.

### FreeFem++ (`freefem/`)
`mesh Th=square`, `fespace Vh(Th,P1/P2)`, `problem laplace = int2d(dx(u)*dx(v)+dy(u)*dy(v)) - int2d(f*v) + on(1,2,3,4,u=g)`, calculo de error L2 `sqrt(int2d((u-uex)2))` y orden `log2(e_old/e)`. Laplace P1 orden 2, P2 orden 3; calor en L.

## 06_raices_interpolacion

Busqueda de raices univariantes: biseccion (`N=ceil(log(b-a)-log epsilon)/log2`), regula falsi, secante, punto fijo (`kepler`), Newton. Analisis de error de redondeo `epsilon` (`while 1+x>1`), aproximacion `e~=(1+1/n)^n`, series `Sigma1/sqrtk`. Interpolacion: Newton con diferencias divididas, evaluacion Horner `evalpol_eqd`, nodos Chebyshev, interpolacion lineal a trozos. Comparacion de error segun distribucion de nodos.

## 07_algebra_lineal

Algebra lineal numerica directa e iterativa: normas vectoriales `||*||_p` y matriciales, convergencia `p->inf`, eliminacion gaussiana con pivote parcial, Gauss-Jordan, factorizacion LU y Cholesky, metodos iterativos Jacobi, Gauss-Seidel, SOR (relajacion), y metodo de la potencia/potencia inversa para autovalores. Analisis de condicionamiento `cond`, numero de condicion, radio espectral. Practicas UMA Metodos Numericos II (Prof. M.J. Castro / F.J. Palma) con 10 notebooks (normas -> factorizaciones -> iterativos -> potencia).
