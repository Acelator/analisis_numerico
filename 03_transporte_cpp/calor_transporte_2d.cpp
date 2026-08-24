// Transporte hiperbolico 2D - extension del modelo 1D
// Transporte lineal y aguas someras 2D con fondo plano, mallado 2D y OpenMP.
// Ver docs/enunciados_resumidos.md#03_transporte_cpp
// Compilacion: g++ -O2 -fopenmp calor_transporte_2d.cpp -o calor_transporte_2d

// Tener cuidado que al declarar un private en una directiva de omp, se crea una copia de la variable en cada hebra,
//      aunque en la primera iteraccion que la hebra realize del bucle, la variable no estara inicializada.
//      Habria que usar firstprivate para conservar valor previo como valor de partida

#include <cassert>
#include <math.h>
#include <stdio.h>
#include <omp.h>
#include <iostream>
#include <fstream>
#include <cstdlib>
#include <time.h>
#include <vector>

using namespace std;
#define _USE_MATH_DEFINES
const double G = 9.81;

int caso_1(int argc, char **argv);
int caso_2(int argc, char **argv);

int main(int argc, char **argv)
{
    int caso = atoi(argv[1]);
    bool retFlag;
    if (caso == 1)
        return caso_1(argc, argv);
    else if (caso == 2)
        return caso_2(argc, argv);
    else
    {
        std::printf("Dicho numero no pertence a ningun caso \n");
        std::printf("Numeros validos son 1-2");
        return 1;
    }

    return 0;
}

// /////////////////////////////////////////////
// CASO 1 -> Transporte 2D

double cond_inicial_1(double x0, double y0)
{
    double ci;
    ci = (x0 * x0 + y0 * y0 < 0.25);
    return ci;
}

int caso_1(int argc, char **argv)
{
    double a, b, c, d, T, c1, c2, cfl, tiempo;
    int npx, npy;
    char *fichero_salida;
    double dx, dy, dt, dtg, tg;
    FILE *fp;
    clock_t t_ini, t_fin;
    double secs;
    t_ini = omp_get_wtime();

    if (argc != 15)
    {
        printf("Uso:\n");
        printf("%s", argv[0]);
        printf("a b c d T npx npy cfl c1 c2 fichero_salida  dtg nh\n");
        printf("a: Comienzo del intervalo, direccion x.\n");
        printf("b: Final del inervalo, direccion x.\n");
        printf("c: Comienzo del intervalo, direccion y.\n");
        printf("d: Final del inervalo, direccion y\n");
        printf("T: Tiempo total de integracion.\n");
        printf("npx: N. de particiones del intervalo [a,b]\n");
        printf("npy: N. de particiones del intervalo [c,d]\n");
        printf("cfl: Coef. estabilidad.\n");
        printf("c1: Vel. transporte en la direccion x.\n");
        printf("c2: Vel. transporte en la direccion y.\n");
        printf("Nombre fichero de salida.\n");
        printf("Dt guardado.\n");
        printf("Numero hilos.\n");
        return -1;
    };

    a = atof(argv[2]);
    b = atof(argv[3]);
    c = atof(argv[4]);
    d = atof(argv[5]);
    T = atof(argv[6]);

    npx = atoi(argv[7]);
    npy = atoi(argv[8]);
    cfl = atof(argv[9]);
    c1 = atof(argv[10]);
    c2 = atof(argv[11]);
    fichero_salida = argv[12];
    dtg = atof(argv[13]);

    int nh = atoi(argv[14]);
    omp_set_num_threads(nh);

    // Verificamos que los valores son validos
    assert(T > 0);
    assert(cfl > 0 && cfl < 1);
    assert(a < b);

    double *sol0; // solucion en el instante n
    double *sol1; // solucion en el instasnte n+1
    double *x;    // discretizacion del dominio espacial
    double *y;    // discretizacion del dominio espacial
    double *aux;  // puntero auxiliar para el intercambio de datos

    // creacion de los arrays
    sol0 = new double[(npx + 1) * (npy + 1)];
    sol1 = new double[(npx + 1) * (npy + 1)];
    x = new double[npx + 1];
    y = new double[npy + 1];

    dx = (b - a) / double(npx);
    dy = (d - c) / double(npy);

    // Imponemos la condiccion de estabilidad numerica
    dt = 0.5 * cfl * min(dx / (fabs(c1) + 1e-16), dy / (fabs(c2) + 1e-16));
    printf("Dt: %12.8f\n", dt);

/* Inicializamos en x e y el mallado del dominio */
// ! Realmente no seria conveniente ya que el overhead es mucho mayor para operaciones tan triviales
#pragma omp parallel for
    for (int i = 0; i <= npx; i++)
    {
        x[i] = a + dx * double(i);
    }
#pragma omp parallel for
    for (int j = 0; j <= npy; j++)
    {
        y[j] = c + dy * double(j);
    }

#pragma omp parallel for
    for (int i = 0; i <= npy; i++)
    {
        // Lo realizamos en el interior para reducir overhead
        for (int j = 0; j <= npx; j++)
        {
            int k = i * (npx + 1) + j;
            sol0[k] = cond_inicial_1(x[j], y[i]);
            sol1[k] = sol0[k];
        }
    }

    tiempo = 0.0;
    tg = dtg;

    // w -> Write | t -> Text
    fp = fopen(fichero_salida, "wt");
    fprintf(fp, "%12.8f %12.8f %5i %12.8f %12.8f %5i", a, b, npx, c, d, npy);
    fprintf(fp, "\n");

    fprintf(fp, "%15.8f", tiempo);
    for (int i = 0; i <= npy; i++)
    {
        for (int j = 0; j <= npx; j++)
        {
            int ij = i * (npx + 1) + j;
            fprintf(fp, "%15.8f", sol0[ij]);
        }
    }
    fprintf(fp, "\n");

    double cDx, cDy;
    int ij, ip1j, im1j, ijp1, ijm1;

    cout << "cx=" << c1 << endl;
    cout << "cy=" << c2 << endl;

    while (tiempo < T - 1e-9)
    {
        if (tiempo + dt > T)
            dt = T - tiempo;

        printf("Tiempo: %12.8f\n", tiempo + dt);

// #pragma omp parallel for
#pragma omp parallel for private(cDx, cDy, ij, ip1j, im1j, ijp1, ijm1)
        // Excluimos condicciones de contorno
        for (int i = 1; i < npy; i++)
        {
            // Realmente tambien podriamos paralelizar esto directamente para evitar overhead de manejar hebras
            for (int j = 1; j < npx; j++)
            {
                ij = i * (npx + 1) + j;

                // El p1 significa que estamos sumando uno en dicha coordenada, m1 restando
                // Calculamos los puntos necesarios para computar ij
                ip1j = (i + 1) * (npx + 1) + j;
                im1j = (i - 1) * (npx + 1) + j;
                ijp1 = ij + 1;
                ijm1 = ij - 1;

                cDx = (c1 * sol0[ijp1] - c1 * sol0[ijm1] - fabs(c1) * (sol0[ijp1] - 2.0 * sol0[ij] + sol0[ijm1]));
                cDy = (c2 * sol0[ip1j] - c2 * sol0[im1j] - fabs(c2) * (sol0[ip1j] - 2.0 * sol0[ij] + sol0[im1j]));

                sol1[ij] = sol0[ij] - dt / (2.0 * dx) * cDx - dt / (2.0 * dy) * cDy;
            }
        }

        // Las esquinas se calculan despues por separado
        // Condiccion contorno fila 0 y npy (Lados inferiores y superiores)
#pragma omp parallel for private(cDx, cDy, ij, ip1j, im1j, ijp1, ijm1)
        for (int j = 1; j < npx; j++)
        {
            // Caso fila 0:
            ij = j;                     // =0*(npx+1)+j
            ip1j = (npx + 1) + j;       // =(0+1)*(npx+1)+j;
            im1j = npy * (npx + 1) + j; // Condicion periodica
            ijp1 = ij + 1;
            ijm1 = ij - 1;

            cDx = (c1 * sol0[ijp1] - c1 * sol0[ijm1] - fabs(c1) * (sol0[ijp1] - 2.0 * sol0[ij] + sol0[ijm1]));
            cDy = (c2 * sol0[ip1j] - c2 * sol0[im1j] - fabs(c2) * (sol0[ip1j] - 2.0 * sol0[ij] + sol0[im1j]));
            sol1[ij] = sol0[ij] - dt / (2.0 * dx) * cDx - dt / (2.0 * dy) * cDy;

            // Caso fila ny: (Borde superior)
            ij = npy * (npx + 1) + j;
            ip1j = j; // Condicion periodica
            im1j = (npy - 1) * (npx + 1) + j;
            ijp1 = ij + 1;
            ijm1 = ij - 1;

            cDx = (c1 * sol0[ijp1] - c1 * sol0[ijm1] - fabs(c1) * (sol0[ijp1] - 2.0 * sol0[ij] + sol0[ijm1]));
            cDy = (c2 * sol0[ip1j] - c2 * sol0[im1j] - fabs(c2) * (sol0[ip1j] - 2.0 * sol0[ij] + sol0[im1j]));
            sol1[ij] = sol0[ij] - dt / (2.0 * dx) * cDx - dt / (2.0 * dy) * cDy;
        }

        // Cond contorno columna 0 y npx (Borde izquierda y derecho)
#pragma omp parallel for private(cDx, cDy, ij, ip1j, im1j, ijp1, ijm1)
        for (int i = 1; i < npy; i++)
        {
            // Caso columna 0
            ij = i * (npx + 1);         // =i*(npx+1)+0;
            ip1j = (i + 1) * (npx + 1); // = (i+1)*(npx+1)+0;
            im1j = (i - 1) * (npx + 1); // = (i-1)*(npx+1)+0;
            ijp1 = ij + 1;
            ijm1 = i * (npx + 1) + npx; // Condicion periodica

            cDx = (c1 * sol0[ijp1] - c1 * sol0[ijm1] - fabs(c1) * (sol0[ijp1] - 2.0 * sol0[ij] + sol0[ijm1]));
            cDy = (c2 * sol0[ip1j] - c2 * sol0[im1j] - fabs(c2) * (sol0[ip1j] - 2.0 * sol0[ij] + sol0[im1j]));

            sol1[ij] = sol0[ij] - dt / (2.0 * dx) * cDx - dt / (2.0 * dy) * cDy;

            // caso columna npx
            ij = i * (npx + 1) + npx;
            ip1j = (i + 1) * (npx + 1) + npx;
            im1j = (i - 1) * (npx + 1) + npx;
            ijp1 = i * (npx + 1); // Condicion periodica
            ijm1 = ij - 1;

            cDx = (c1 * sol0[ijp1] - c1 * sol0[ijm1] - fabs(c1) * (sol0[ijp1] - 2.0 * sol0[ij] + sol0[ijm1]));
            cDy = (c2 * sol0[ip1j] - c2 * sol0[im1j] - fabs(c2) * (sol0[ip1j] - 2.0 * sol0[ij] + sol0[im1j]));

            sol1[ij] = sol0[ij] - dt / (2.0 * dx) * cDx - dt / (2.0 * dy) * cDy;
        }

        // Quedan los 4 vertices del rectangulo (esquinas)
        {
            int i = 0;
            int j = 0;

            ij = i * (npx + 1) + j;
            ip1j = (i + 1) * (npx + 1) + j;
            im1j = npy * (npx + 1) + j; // Condicion periodica
            ijp1 = ij + 1;
            ijm1 = i * (npx + 1) + npx; // Condicion periodica

            cDx = (c1 * sol0[ijp1] - c1 * sol0[ijm1] - fabs(c1) * (sol0[ijp1] - 2.0 * sol0[ij] + sol0[ijm1]));
            cDy = (c2 * sol0[ip1j] - c2 * sol0[im1j] - fabs(c2) * (sol0[ip1j] - 2.0 * sol0[ij] + sol0[im1j]));

            sol1[ij] = sol0[ij] - dt / (2.0 * dx) * cDx - dt / (2.0 * dy) * cDy;

            i = 0;
            j = npx;

            ij = i * (npx + 1) + j;
            ip1j = (i + 1) * (npx + 1) + j;
            im1j = npy * (npx + 1) + j; // Condicion periodica
            ijp1 = i * (npx + 1);       // Condicion periodica
            ijm1 = ij - 1;

            cDx = (c1 * sol0[ijp1] - c1 * sol0[ijm1] - fabs(c1) * (sol0[ijp1] - 2.0 * sol0[ij] + sol0[ijm1]));
            cDy = (c2 * sol0[ip1j] - c2 * sol0[im1j] - fabs(c2) * (sol0[ip1j] - 2.0 * sol0[ij] + sol0[im1j]));

            sol1[ij] = sol0[ij] - dt / (2.0 * dx) * cDx - dt / (2.0 * dy) * cDy;

            i = npy;
            j = 0;

            ij = i * (npx + 1) + j;
            ip1j = j; // Condicion periodica
            im1j = (i - 1) * (npx + 1) + j;
            ijp1 = ij + 1;
            ijm1 = i * (npx + 1) + npx; // Condicion periodica

            cDx = (c1 * sol0[ijp1] - c1 * sol0[ijm1] - fabs(c1) * (sol0[ijp1] - 2.0 * sol0[ij] + sol0[ijm1]));
            cDy = (c2 * sol0[ip1j] - c2 * sol0[im1j] - fabs(c2) * (sol0[ip1j] - 2.0 * sol0[ij] + sol0[im1j]));

            sol1[ij] = sol0[ij] - dt / (2.0 * dx) * cDx - dt / (2.0 * dy) * cDy;

            i = npy;
            j = npx;

            ij = i * (npx + 1) + j;
            ip1j = j; // Condicion periodica
            im1j = (i - 1) * (npx + 1) + j;
            ijp1 = i * (npx + 1); // Condicion periodica
            ijm1 = ij - 1;

            cDx = (c1 * sol0[ijp1] - c1 * sol0[ijm1] - fabs(c1) * (sol0[ijp1] - 2.0 * sol0[ij] + sol0[ijm1]));
            cDy = (c2 * sol0[ip1j] - c2 * sol0[im1j] - fabs(c2) * (sol0[ip1j] - 2.0 * sol0[ij] + sol0[im1j]));

            sol1[ij] = sol0[ij] - dt / (2.0 * dx) * cDx - dt / (2.0 * dy) * cDy;
        }

        aux = sol0;
        sol0 = sol1;
        sol1 = aux;
        tiempo += dt;

        if (tiempo >= tg || fabs(tiempo - T) < 1e-9)
        {
            fprintf(fp, "%15.8f", tiempo);
            for (int i = 0; i <= npy; i++)
            {
                for (int j = 0; j <= npx; j++)
                {
                    int ij = i * (npx + 1) + j;
                    fprintf(fp, "%15.8f", sol0[ij]);
                }
            }
            fprintf(fp, "\n");
            tg += dtg;
        }
    }

    t_fin = omp_get_wtime();
    secs = (double)(t_fin - t_ini) / CLOCKS_PER_SEC;
    printf("%.16g segundos\n", secs * 1000.0);
    fclose(fp);

    delete[] sol0;
    delete[] sol1;

    delete[] x;
    delete[] y;

    return 0;
}

// /////////////////////////////////////////////
// CASO 2 -> Aguas someras 2D con fondo plano
// FUNCIONES AUXILIARES

// Calcula el vector de Flujo en direccion X: F(U) = [v1, v1^2/v0 + g*v0^2/2, v1*v2/v0]
void calc_flujo_x(double v0, double v1, double v2, double &f0, double &f1, double &f2)
{
    f0 = v1;
    f1 = (v1 * v1) / v0 + 0.5 * G * v0 * v0;
    f2 = (v1 * v2) / v0;
}

// Calcula el vector de Flujo en direccion Y: G(U) = [v2, v1*v2/v0, v2^2/v0 + g*v0^2/2]
void calc_flujo_y(double v0, double v1, double v2, double &f0, double &f1, double &f2)
{
    f0 = v2;
    f1 = (v1 * v2) / v0;
    f2 = (v2 * v2) / v0 + 0.5 * G * v0 * v0;
}

// Condicion inicial
double valor_inicial(double x, double y)
{
    return 1.0 + exp(-4.0 * (x * x + y * y));
}

// Calculo del paso de tiempo estable
double obtener_dt(int npx, int npy, double dx, double dy, double gamma,
                  double *v0, double *v1, double *v2)
{
    double max_vel_x = 0.0;
    double max_vel_y = 0.0;
    int size = (npx + 1) * (npy + 1);

    // Calculamos los nuevos valores de lambda_(x e y)
#pragma omp parallel for reduction(max : max_vel_x, max_vel_y)
    for (int k = 0; k < size; k++)
    {
        double val = v0[k];
        if (val <= 1e-9)
            val = 1e-9;

        double c = sqrt(G * val);
        double vel_x = fabs(v1[k] / val) + c;
        double vel_y = fabs(v2[k] / val) + c;

        if (vel_x > max_vel_x)
            max_vel_x = vel_x;
        if (vel_y > max_vel_y)
            max_vel_y = vel_y;
    }

    // No dividir por cero
    if (max_vel_x < 1e-9)
        max_vel_x = 1e-9;
    if (max_vel_y < 1e-9)
        max_vel_y = 1e-9;

    return 0.5 * gamma * min(dx / max_vel_x, dy / max_vel_y);
}

// Funcion para guardar datos en un fichero especifico
void guardar_estado(FILE *fp, double tiempo, int npx, int npy, double *data)
{
    fprintf(fp, "%15.8f", tiempo);
    for (int i = 0; i <= npy; i++)
    {
        for (int j = 0; j <= npx; j++)
        {
            fprintf(fp, "%15.8f", data[i * (npx + 1) + j]);
        }
    }
    fprintf(fp, "\n");
}

int caso_2(int argc, char **argv)
{
    double a, b, c, d, T, gamma, tiempo;
    int npx, npy;
    string base_fichero;
    double dx, dy, dt{0}, dtg, tg;
    clock_t t_ini, t_fin;
    double secs;

    if (argc != 13)
    {
        printf("Uso incorrecto. Argumentos necesarios:\n");
        printf("a b c d T npx npy gamma base_fichero dtg\n");
        return -1;
    }

    // t_ini = clock();
    t_ini = omp_get_wtime();

    a = atof(argv[2]);
    b = atof(argv[3]);
    c = atof(argv[4]);
    d = atof(argv[5]);
    T = atof(argv[6]);
    npx = atoi(argv[7]);
    npy = atoi(argv[8]);
    gamma = atof(argv[9]);
    base_fichero = argv[10];
    dtg = atof(argv[11]);

    int nh = atoi(argv[12]);

    // FUNDAMENTALLLLLLLLLLLLLLLLLLLLLLLLLLLLLLLLLLLLLLLLLLLLLLLLLLLLLLLLLLL
    // FUNDAMENTALLLLLLLLLLLLLLLLLLLLLLLLLLLLLLLLLLLLLLLLLLLLLLLLLLLLLLLLLLL
    // FUNDAMENTALLLLLLLLLLLLLLLLLLLLLLLLLLLLLLLLLLLLLLLLLLLLLLLLLLLLLLLLLLL
    // FUNDAMENTALLLLLLLLLLLLLLLLLLLLLLLLLLLLLLLLLLLLLLLLLLLLLLLLLLLLLLLLLLL
    omp_set_num_threads(nh);

    double *h0, *q1_0, *q2_0; // Instante n
    double *h1, *q1_1, *q2_1; // Instante n+1
    double *x, *y;
    double *tmp;

    int total_nodos = (npx + 1) * (npy + 1);

    h0 = new double[total_nodos];
    q1_0 = new double[total_nodos];
    q2_0 = new double[total_nodos];
    h1 = new double[total_nodos];
    q1_1 = new double[total_nodos];
    q2_1 = new double[total_nodos];

    x = new double[npx + 1];
    y = new double[npy + 1];

    dx = (b - a) / double(npx);
    dy = (d - c) / double(npy);

    // Inicializacion espacial
    for (int i = 0; i <= npx; i++)
        x[i] = a + dx * double(i);
    for (int j = 0; j <= npy; j++)
        y[j] = c + dy * double(j);

    // Inicializacion valores
#pragma omp parallel for
    for (int i = 0; i <= npy; i++)
    {
        for (int j = 0; j <= npx; j++)
        {
            int k = i * (npx + 1) + j;
            h0[k] = valor_inicial(x[j], y[i]); // Variable 0 (altura)
            q1_0[k] = 0.0;                     // Variable 1 (momento x)
            q2_0[k] = 0.0;                     // Variable 2 (momento y)

            // Inicializamos siguiente instante de tiempo con el mismo valor inicial
            h1[k] = h0[k];
            q1_1[k] = q1_0[k];
            q2_1[k] = q2_0[k];
        }
    }

    tiempo = 0.0;
    tg = dtg;

    // Preparar ficheros de salida
    string fh = base_fichero + "_h.txt";   // h
    string fq1 = base_fichero + "_q1.txt"; // q1
    string fq2 = base_fichero + "_q2.txt"; // q2

    FILE *fp0 = fopen(fh.c_str(), "wt");
    FILE *fp1 = fopen(fq1.c_str(), "wt");
    FILE *fp2 = fopen(fq2.c_str(), "wt");

    // Escribir cabeceras en todos los archivos
    fprintf(fp0, "%12.8f %12.8f %5i %12.8f %12.8f %5i\n", a, b, npx, c, d, npy);
    fprintf(fp1, "%12.8f %12.8f %5i %12.8f %12.8f %5i\n", a, b, npx, c, d, npy);
    fprintf(fp2, "%12.8f %12.8f %5i %12.8f %12.8f %5i\n", a, b, npx, c, d, npy);

    // Guardado inicial
    guardar_estado(fp0, tiempo, npx, npy, h0);
    guardar_estado(fp1, tiempo, npx, npy, q1_0);
    guardar_estado(fp2, tiempo, npx, npy, q2_0);

    // Variables de bucle e indices
    int idx, idx_xp, idx_xm, idx_yp, idx_ym;

    // Coeficientes numericos
    double kx, ky, k_visc;

    while (tiempo < T - 1e-9)
    {
        if (tiempo + dt > T)
            // Tomamos el min para tener asegurada estabilidad
            dt = std::min((T - tiempo), obtener_dt(npx, npy, dx, dy, gamma, h0, q1_0, q2_0));
        else
            dt = obtener_dt(npx, npy, dx, dy, gamma, h0, q1_0, q2_0);

        kx = dt / (2.0 * dx);
        ky = dt / (2.0 * dy);
        k_visc = gamma / 4.0; // Coeficiente del termino de estabilizacion

        printf("Tiempo: %12.8f | dt: %12.8f\n", tiempo + dt, dt);

        // =========================================================
        // BUCLE PRINCIPAL (INTERIOR + BORDES + ESQUINAS)
        // =========================================================

        // Iteramos todo el dominio incluyendo bordes, pero ajustando los indices
        // 1. INTERIOR
        // ! Variables de arrays dentro de bucle mejor para mejorar rendimiento (ademas la pila de las hebras pueden llenarse al usarse como privadas,
        // !    aunque evitas tener que ser las variables creadas en cada iteraccion)

        // Es el bucle con mayor carga computacional O(N^2) luego es el principal donde si merece la pena paralelizar
#pragma omp parallel for private(idx, idx_xp, idx_xm, idx_yp, idx_ym)
        for (int i = 1; i < npy; i++)
        {
            // Variables temporales para flujos (F = flux X, G = flux Y)
            double F_xp[3], F_xn[3];
            double G_yp[3], G_yn[3];

            // Variables para disipacion
            double lap_x[3], lap_y[3];

            for (int j = 1; j < npx; j++)
            {
                idx = i * (npx + 1) + j;

                // Vecinos directos
                idx_xp = idx + 1;                 // x positivo (derecha) (j+1)
                idx_xm = idx - 1;                 // x negativo (izquierda)
                idx_yp = (i + 1) * (npx + 1) + j; // y positivo (arriba)
                idx_ym = (i - 1) * (npx + 1) + j; // y negativo (abajo)

                // Calculo de flujos X en vecinos derecho e izquierdo
                // La funcion nos actualiza los valores en F
                calc_flujo_x(h0[idx_xp], q1_0[idx_xp], q2_0[idx_xp], F_xp[0], F_xp[1], F_xp[2]);
                calc_flujo_x(h0[idx_xm], q1_0[idx_xm], q2_0[idx_xm], F_xn[0], F_xn[1], F_xn[2]);

                // Calculo de flujos Y en vecinos arriba y abajo
                calc_flujo_y(h0[idx_yp], q1_0[idx_yp], q2_0[idx_yp], G_yp[0], G_yp[1], G_yp[2]);
                calc_flujo_y(h0[idx_ym], q1_0[idx_ym], q2_0[idx_ym], G_yn[0], G_yn[1], G_yn[2]);

                // Diferencias centradas de segundo orden
                // Usamos la formula: Val_pos - 2*Val_cen + Val_neg
                lap_x[0] = h0[idx_xp] - 2.0 * h0[idx] + h0[idx_xm];
                lap_x[1] = q1_0[idx_xp] - 2.0 * q1_0[idx] + q1_0[idx_xm];
                lap_x[2] = q2_0[idx_xp] - 2.0 * q2_0[idx] + q2_0[idx_xm];

                lap_y[0] = h0[idx_yp] - 2.0 * h0[idx] + h0[idx_ym];
                lap_y[1] = q1_0[idx_yp] - 2.0 * q1_0[idx] + q1_0[idx_ym];
                lap_y[2] = q2_0[idx_yp] - 2.0 * q2_0[idx] + q2_0[idx_ym];

                // Actualizacion del estado
                h1[idx] = h0[idx] - kx * (F_xp[0] - F_xn[0]) - ky * (G_yp[0] - G_yn[0]) + k_visc * (lap_x[0] + lap_y[0]);
                q1_1[idx] = q1_0[idx] - kx * (F_xp[1] - F_xn[1]) - ky * (G_yp[1] - G_yn[1]) + k_visc * (lap_x[1] + lap_y[1]);
                q2_1[idx] = q2_0[idx] - kx * (F_xp[2] - F_xn[2]) - ky * (G_yp[2] - G_yn[2]) + k_visc * (lap_x[2] + lap_y[2]);
            }
        }

// 2. BORDES PERIODICOS Y (Arriba/Abajo)
// Solo estamos calculando una fila (npx) si el numero de puntos es pequeno no es rentable debido al overhead introducido al
//  gestionar los threads
#pragma omp parallel for private(idx, idx_xp, idx_xm, idx_yp, idx_ym) if (npx > 5000)
        for (int j = 1; j < npx; j++)
        {
            // Variables temporales para flujos (F = flux X, G = flux Y)
            double F_xp[3], F_xn[3];
            double G_yp[3], G_yn[3];

            // Variables para disipacion
            double lap_x[3], lap_y[3];

            // Fila Inferior (i=0)
            idx = j;
            idx_xp = idx + 1;
            idx_xm = idx - 1;
            idx_yp = (1) * (npx + 1) + j; // i+1
            idx_ym = npy * (npx + 1) + j; // i-1 (periodico -> ultima fila)

            calc_flujo_x(h0[idx_xp], q1_0[idx_xp], q2_0[idx_xp], F_xp[0], F_xp[1], F_xp[2]);
            calc_flujo_x(h0[idx_xm], q1_0[idx_xm], q2_0[idx_xm], F_xn[0], F_xn[1], F_xn[2]);
            calc_flujo_y(h0[idx_yp], q1_0[idx_yp], q2_0[idx_yp], G_yp[0], G_yp[1], G_yp[2]);
            calc_flujo_y(h0[idx_ym], q1_0[idx_ym], q2_0[idx_ym], G_yn[0], G_yn[1], G_yn[2]);

            lap_x[0] = h0[idx_xp] - 2.0 * h0[idx] + h0[idx_xm];
            lap_x[1] = q1_0[idx_xp] - 2.0 * q1_0[idx] + q1_0[idx_xm];
            lap_x[2] = q2_0[idx_xp] - 2.0 * q2_0[idx] + q2_0[idx_xm];

            lap_y[0] = h0[idx_yp] - 2.0 * h0[idx] + h0[idx_ym];
            lap_y[1] = q1_0[idx_yp] - 2.0 * q1_0[idx] + q1_0[idx_ym];
            lap_y[2] = q2_0[idx_yp] - 2.0 * q2_0[idx] + q2_0[idx_ym];

            h1[idx] = h0[idx] - kx * (F_xp[0] - F_xn[0]) - ky * (G_yp[0] - G_yn[0]) + k_visc * (lap_x[0] + lap_y[0]);
            q1_1[idx] = q1_0[idx] - kx * (F_xp[1] - F_xn[1]) - ky * (G_yp[1] - G_yn[1]) + k_visc * (lap_x[1] + lap_y[1]);
            q2_1[idx] = q2_0[idx] - kx * (F_xp[2] - F_xn[2]) - ky * (G_yp[2] - G_yn[2]) + k_visc * (lap_x[2] + lap_y[2]);

            // Fila Superior (i=npy)
            idx = npy * (npx + 1) + j;
            idx_xp = idx + 1;
            idx_xm = idx - 1;
            idx_yp = j;                         // i+1 (periodico -> primera fila)
            idx_ym = (npy - 1) * (npx + 1) + j; // i-1

            calc_flujo_x(h0[idx_xp], q1_0[idx_xp], q2_0[idx_xp], F_xp[0], F_xp[1], F_xp[2]);
            calc_flujo_x(h0[idx_xm], q1_0[idx_xm], q2_0[idx_xm], F_xn[0], F_xn[1], F_xn[2]);
            calc_flujo_y(h0[idx_yp], q1_0[idx_yp], q2_0[idx_yp], G_yp[0], G_yp[1], G_yp[2]);
            calc_flujo_y(h0[idx_ym], q1_0[idx_ym], q2_0[idx_ym], G_yn[0], G_yn[1], G_yn[2]);

            lap_x[0] = h0[idx_xp] - 2.0 * h0[idx] + h0[idx_xm];
            lap_x[1] = q1_0[idx_xp] - 2.0 * q1_0[idx] + q1_0[idx_xm];
            lap_x[2] = q2_0[idx_xp] - 2.0 * q2_0[idx] + q2_0[idx_xm];
            lap_y[0] = h0[idx_yp] - 2.0 * h0[idx] + h0[idx_ym];
            lap_y[1] = q1_0[idx_yp] - 2.0 * q1_0[idx] + q1_0[idx_ym];
            lap_y[2] = q2_0[idx_yp] - 2.0 * q2_0[idx] + q2_0[idx_ym];

            h1[idx] = h0[idx] - kx * (F_xp[0] - F_xn[0]) - ky * (G_yp[0] - G_yn[0]) + k_visc * (lap_x[0] + lap_y[0]);
            q1_1[idx] = q1_0[idx] - kx * (F_xp[1] - F_xn[1]) - ky * (G_yp[1] - G_yn[1]) + k_visc * (lap_x[1] + lap_y[1]);
            q2_1[idx] = q2_0[idx] - kx * (F_xp[2] - F_xn[2]) - ky * (G_yp[2] - G_yn[2]) + k_visc * (lap_x[2] + lap_y[2]);
        }

// 3. BORDES PERIODICOS X (Borde en la Izquierda/Derecha)
// Cond contorno columna 0 y npx
// Razonamiento analago al borde anterior
#pragma omp parallel for private(idx, idx_xp, idx_xm, idx_yp, idx_ym) if (npy > 5000)
        for (int i = 1; i < npy; i++)
        {
            // Variables temporales para flujos (F = flux X, G = flux Y)
            double F_xp[3], F_xn[3];
            double G_yp[3], G_yn[3];

            // Variables para disipacion
            double lap_x[3], lap_y[3];

            // Columna Izquierda (j=0)
            idx = i * (npx + 1);
            idx_xp = idx + 1;
            idx_xm = i * (npx + 1) + npx; // j-1 (periodico -> ultima columna)
            idx_yp = (i + 1) * (npx + 1);
            idx_ym = (i - 1) * (npx + 1);

            calc_flujo_x(h0[idx_xp], q1_0[idx_xp], q2_0[idx_xp], F_xp[0], F_xp[1], F_xp[2]);
            calc_flujo_x(h0[idx_xm], q1_0[idx_xm], q2_0[idx_xm], F_xn[0], F_xn[1], F_xn[2]);

            calc_flujo_y(h0[idx_yp], q1_0[idx_yp], q2_0[idx_yp], G_yp[0], G_yp[1], G_yp[2]);
            calc_flujo_y(h0[idx_ym], q1_0[idx_ym], q2_0[idx_ym], G_yn[0], G_yn[1], G_yn[2]);

            lap_x[0] = h0[idx_xp] - 2.0 * h0[idx] + h0[idx_xm];
            lap_x[1] = q1_0[idx_xp] - 2.0 * q1_0[idx] + q1_0[idx_xm];
            lap_x[2] = q2_0[idx_xp] - 2.0 * q2_0[idx] + q2_0[idx_xm];

            lap_y[0] = h0[idx_yp] - 2.0 * h0[idx] + h0[idx_ym];
            lap_y[1] = q1_0[idx_yp] - 2.0 * q1_0[idx] + q1_0[idx_ym];
            lap_y[2] = q2_0[idx_yp] - 2.0 * q2_0[idx] + q2_0[idx_ym];

            h1[idx] = h0[idx] - kx * (F_xp[0] - F_xn[0]) - ky * (G_yp[0] - G_yn[0]) + k_visc * (lap_x[0] + lap_y[0]);
            q1_1[idx] = q1_0[idx] - kx * (F_xp[1] - F_xn[1]) - ky * (G_yp[1] - G_yn[1]) + k_visc * (lap_x[1] + lap_y[1]);
            q2_1[idx] = q2_0[idx] - kx * (F_xp[2] - F_xn[2]) - ky * (G_yp[2] - G_yn[2]) + k_visc * (lap_x[2] + lap_y[2]);

            // Columna Derecha (j=npx)
            idx = i * (npx + 1) + npx;
            idx_xp = i * (npx + 1); // j+1 (periodico -> columna 0)
            idx_xm = idx - 1;
            idx_yp = (i + 1) * (npx + 1) + npx;
            idx_ym = (i - 1) * (npx + 1) + npx;

            calc_flujo_x(h0[idx_xp], q1_0[idx_xp], q2_0[idx_xp], F_xp[0], F_xp[1], F_xp[2]);
            calc_flujo_x(h0[idx_xm], q1_0[idx_xm], q2_0[idx_xm], F_xn[0], F_xn[1], F_xn[2]);

            calc_flujo_y(h0[idx_yp], q1_0[idx_yp], q2_0[idx_yp], G_yp[0], G_yp[1], G_yp[2]);
            calc_flujo_y(h0[idx_ym], q1_0[idx_ym], q2_0[idx_ym], G_yn[0], G_yn[1], G_yn[2]);

            lap_x[0] = h0[idx_xp] - 2.0 * h0[idx] + h0[idx_xm];
            lap_x[1] = q1_0[idx_xp] - 2.0 * q1_0[idx] + q1_0[idx_xm];
            lap_x[2] = q2_0[idx_xp] - 2.0 * q2_0[idx] + q2_0[idx_xm];

            lap_y[0] = h0[idx_yp] - 2.0 * h0[idx] + h0[idx_ym];
            lap_y[1] = q1_0[idx_yp] - 2.0 * q1_0[idx] + q1_0[idx_ym];
            lap_y[2] = q2_0[idx_yp] - 2.0 * q2_0[idx] + q2_0[idx_ym];

            h1[idx] = h0[idx] - kx * (F_xp[0] - F_xn[0]) - ky * (G_yp[0] - G_yn[0]) + k_visc * (lap_x[0] + lap_y[0]);
            q1_1[idx] = q1_0[idx] - kx * (F_xp[1] - F_xn[1]) - ky * (G_yp[1] - G_yn[1]) + k_visc * (lap_x[1] + lap_y[1]);
            q2_1[idx] = q2_0[idx] - kx * (F_xp[2] - F_xn[2]) - ky * (G_yp[2] - G_yn[2]) + k_visc * (lap_x[2] + lap_y[2]);
        }

        // 4. ESQUINAS (4 PUNTOS CON DOBLE PERIODICIDAD)
        {
            // Variables temporales para flujos (F = flux X, G = flux Y)
            double F_xp[3], F_xn[3];
            double G_yp[3], G_yn[3];

            // Variables para disipacion
            double lap_x[3], lap_y[3];

            // Esquina (0,0)
            idx = 0;
            idx_xp = 1;
            idx_xm = npx; // j-1 -> npx
            idx_yp = npx + 1;
            idx_ym = npy * (npx + 1); // i-1 -> npy

            calc_flujo_x(h0[idx_xp], q1_0[idx_xp], q2_0[idx_xp], F_xp[0], F_xp[1], F_xp[2]);
            calc_flujo_x(h0[idx_xm], q1_0[idx_xm], q2_0[idx_xm], F_xn[0], F_xn[1], F_xn[2]);

            calc_flujo_y(h0[idx_yp], q1_0[idx_yp], q2_0[idx_yp], G_yp[0], G_yp[1], G_yp[2]);
            calc_flujo_y(h0[idx_ym], q1_0[idx_ym], q2_0[idx_ym], G_yn[0], G_yn[1], G_yn[2]);

            lap_x[0] = h0[idx_xp] - 2.0 * h0[idx] + h0[idx_xm];
            lap_x[1] = q1_0[idx_xp] - 2.0 * q1_0[idx] + q1_0[idx_xm];
            lap_x[2] = q2_0[idx_xp] - 2.0 * q2_0[idx] + q2_0[idx_xm];

            lap_y[0] = h0[idx_yp] - 2.0 * h0[idx] + h0[idx_ym];
            lap_y[1] = q1_0[idx_yp] - 2.0 * q1_0[idx] + q1_0[idx_ym];
            lap_y[2] = q2_0[idx_yp] - 2.0 * q2_0[idx] + q2_0[idx_ym];

            h1[idx] = h0[idx] - kx * (F_xp[0] - F_xn[0]) - ky * (G_yp[0] - G_yn[0]) + k_visc * (lap_x[0] + lap_y[0]);
            q1_1[idx] = q1_0[idx] - kx * (F_xp[1] - F_xn[1]) - ky * (G_yp[1] - G_yn[1]) + k_visc * (lap_x[1] + lap_y[1]);
            q2_1[idx] = q2_0[idx] - kx * (F_xp[2] - F_xn[2]) - ky * (G_yp[2] - G_yn[2]) + k_visc * (lap_x[2] + lap_y[2]);

            // Esquina (0, npx)
            idx = npx;
            idx_xp = 0; // j+1 -> 0
            idx_xm = idx - 1;
            idx_yp = idx + (npx + 1);
            idx_ym = npy * (npx + 1) + npx; // i-1 -> npy

            calc_flujo_x(h0[idx_xp], q1_0[idx_xp], q2_0[idx_xp], F_xp[0], F_xp[1], F_xp[2]);
            calc_flujo_x(h0[idx_xm], q1_0[idx_xm], q2_0[idx_xm], F_xn[0], F_xn[1], F_xn[2]);

            calc_flujo_y(h0[idx_yp], q1_0[idx_yp], q2_0[idx_yp], G_yp[0], G_yp[1], G_yp[2]);
            calc_flujo_y(h0[idx_ym], q1_0[idx_ym], q2_0[idx_ym], G_yn[0], G_yn[1], G_yn[2]);

            lap_x[0] = h0[idx_xp] - 2.0 * h0[idx] + h0[idx_xm];
            lap_x[1] = q1_0[idx_xp] - 2.0 * q1_0[idx] + q1_0[idx_xm];
            lap_x[2] = q2_0[idx_xp] - 2.0 * q2_0[idx] + q2_0[idx_xm];

            lap_y[0] = h0[idx_yp] - 2.0 * h0[idx] + h0[idx_ym];
            lap_y[1] = q1_0[idx_yp] - 2.0 * q1_0[idx] + q1_0[idx_ym];
            lap_y[2] = q2_0[idx_yp] - 2.0 * q2_0[idx] + q2_0[idx_ym];

            h1[idx] = h0[idx] - kx * (F_xp[0] - F_xn[0]) - ky * (G_yp[0] - G_yn[0]) + k_visc * (lap_x[0] + lap_y[0]);
            q1_1[idx] = q1_0[idx] - kx * (F_xp[1] - F_xn[1]) - ky * (G_yp[1] - G_yn[1]) + k_visc * (lap_x[1] + lap_y[1]);
            q2_1[idx] = q2_0[idx] - kx * (F_xp[2] - F_xn[2]) - ky * (G_yp[2] - G_yn[2]) + k_visc * (lap_x[2] + lap_y[2]);

            // Esquina (npy, 0)
            idx = npy * (npx + 1);
            idx_xp = idx + 1;
            idx_xm = idx + npx; // j-1 -> npx
            idx_yp = 0;         // i+1 -> 0
            idx_ym = idx - (npx + 1);

            calc_flujo_x(h0[idx_xp], q1_0[idx_xp], q2_0[idx_xp], F_xp[0], F_xp[1], F_xp[2]);
            calc_flujo_x(h0[idx_xm], q1_0[idx_xm], q2_0[idx_xm], F_xn[0], F_xn[1], F_xn[2]);

            calc_flujo_y(h0[idx_yp], q1_0[idx_yp], q2_0[idx_yp], G_yp[0], G_yp[1], G_yp[2]);
            calc_flujo_y(h0[idx_ym], q1_0[idx_ym], q2_0[idx_ym], G_yn[0], G_yn[1], G_yn[2]);

            lap_x[0] = h0[idx_xp] - 2.0 * h0[idx] + h0[idx_xm];
            lap_x[1] = q1_0[idx_xp] - 2.0 * q1_0[idx] + q1_0[idx_xm];
            lap_x[2] = q2_0[idx_xp] - 2.0 * q2_0[idx] + q2_0[idx_xm];

            lap_y[0] = h0[idx_yp] - 2.0 * h0[idx] + h0[idx_ym];
            lap_y[1] = q1_0[idx_yp] - 2.0 * q1_0[idx] + q1_0[idx_ym];
            lap_y[2] = q2_0[idx_yp] - 2.0 * q2_0[idx] + q2_0[idx_ym];

            h1[idx] = h0[idx] - kx * (F_xp[0] - F_xn[0]) - ky * (G_yp[0] - G_yn[0]) + k_visc * (lap_x[0] + lap_y[0]);
            q1_1[idx] = q1_0[idx] - kx * (F_xp[1] - F_xn[1]) - ky * (G_yp[1] - G_yn[1]) + k_visc * (lap_x[1] + lap_y[1]);
            q2_1[idx] = q2_0[idx] - kx * (F_xp[2] - F_xn[2]) - ky * (G_yp[2] - G_yn[2]) + k_visc * (lap_x[2] + lap_y[2]);

            // Esquina (npy, npx)
            idx = npy * (npx + 1) + npx;
            idx_xp = npy * (npx + 1); // j+1 -> 0
            idx_xm = idx - 1;
            idx_yp = npx; // i+1 -> 0
            idx_ym = idx - (npx + 1);

            calc_flujo_x(h0[idx_xp], q1_0[idx_xp], q2_0[idx_xp], F_xp[0], F_xp[1], F_xp[2]);
            calc_flujo_x(h0[idx_xm], q1_0[idx_xm], q2_0[idx_xm], F_xn[0], F_xn[1], F_xn[2]);

            calc_flujo_y(h0[idx_yp], q1_0[idx_yp], q2_0[idx_yp], G_yp[0], G_yp[1], G_yp[2]);
            calc_flujo_y(h0[idx_ym], q1_0[idx_ym], q2_0[idx_ym], G_yn[0], G_yn[1], G_yn[2]);

            lap_x[0] = h0[idx_xp] - 2.0 * h0[idx] + h0[idx_xm];
            lap_x[1] = q1_0[idx_xp] - 2.0 * q1_0[idx] + q1_0[idx_xm];
            lap_x[2] = q2_0[idx_xp] - 2.0 * q2_0[idx] + q2_0[idx_xm];

            lap_y[0] = h0[idx_yp] - 2.0 * h0[idx] + h0[idx_ym];
            lap_y[1] = q1_0[idx_yp] - 2.0 * q1_0[idx] + q1_0[idx_ym];
            lap_y[2] = q2_0[idx_yp] - 2.0 * q2_0[idx] + q2_0[idx_ym];

            h1[idx] = h0[idx] - kx * (F_xp[0] - F_xn[0]) - ky * (G_yp[0] - G_yn[0]) + k_visc * (lap_x[0] + lap_y[0]);
            q1_1[idx] = q1_0[idx] - kx * (F_xp[1] - F_xn[1]) - ky * (G_yp[1] - G_yn[1]) + k_visc * (lap_x[1] + lap_y[1]);
            q2_1[idx] = q2_0[idx] - kx * (F_xp[2] - F_xn[2]) - ky * (G_yp[2] - G_yn[2]) + k_visc * (lap_x[2] + lap_y[2]);
        }

        // Intercambio de punteros (Swap)
        tmp = h0;
        h0 = h1;
        h1 = tmp;
        tmp = q1_0;
        q1_0 = q1_1;
        q1_1 = tmp;
        tmp = q2_0;
        q2_0 = q2_1;
        q2_1 = tmp;

        tiempo += dt;

        // Guardado periodico en los 3 archivos
        if (tiempo >= tg || fabs(tiempo - T) < 1e-9)
        {
            guardar_estado(fp0, tiempo, npx, npy, h0);
            guardar_estado(fp1, tiempo, npx, npy, q1_0);
            guardar_estado(fp2, tiempo, npx, npy, q2_0);
            tg += dtg;
        }
    }

    t_fin = omp_get_wtime();
    secs = (double)(t_fin - t_ini);
    printf("Tiempo de ejecucion: %.16g segundos\n", secs);

    // Cerrar archivos y liberar memoria
    fclose(fp0);
    fclose(fp1);
    fclose(fp2);

    delete[] h0;
    delete[] q1_0;
    delete[] q2_0;

    delete[] h1;
    delete[] q1_1;
    delete[] q2_1;

    delete[] x;
    delete[] y;

    return 0;
}
