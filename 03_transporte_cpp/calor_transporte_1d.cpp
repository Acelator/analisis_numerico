// Transporte hiperbolico 1D - upwind y aguas someras
// Ecuaciones: transporte lineal u_t + c u_x = 0, Burgers u_t + u u_x = alfa u_xx,
// y Saint-Venant 1D (h_t + q_x = 0, ...). Esquemas paralelos con OpenMP.
// Ver docs/enunciados_resumidos.md#03_transporte_cpp
// Compilacion: g++ -O2 -fopenmp calor_transporte_1d.cpp -o calor_transporte_1d

#include <math.h>
#include <cmath>
#include <omp.h>
#include <cassert>
#include <stdio.h>
#include <iostream>
#include <fstream>
#include <cstdlib>
#include <time.h>

#define _USE_MATH_DEFINES

int transporte(int argc, char **argv);
int caso_2(int argc, char **argv);
int caso_3(int argc, char **argv);
int caso_4(int argc, char **argv);
int caso_4_alt(int argc, char **argv);

using namespace std;

int main(int argc, char **argv)
{
    int caso = atoi(argv[1]);
    bool retFlag;
    if (caso == 1)
        transporte(argc, argv);
    else if (caso == 2)
        caso_2(argc, argv);
    else if (caso == 3)
        caso_3(argc, argv);
    else if (caso == 41)
        caso_4(argc, argv);
    else if (caso == 42)
        caso_4_alt(argc, argv);
    else
    {
        printf("Dicho numero no pertence a ningun caso \n");
        printf("Numeros validos son 1-4");
        return 1;
    }

    return 0;
}

// /////////////////////////////////////////////
// CASO 1

double cond_inicial_ejer1(double x0)
{
    double ci = 0.0;
    return ci;
}

double cond_contorno_ej1(double x0, double t)
{
    double val = cos(M_PI * 2 * t) > 0 ? 1.0 : 0.0;
    return val;
}

int transporte(int argc, char **argv)
{
    std::cout << "Caso 1 \n ";

    double a, b, T, c, phi, tiempo, tg, dtg;
    int npx, i, j;
    char *fichero_salida;
    double dx, dt;
    FILE *fp;
    clock_t t_ini, t_fin;
    double secs;
    t_ini = omp_get_wtime();

    if (argc != 11)
    {
        printf("Uso (Ecuacion del transporte):\n");
        printf("%s", argv[0]);
        printf(" a b T npx phi c fichero_salida dt_guardado nh \n");
        printf("a: Comienzo del intervalo.\n");
        printf("b: Final del inervalo.\n");
        printf("T: Tiempo total de integracion.\n");
        printf("npx: N. de particiones del intervalo [a,b]\n");
        printf("phi: Coef. estabilidad. (Entre 0 y 1)\n");
        printf("c: Coef (c > 0).\n");
        printf("Nombre fichero de salida.\n");
        printf("Dt guardado.\n");
        printf("numeros hilos.\n");
        return -1;
    }

    a = atof(argv[2]);
    b = atof(argv[3]);
    T = atof(argv[4]);
    npx = atoi(argv[5]);
    phi = atof(argv[6]);
    c = atof(argv[7]);
    fichero_salida = argv[8];
    dtg = atof(argv[9]);

    int nh = atoi(argv[10]);
    omp_set_num_threads(nh);

    assert(c > 0);
    assert(phi > 0 && phi <= 1);

    double *sol0{nullptr}; // solucion en el instante n
    double *sol1{nullptr}; // solucion en el instante n+1
    double *x{nullptr};    // particion del intervalo
    double *aux{nullptr};

    // Definicion de los tableros
    sol0 = new double[npx + 1];
    sol1 = new double[npx + 1];
    x = new double[npx + 1];
    dx = (b - a) / double(npx);

    // Condiccion de estabilidad (metodo explicito)
    dt = phi * dx / fabs(c);
    printf("nu: %12.8f | ", c);
    printf("dt: %12.8f\n", dt);

#pragma omp parallel for
    for (i = 0; i <= npx; i++)
    {
        x[i] = a + dx * double(i);
        sol0[i] = cond_inicial_ejer1(x[i]);
    }

    tiempo = 0.0;
    tg = dtg;
    fp = fopen(fichero_salida, "wt");

    for (i = 0; i <= npx; i++)
    {
        fprintf(fp, "%12.8f", x[i]);
    }
    fprintf(fp, "\n");

    fprintf(fp, "%12.8f", tiempo);
    for (i = 0; i <= npx; i++)
    {
        fprintf(fp, "%12.8f", sol0[i]);
    }
    fprintf(fp, "\n");

    // Variable para recordar cuando fue la ultima vez que escribimos en el fichero
    double ultimo_guardado = tiempo;
    double alfa = dt / (2 * dx);

    while (tiempo < T)
    {
        // Ajuste del ultimo paso de tiempo si es necesario para caer exactamente en T
        if (tiempo + dt > T)
            dt = T - tiempo;
        // printf("Tiempo: %12.8f\n", tiempo + dt);

#pragma omp parallel for if (npx > 5000)
        for (i = 1; i < npx; i++)
        {
            sol1[i] = sol0[i] - alfa * (c * (sol0[i + 1] - sol0[i - 1]) - fabs(c) * (sol0[i + 1] - 2 * sol0[i] + sol0[i - 1]));
        }

        // condiciones de contorno
        sol1[0] = sol0[0] - alfa * (c * (sol0[1] - cond_contorno_ej1(a, tiempo)) - fabs(c) * (sol0[1] - 2 * sol0[0] + cond_contorno_ej1(a, tiempo)));

        // Como u_N+1 es un valor cualquiera, tomo cero
        sol1[npx] = sol0[npx] - alfa * (c * (-sol0[npx - 1]) - fabs(c) * (-2 * sol0[npx] + sol0[npx - 1]));

        aux = sol0;
        sol0 = sol1;
        sol1 = aux;
        tiempo += dt;

        if (tiempo >= tg || fabs(tiempo - T) < 1e-9)
        {
            fprintf(fp, "%12.8f", tiempo);
            for (i = 0; i <= npx; i++)
            {
                fprintf(fp, "%12.8f", sol0[i]);
            }
            fprintf(fp, "\n");
            tg += dtg;
        }
    }

    t_fin = omp_get_wtime();
    secs = (double)(t_fin - t_ini);
    printf("%.16g milisegundos\n", secs * 1000.0);
    fclose(fp);

    delete[] sol0;
    delete[] sol1;
    delete[] x;

    return {};
}

// /////////////////////////////////////////////
// CASO 2

double cond_inicial_ejer2(double x0)
{
    return std::exp(-(x0 * x0));
}

double cond_contorno_ej2(double x0, double t)
{
    if (cos(M_PI * 2 * t) > 0)
    {
        return 1.0;
    }
    else
    {
        return 0.0;
    }
}

int caso_2(int argc, char **argv)
{
    std::cout << "Caso 2 \n ";

    double a, b, T, c, phi, tiempo, tg, dtg;
    int npx, i, j;
    char *fichero_salida;
    double dx, dt;
    FILE *fp;
    clock_t t_ini, t_fin;
    double secs;
    t_ini = omp_get_wtime();

    if (argc != 10)
    {
        printf("Uso (Ecuacion del transporte):\n");
        printf("%s", argv[0]);
        printf(" a b T npx phi c fichero_salida dt_guardado \n");
        printf("a: Comienzo del intervalo.\n");
        printf("b: Final del inervalo.\n");
        printf("T: Tiempo total de integracion.\n");
        printf("npx: N. de particiones del intervalo [a,b]\n");
        printf("phi: Coef. estabilidad. (Entre 0 y 1)\n");
        printf("c: Coef (c > 0).\n");
        printf("Nombre fichero de salida.\n");
        printf("Dt guardado.\n");
        return -1;
    }

    a = atof(argv[2]);
    b = atof(argv[3]);
    T = atof(argv[4]);
    npx = atoi(argv[5]);
    phi = atof(argv[6]);
    c = atof(argv[7]);
    fichero_salida = argv[8];
    dtg = atof(argv[9]);

    int nh = atoi(argv[10]);
    omp_set_num_threads(nh);

    assert(c > 0);

    double *sol0{nullptr}, *sol1{nullptr}, *aux{nullptr};
    double *x{nullptr};

    // definicion de los tableros
    sol0 = new double[npx + 1];
    sol1 = new double[npx + 1];
    x = new double[npx + 1];
    dx = (b - a) / double(npx);

    dt = phi * dx / fabs(c) * 0.95;
    printf("nu: %12.8f | ", c);
    printf("dt: %12.8f\n", dt);

#pragma omp parallel for
    for (i = 0; i <= npx; i++)
    {
        x[i] = a + dx * double(i);
        sol0[i] = cond_inicial_ejer2(x[i]);
    }

    // Asegurar periodicidad inicial estricta (u_0 = u_N)
    sol0[npx] = sol0[0];

    tiempo = 0.0;
    tg = dtg;
    fp = fopen(fichero_salida, "wt");

    // Escritura de estado inicial
    for (i = 0; i <= npx; i++)
    {
        fprintf(fp, "%12.8f", x[i]);
    }

    fprintf(fp, "\n");
    fprintf(fp, "%12.8f", tiempo);

    for (i = 0; i <= npx; i++)
    {
        fprintf(fp, "%12.8f", sol0[i]);
    }
    fprintf(fp, "\n");

    while (tiempo < T - 1e-9)
    {
        if (tiempo + dt > T)
            dt = T - tiempo;

        printf("Tiempo: %12.8f\n", tiempo + dt);

// 1. Nodos INTERIORES (1 a N-1) -> Esquema de diferencias finitas estandar
#pragma omp parallel for
        for (i = 1; i < npx; i++)
            sol1[i] = sol0[i] - dt / (2 * dx) * (c * (sol0[i + 1] - sol0[i - 1]) - fabs(c) * (sol0[i + 1] - 2 * sol0[i] + sol0[i - 1]));

        // 2. Condicion de Contorno PERIODICA en i = 0
        // Vecino derecha: sol0[1]
        // Vecino izquierda: sol0[npx-1] (debido a periodicidad u_{-1} = u_{N-1})

        double u_l = sol0[npx - 1]; // u_{i-1} donde i=0 mapea a N-1
        double u_r = sol0[1];       // u_{i+1} donde i=0 es 1
        double u_c = sol0[0];

        sol1[0] = u_c - dt / (2 * dx) * (c * (u_r - u_l) - fabs(c) * (u_r - 2 * u_c + u_l));

        // 3. i = N
        sol1[npx] = sol1[0];

        // Actualizacion de punteros
        aux = sol0;
        sol0 = sol1;
        sol1 = aux;
        tiempo += dt;

        if (tiempo >= tg || fabs(tiempo - T) < 1e-9)
        {
            fprintf(fp, "%12.8f", tiempo);
            for (i = 0; i <= npx; i++)
            {
                fprintf(fp, "%12.8f", sol0[i]);
            }
            fprintf(fp, "\n");
            tg += dtg;
        }
    }

    t_fin = omp_get_wtime();
    secs = (double)(t_fin - t_ini);
    printf("%.16g milisegundos\n", secs * 1000.0);
    fclose(fp);

    delete[] sol0;
    delete[] sol1;
    delete[] x;

    return {};
}

// /////////////////////////////////////////////
// CASO 3: Ecuacion de Burgers
// /////////////////////////////////////////////

double cond_inicial_ej3(double x)
{
    return 1.0 / (1.0 + exp(5.0 * x));
}

// Condiciones de contorno Dirichlet fijas para Burgers segun enunciado
double bc_left_ej3(double t) { return 1.0; }
double bc_right_ej3(double t) { return 0.0; }

int caso_3(int argc, char **argv)
{
    std::cout << "Caso 3: Ecuacion de Burgers\n";

    if (argc != 10)
    {
        printf("Uso: %s 3 a b T npx gamma fichero_salida dt_guardado\n", argv[0]);
        return -1;
    }

    double a = atof(argv[2]);
    double b = atof(argv[3]);
    double T = atof(argv[4]);
    int npx = atoi(argv[5]);
    double gamma = atof(argv[6]);

    char *fichero_salida = argv[7];
    double dtg = atof(argv[8]);

    int nh = atoi(argv[9]);
    omp_set_num_threads(nh);

    double *u0 = new double[npx + 1];
    double *u1 = new double[npx + 1];
    double *x = new double[npx + 1];
    double *aux{nullptr};

    double dx = (b - a) / double(npx);

    // Inicializacion
    double max_u = 0.0;
#pragma omp parallel for reduction(max : max_u)
    for (int i = 0; i <= npx; i++)
    {
        x[i] = a + dx * double(i);
        u0[i] = cond_inicial_ej3(x[i]);

        double v = fabs(u0[i]);
        if (v > max_u)
            max_u = v;
    }

    assert(max_u > 0);

    // Condicion de estabilidad
    double dt = gamma * dx / max_u;
    double alpha = gamma * dx / (2.0 * dt);

    printf("dx: %f | dt: %f | alpha: %f\n", dx, dt, alpha);

    double tiempo = 0.0;
    double tg = dtg;
    FILE *fp = fopen(fichero_salida, "wt");

    // Guardado inicial
    for (int i = 0; i <= npx; i++)
        fprintf(fp, "%12.8f ", x[i]);
    fprintf(fp, "\n%12.8f ", tiempo);

    for (int i = 0; i <= npx; i++)
        fprintf(fp, "%12.8f ", u0[i]);
    fprintf(fp, "\n");

    clock_t t_ini = omp_get_wtime();

    while (tiempo < T - 1e-9)
    {
        if (tiempo + dt > T)
            dt = T - tiempo;

        // Calculamos nuevo dt
        double max_u = 0.0;
#pragma omp parallel for reduction(max : max_u)
        for (int i = 0; i <= npx; i++)
        {
            double val = fabs(u0[i]);
            if (val > max_u)
                max_u = val;
        }

        assert(max_u > 0);

        dt = gamma * dx / max_u;
        alpha = gamma * dx / (2 * dt);

        // Esquema explicito para nodos interiores 0 < i < N

        //  Todas las variblaes introducidas dentro del bucle de paralelizacion son privadas por defecto
#pragma omp parallel for
        for (int i = 1; i < npx; i++)
        {
            double primer_termino = ((u0[i + 1] * u0[i + 1]) / 4.0) - ((u0[i - 1] * u0[i - 1]) / 4.0);
            double segundo_termino = alpha * (u0[i + 1] - 2.0 * u0[i] + u0[i - 1]);
            u1[i] = u0[i] - (dt / dx) * (primer_termino - segundo_termino);
        }

        // Fronteras
        // i = 0
        double contorno_izquierda = bc_left_ej3(tiempo);
        double primer_0 = ((u0[1] * u0[1]) / 4.0) - ((contorno_izquierda * contorno_izquierda) / 4.0);
        double segundo_0 = alpha * (u0[1] - 2.0 * u0[0] + contorno_izquierda);
        u1[0] = u0[0] - (dt / dx) * (primer_0 - segundo_0);

        // i = N
        double contorno_derecha = bc_right_ej3(tiempo);
        double primer_N = ((contorno_derecha * contorno_derecha) / 4.0) - ((u0[npx - 1] * u0[npx - 1]) / 4.0);
        double final_N = alpha * (contorno_derecha - 2.0 * u0[npx] + u0[npx - 1]);
        u1[npx] = u0[npx] - (dt / dx) * (primer_N - final_N);

        aux = u0;
        u0 = u1;
        u1 = aux;
        tiempo += dt;

        if (tiempo >= tg - 1e-9 || fabs(tiempo - T) < 1e-9)
        {
            fprintf(fp, "%12.8f ", tiempo);
            for (int i = 0; i <= npx; i++)
                fprintf(fp, "%12.8f ", u0[i]);
            fprintf(fp, "\n");
            tg += dtg;
        }
    }

    clock_t t_fin = omp_get_wtime();
    printf("Tiempo CPU: %.16g ms\n", (double)(t_fin - t_ini) / 1000.0);
    fclose(fp);

    delete[] u0;
    delete[] u1;
    delete[] x;

    return 0;
}

// /////////////////////////////////////////////
// CASO 4: Aguas Someras (Shallow Water)
// /////////////////////////////////////////////

double h_ini_ej4(double x)
{
    return 1.0 + exp(-x * x);
}
double q_ini_ej4(double x)
{
    return 0.0;
}

// Constante gravedad
const double G = 9.81;

// Caso con condicciones de contorno periodicas
int caso_4(int argc, char **argv)
{
    std::cout << "Caso 4: Aguas Someras (Fondo Plano)\n";

    if (argc != 11)
    {
        printf("Uso: %s 4 a b T npx gamma fichero_salida_1 fichero_salida_2 dt_guardado\n", argv[0]);
        return -1;
    }

    double a = atof(argv[2]);
    double b = atof(argv[3]);
    double T = atof(argv[4]);
    int npx = atoi(argv[5]);
    double gamma = atof(argv[6]);

    // Nombres fijos (es mas comodo) (ignoro input)
    const char *fichero_h = "h_data.txt";
    const char *fichero_q = "q_data.txt";
    double dtg = atof(argv[9]);

    int nh = atoi(argv[10]);
    omp_set_num_threads(nh);

    // Arrays para h y q (n y n+1)
    double *h0 = new double[npx + 1];
    double *h1 = new double[npx + 1];
    double *q0 = new double[npx + 1];
    double *q1 = new double[npx + 1];
    double *x = new double[npx + 1];

    double dx = (b - a) / double(npx);

    // Inicializacion y calculo de lambda inicial para dt
    double max_lambda = 0.0;
#pragma omp parallel for reduction(max : max_lambda)
    for (int i = 0; i <= npx; i++)
    {
        x[i] = a + dx * double(i);
        h0[i] = h_ini_ej4(x[i]);
        q0[i] = q_ini_ej4(x[i]);

        double lambda = fabs(q0[i] / h0[i]) + sqrt(G * h0[i]);
        if (lambda > max_lambda)
            max_lambda = lambda;
    }

    // Calculo inicial de dt y alpha
    double dt = gamma * dx / max_lambda;
    double alpha = gamma * dx / (2.0 * dt);

    printf("dx: %f, dt inicial: %f, alpha: %f\n", dx, dt, alpha);

    double tiempo = 0.0;
    double tg = dtg;

    FILE *fp_h = fopen(fichero_h, "wt");
    FILE *fp_q = fopen(fichero_q, "wt");

    // Escritura inicial
    for (int i = 0; i <= npx; i++)
    {
        fprintf(fp_h, "%12.8f ", x[i]);
        fprintf(fp_q, "%12.8f ", x[i]);
    }
    fprintf(fp_h, "\n%12.8f ", tiempo);
    fprintf(fp_q, "\n%12.8f ", tiempo);
    for (int i = 0; i <= npx; i++)
    {
        fprintf(fp_h, "%12.8f ", h0[i]);
        fprintf(fp_q, "%12.8f ", q0[i]);
    }
    fprintf(fp_h, "\n");
    fprintf(fp_q, "\n");

    clock_t t_ini = omp_get_wtime();

    while (tiempo < T - 1e-9)
    {
        max_lambda = 0.0;
#pragma omp parallel for reduction(max : max_lambda)
        for (int i = 0; i <= npx; i++)
        {
            double lambda = fabs(q0[i] / h0[i]) + sqrt(G * h0[i]);
            if (lambda > max_lambda)
                max_lambda = lambda;
        }

        dt = gamma * dx / max_lambda;
        alpha = gamma * dx / (2.0 * dt);

        if (tiempo + dt > T)
            dt = T - tiempo;

// Actualizacion Nodos Interiores (1 a N-1)
#pragma omp parallel for
        for (int i = 1; i < npx; i++)
        {
            // Ecuacion de masa (h)
            h1[i] = h0[i] - (dt / (2.0 * dx)) * (q0[i + 1] - q0[i - 1]) + gamma / 2 * (dt / dx) * (h0[i + 1] - 2.0 * h0[i] + h0[i - 1]);

            // Ecuacion de momento (q)
            double term1 = ((q0[i + 1] * q0[i + 1]) / h0[i + 1] - ((q0[i - 1] * q0[i - 1]) / h0[i - 1]));
            double term2 = (G / 2.0) * ((h0[i + 1] * h0[i + 1]) - (h0[i - 1] * h0[i - 1]));

            q1[i] = q0[i] - (dt / (2.0 * dx)) * (term1 + term2) + alpha * (dt / dx) * (q0[i + 1] - 2.0 * q0[i] + q0[i - 1]);
        }

        // Condiciones de Contorno PERIODICAS
        // h(a,t) = h(b,t) => h_0 = h_N
        // Vecino izq de 0 es N-1. Vecino der de N es 1.

        // Actualizacion i = 0
        int im1 = npx - 1;
        int ip1 = 1;

        h1[0] = h0[0] - (dt / (2.0 * dx)) * (q0[ip1] - q0[im1]) + alpha * (dt / dx) * (h0[ip1] - 2.0 * h0[0] + h0[im1]);

        double t1_0 = (pow(q0[ip1], 2) / h0[ip1]) - (pow(q0[im1], 2) / h0[im1]);
        double t2_0 = (G / 2.0) * (pow(h0[ip1], 2) - pow(h0[im1], 2));
        q1[0] = q0[0] - (dt / (2.0 * dx)) * (t1_0 + t2_0) + alpha * (dt / dx) * (q0[ip1] - 2.0 * q0[0] + q0[im1]);

        // Actualizacion i = N (Es identico a i=0 por periodicidad)
        h1[npx] = h1[0];
        q1[npx] = q1[0];

        // Swap
        double *swp;
        swp = h0;
        h0 = h1;
        h1 = swp;
        swp = q0;
        q0 = q1;
        q1 = swp;
        delete[] swp;

        tiempo += dt;

        if (tiempo >= tg - 1e-9 || fabs(tiempo - T) < 1e-9)
        {
            fprintf(fp_h, "%12.8f ", tiempo);
            fprintf(fp_q, "%12.8f ", tiempo);
            for (int i = 0; i <= npx; i++)
            {
                fprintf(fp_h, "%12.8f ", h0[i]);
                fprintf(fp_q, "%12.8f ", q0[i]);
            }
            fprintf(fp_h, "\n");
            fprintf(fp_q, "\n");
            tg += dtg;
        }
    }

    clock_t t_fin = omp_get_wtime();
    printf("Tiempo CPU: %.16g ms\n", (double)(t_fin - t_ini) / 1000.0);

    fclose(fp_h);
    fclose(fp_q);

    delete[] h0;
    delete[] h1;
    delete[] q0;
    delete[] q1;
    delete[] x;

    return 0;
}

// Condiciones iniciales (misma que Ej 4 original)
double h_ini_alt(double x)
{
    if (x < -2.0)
        return 1.0;
    else
        return 0.25;
}

double q_ini_alt(double x) { return 0.0; }

int caso_4_alt(int argc, char **argv)
{
    std::cout << "Caso 4: Aguas Someras (Fronteras Abiertas/Alternativas)\n";

    assert(argc == 9);

    double a = atof(argv[2]);
    double b = atof(argv[3]);
    double T = atof(argv[4]);
    int npx = atoi(argv[5]);
    double gamma = atof(argv[6]);
    const char *fichero_h = "h_out_alt.txt";
    const char *fichero_q = "q_out_alt.txt";
    double dtg = atof(argv[7]);

    int nh = atoi(argv[8]);
    omp_set_num_threads(nh);

    double *h0 = new double[npx + 1];
    double *h1 = new double[npx + 1];
    double *q0 = new double[npx + 1];
    double *q1 = new double[npx + 1];
    double *x = new double[npx + 1];

    double dx = (b - a) / double(npx);

    // Inicializacion
#pragma omp parallel for
    for (int i = 0; i <= npx; i++)
    {
        // Para guardar en la primera fila los headers del intervalo [a,b]
        x[i] = a + dx * double(i);
        h0[i] = h_ini_alt(x[i]);
        q0[i] = q_ini_alt(x[i]);
    }

    double tiempo = 0.0;
    double tg = dtg;
    double dt;

    FILE *fp_h = fopen(fichero_h, "wt");
    FILE *fp_q = fopen(fichero_q, "wt");

    // Guardado inicial
    for (int i = 0; i <= npx; i++)
    {
        // %12.8f -> Maximo 12 caracteres y no mas de 8 decimales | f indicia espera double de input
        fprintf(fp_h, "%12.8f ", x[i]);
        fprintf(fp_q, "%12.8f ", x[i]);
    }

    fprintf(fp_h, "\n%12.8f ", tiempo);
    fprintf(fp_q, "\n%12.8f ", tiempo);

    for (int i = 0; i <= npx; i++)
    {
        fprintf(fp_h, "%12.8f ", h0[i]);
        fprintf(fp_q, "%12.8f ", q0[i]);
    }
    fprintf(fp_h, "\n");
    fprintf(fp_q, "\n");

    clock_t t_ini = omp_get_wtime();

    while (tiempo < T - 1e-9)
    {
        double max_lambda = 0.0;
#pragma omp parallel for reduction(max : max_lambda) if (npx > 500)
        for (int i = 0; i <= npx; i++)
        {
            // Evitamos h muy pequenos (underflow)
            double h_val = (h0[i] < 1e-8) ? 1e-8 : h0[i];
            double lambda = fabs(q0[i] / h_val) + sqrt(G * h_val);
            if (lambda > max_lambda)
                max_lambda = lambda;
        }

        dt = gamma * dx / max_lambda;

        if (tiempo + dt > T)
            dt = T - tiempo;

        double alpha = gamma * dx / (2.0 * dt);

        double coef_izq = dt / (2.0 * dx);
        double coef_der = alpha * dt / dx;

// A. Nodos INTERIORES (1 a N-1)
#pragma omp parallel for if (npx > 5000)
        for (int i = 1; i < npx; i++)
        {
            h1[i] = h0[i] - coef_izq * (q0[i + 1] - q0[i - 1]) + coef_der * (h0[i + 1] - 2.0 * h0[i] + h0[i - 1]);

            // Ecuacion q
            double parte_pos = ((q0[i + 1] * q0[i + 1]) / h0[i + 1]) + (G / 2.0) * (h0[i + 1] * h0[i + 1]);
            double parte_neg = ((q0[i - 1] * q0[i - 1]) / h0[i - 1]) + (G / 2.0) * (h0[i - 1] * h0[i - 1]);

            q1[i] = q0[i] - coef_izq * (parte_pos - parte_neg) + coef_der * (q0[i + 1] - 2.0 * q0[i] + q0[i - 1]);
        }

        // B. Condiciones de Contorno
        // B.1 Frontera h
        h1[0] = h0[0] - coef_izq * (q0[1] - q0[0]) + coef_der * (h0[1] - h0[0]);
        h1[npx] = h0[npx] - coef_izq * (q0[npx] - q0[npx - 1]) + coef_der * (h0[npx - 1] - h0[npx]);

        // B.2 Frontera q
        double Fq_1 = (q0[1] * q0[1] / h0[1]) + (G / 2.0) * (h0[1] * h0[1]);
        double Fq_0 = (q0[0] * q0[0] / h0[0]) + (G / 2.0) * (h0[0] * h0[0]);

        // Multiplicando por tres el ultimo termindo obtenemos las otras condicciones de contorno
        q1[0] = q0[0] - coef_izq * (Fq_1 - Fq_0) + coef_der * (q0[1] - q0[0]);

        double Fq_N = (q0[npx] * q0[npx] / h0[npx]) + (G / 2.0) * (h0[npx] * h0[npx]);
        double Fq_Nm1 = (q0[npx - 1] * q0[npx - 1] / h0[npx - 1]) + (G / 2.0) * (h0[npx - 1] * h0[npx - 1]);

        // Multiplicando por tres el ultimo termindo obtenemos las otras condicciones de contorno
        q1[npx] = q0[npx] - coef_izq * (Fq_N - Fq_Nm1) + coef_der * (q0[npx - 1] - q0[npx]);

        double *tmp;
        tmp = h0;
        h0 = h1;
        h1 = tmp;
        tmp = q0;
        q0 = q1;
        q1 = tmp;

        tiempo += dt;

        if (tiempo >= tg - 1e-9 || fabs(tiempo - T) < 1e-9)
        {
            fprintf(fp_h, "%12.8f ", tiempo);
            fprintf(fp_q, "%12.8f ", tiempo);
            for (int i = 0; i <= npx; i++)
            {
                fprintf(fp_h, "%12.8f ", h0[i]);
                fprintf(fp_q, "%12.8f ", q0[i]);
            }
            fprintf(fp_h, "\n");
            fprintf(fp_q, "\n");
            tg += dtg;
        }
    }

    clock_t t_fin = omp_get_wtime();
    printf("Tiempo: %.16g ms\n", (double)(t_fin - t_ini) / 1000.0);

    fclose(fp_h);
    fclose(fp_q);

    delete[] h0;
    delete[] h1;
    delete[] q0;
    delete[] q1;
    delete[] x;

    return 0;
}
