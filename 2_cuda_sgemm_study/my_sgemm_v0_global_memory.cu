#include <cstdio>
#define A(i, j) a[(i) * n + (j)]

void random_matrix(int m, int n, float *a)
{
    for (int i = 0; i < m; i++)
        for (int j = 0; j < n; j++)
#if 1
            A(i, j) = 2.0 * (float)drand48() - 1.0;
#else
            A(i, j) = (j - i) % 3;
#endif
}
int main()
{
    int m = 2048;
    int n = 2048;
    int k = 2048;

    const size_t mem_size_A = m * k * sizeof(float);
    const size_t mem_size_B = n * k * sizeof(float);
    const size_t mem_size_C = m * n * sizeof(float);

    float *matrix_A_host = (float *)malloc(mem_size_A);
    float *matrix_B_host = (float *)malloc(mem_size_B);

    float *matrix_C_host_gpu_calc = (float *)malloc(mem_size_C);
    float *matrix_C_host_cpu_calc = (float *)malloc(mem_size_C);

    random_matrix(m, k, matrix_A_host);
    random_matrix(n, k, matrix_B_host);
    printf("sgemm\n");

    return 0;
}