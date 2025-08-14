#include <cstdio>

#define THREAD_PER_BLOCK 256

__device__ void warpReduce(volatile float *cache, unsigned int tid)
{
    cache[tid] += cache[tid + 32];
    //__syncthreads();
    cache[tid] += cache[tid + 16];
    //__syncthreads();
    cache[tid] += cache[tid + 8];
    //__syncthreads();
    cache[tid] += cache[tid + 4];
    //__syncthreads();
    cache[tid] += cache[tid + 2];
    //__syncthreads();
    cache[tid] += cache[tid + 1];
    //__syncthreads();
}
__global__ void reduce(float *d_input, float *d_output)
{
    int tid = threadIdx.x;
    volatile __shared__ float shared[THREAD_PER_BLOCK];

    float *input_begin = d_input + blockDim.x * blockIdx.x * 2;
    shared[tid] = input_begin[tid] + input_begin[tid + blockDim.x];
    __syncthreads();
    // #pragma unroll // 展开
    //     for (int i = blockDim.x / 2; i > 32; i /= 2)
    //     {
    //         if (tid < i)
    //         {
    //             shared[tid] += shared[tid + i];
    //         }
    //         __syncthreads();
    //     }

    if (tid < 128)
    {
        shared[tid] += shared[tid + 128];
    }
    __syncthreads();
    if (tid < 64)
    {
        shared[tid] += shared[tid + 64];
    }
    __syncthreads();

    if (tid < 32)
    {
        warpReduce(shared, tid);
    }

    if (tid == 0)
        d_output[blockIdx.x] = shared[0];
}

bool check(float *out, float *res, int n)
{
    for (int i = 0; i < n; i++)
    {
        if (abs(out[i] - res[i]) > 1e-3)
            return false;
    }
    return true;
}

int main()
{
    const int N = 32 * 1024 * 1024;
    float *input = (float *)malloc(N * sizeof(float));
    float *d_input;
    cudaMalloc((void **)&d_input, N * sizeof(float));

    int block_num = N / THREAD_PER_BLOCK / 2;
    float *output = (float *)malloc(block_num * sizeof(float));
    float *d_output;
    cudaMalloc((void **)&d_output, block_num * sizeof(float));
    float *result = (float *)malloc(block_num * sizeof(float));

    for (int i = 0; i < N; i++)
    {
        input[i] = 2.0 * (float)drand48() - 1.0;
    }

    for (int i = 0; i < block_num; i++)
    {
        float cur = 0;
        for (int j = 0; j < 2 * THREAD_PER_BLOCK; j++)
        {
            cur += input[2 * i * THREAD_PER_BLOCK + j];
        }
        result[i] = cur;
    }

    cudaMemcpy(d_input, input, N * sizeof(float), cudaMemcpyHostToDevice);

    dim3 Grid(block_num, 1);
    dim3 Block(THREAD_PER_BLOCK, 1);

    reduce<<<Grid, Block>>>(d_input, d_output);

    cudaMemcpy(output, d_output, block_num * sizeof(float), cudaMemcpyDeviceToHost);
    if (check(output, result, block_num))
        printf("the ans is right\n");
    else
    {
        printf("the ans is wrong\n");
        for (int i = 0; i < block_num; i++)
        {
            printf("%lf ", output[i]);
        }
        printf("\n");
    }
    cudaFree(d_input);
    cudaFree(d_output);
}