#include "cuda_op/linear.h"
__global__ void mat_add_impl(float* input1, float* input2,
    float* output, int thread_size, int step)
    {

        int threadId = blockIdx.x * gridDim.y * blockDim.x + blockIdx.y * blockDim.x + threadIdx.x;  
        for(int i=0;i<step;i++){
            output[threadId*step + i] = input1[threadId*step + i]+input2[threadId*step + i];
        }
    }
void mat_add(const TensorCUDA& input1,
    const TensorCUDA& input2,
    TensorCUDA& result,
    int block_x, int block_y, int thread_x){
        dim3 threadsPerBlock(thread_x);  
        dim3 numBlocks(block_x,  block_y);
        int thread_size = block_x*block_y*thread_x;
        int step = input1.get_size()/thread_size;
        mat_add_impl<<<numBlocks,threadsPerBlock>>>(input1.get(), input2.get(), result.get(),thread_size,  step);
    }