#include "index_select.h"
__global__ void index_select_op(float* embedding, int* index,
    float* result, int dim, int length)
    {

        int threadId = blockIdx.x * gridDim.y * blockDim.x + blockIdx.y * blockDim.x + threadIdx.x;  
        for(int i=0;i<length;i++){
            // result[threadId*length + i] =embedding[threadId*length+index[i]];
            result[threadId + i*dim] =embedding[threadId + index[i]*dim];
            // result[threadId + i*dim] = threadId + i*dim;
            // result[threadId + i*dim] =threadId + index[i]*dim;
        }
    }


void index_select(const TensorCUDA& embedding,
    const TensorIntCUDA& index,
    TensorCUDA& result,
    int block_x, int block_y, int thread_x){
    float* _embedding_ptr = embedding.get();
    int* _index_ptr = index.get();
    float* _result_ptr  = result.get();
    dim3 threadsPerBlock(thread_x);  
    dim3 numBlocks(block_x,  block_y);
    index_select_op<<<numBlocks,threadsPerBlock>>>(_embedding_ptr, _index_ptr, _result_ptr, result.get_shape()[1],  result.get_shape()[0]);

}