#include "tensor/tensor_cuda.h"
void rope(const TensorCUDA& input1,
                const TensorCUDA& input2,
                TensorCUDA& result,
                int block_x,  int thread_x);