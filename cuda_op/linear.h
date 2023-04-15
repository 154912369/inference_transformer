#pragma once
#include "tensor/tensor_cuda.h"
void mat_add(const TensorCUDA& input1,
                const TensorCUDA& input2,
                TensorCUDA& result,
                int block_x, int block_y, int thread_x);
