#pragma once
#include "tensor/tensor_cuda.h"
void index_select(const TensorCUDA& embedding,
                const TensorIntCUDA& index,
                TensorCUDA& result,
                int block_x,
                int block_y,
                int thread_x);
void index_select(const TensorCUDA& embedding,
                const TensorIntCUDA& index,
                TensorCUDA& result,
                int block_x,
                int block_y,
                int thread_x,int,int);