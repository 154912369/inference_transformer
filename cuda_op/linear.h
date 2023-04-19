#pragma once
#include "tensor/tensor_cuda.h"
void mat_add(const TensorCUDA& input1,
                const TensorCUDA& input2,
                TensorCUDA& result,
                int block_x, int block_y, int thread_x);

void mat_2d_reduce_sum(const TensorCUDA& input1,
                TensorCUDA& result,
                int block_x,  int thread_x);

void mat_reduce_vector(TensorCUDA& input,
                TensorCUDA& mean,
                int block_x,  int thread_x);

void mat_2d_reduce_var(TensorCUDA& input,
                TensorCUDA& mean,
                int block_x,  int thread_x);

void mat_layer_normlize_scale(const TensorCUDA& input1, const TensorCUDA& var, const TensorCUDA& scalar, int block_x,  int thread_x);