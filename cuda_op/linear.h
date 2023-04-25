#pragma once
#include "tensor/tensor_cuda.h"
void mat_add(const TensorCUDA& input1,
                const TensorCUDA& input2,
                TensorCUDA& result,
                int block_x, int block_y, int thread_x);

void mat_2d_reduce_sum(const TensorCUDA& input1,
                TensorCUDA& result,
                int block_x,  int thread_x);

void mat_3d_reduce_sum(const TensorCUDA& input1,
                TensorCUDA& result);
void batch_mat_divide_mat(TensorCUDA& input,
                TensorCUDA& mean);
void mat_reduce_vector(TensorCUDA& input,
                TensorCUDA& mean,TensorCUDA& output,
                int block_x,  int thread_x);

void mat_2d_reduce_var(TensorCUDA& input,
                TensorCUDA& mean,
                int block_x,  int thread_x);

void mat_layer_normlize_scale(const TensorCUDA& input1, const TensorCUDA& var, const TensorCUDA& scalar, int block_x,  int thread_x);

void matmul(const TensorCUDA& left,
            const TensorCUDA& right,
            TensorCUDA& result);

void batch_matmul(const TensorCUDA& left,
            const TensorCUDA& right,
            TensorCUDA& result);

void batch_matmul_without_transpose(const TensorCUDA& left,
            const TensorCUDA& right,
            TensorCUDA& result);

int get_last_token(const TensorCUDA& left,
            const TensorCUDA& right);
