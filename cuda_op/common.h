#pragma once
#include "tensor/tensor_cuda.h"
void reshape_copy_2d(TensorCUDA& input, TensorCUDA& result);

void expf(TensorCUDA& input,bool is_mask=true);
void gelu(TensorCUDA& input);
void init_cuda();