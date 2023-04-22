#pragma once
#include "tensor/tensor_cuda.h"
void reshape_copy_2d(TensorCUDA& input, TensorCUDA& result);

void expf(TensorCUDA& input);
void gelu(TensorCUDA& input);