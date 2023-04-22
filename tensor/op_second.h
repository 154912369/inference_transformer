#pragma once
#include "tensor/tensor_cuda.h"

class RoPeOP{
    int _block_x;
    int _thread_x;
    public:
        RoPeOP(int block_x = 2, int thread_x = 1024);
        void process(const TensorCUDA&, const TensorCUDA&, TensorCUDA& output);
};