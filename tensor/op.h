#pragma once
#include "tensor/tensor_cuda.h"

class MatAddOP{
    int _block_x;
    int _block_y;
    int _thread_x;
    public:
        MatAddOP(int block_x = 2, int block_y = 32, int thread_x = 32);
        void process(const TensorCUDA&, const TensorCUDA&, TensorCUDA& output);
};

class EmbeddingOP{
    private:
        TensorCUDA* _embedding=NULL;
        int _block_x;
        int _block_y;
        int _thread_x;
        int _dim;
        int _size;

       public:
        EmbeddingOP(const std::string& file_path,
                    int block_x=2,
                    int block_y=32,
                    int thread_x=32);
        ~EmbeddingOP();
        void process(const TensorIntCUDA&, TensorCUDA& output);
        int get_dim();
        int get_size();
};