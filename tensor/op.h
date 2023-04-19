#pragma once
#include "tensor/tensor_cuda.h"

class MatAddOP{
    int _block_x;
    int _block_y;
    int _thread_x;
    public:
        MatAddOP(int block_x = 2, int block_y = 1, int thread_x = 1024);
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
                    int block_y=1,
                    int thread_x=1024);
        ~EmbeddingOP();
        void process(const TensorIntCUDA&, TensorCUDA& output);
        int get_dim();
        int get_size();
};

class LayerNormlizeOP{
    private:
        int _block_x;
        int _thread_x;
        TensorCUDA* _scalar = NULL;
        void get_bias(TensorCUDA& input, TensorCUDA& mean);
        void get_reduce_mean(const TensorCUDA&,TensorCUDA& output);
        void get_var_scalar(TensorCUDA& input, TensorCUDA& mean);
        void normlize_scale(TensorCUDA& input, TensorCUDA& var);

    public:
        LayerNormlizeOP(const std::string& file_path, int block_x = 1, int thread_x = 1024);
        ~LayerNormlizeOP();
        void prcocess(TensorCUDA&);
};