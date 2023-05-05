#pragma once
#include "tensor/tensor_cuda.h"
#include <cublas_v2.h>
#include "multi/synchronize_cuda.h"

class MatAddOP{
    int _block_x;
    int _block_y;
    int _thread_x;
    public:
        MatAddOP(int block_x = 2, int block_y = 1, int thread_x = 1024);
        void process(const TensorCUDA&, const TensorCUDA&, TensorCUDA& output);
        void process(const TensorCUDA& input1, const TensorCUDA& input2, TensorCUDA& output,SynchronizeCUDA* sync);
};

class EmbeddingOP{
    private:
        TensorCUDA* _embedding=NULL;
        int _block_x;
        int _block_y;
        int _thread_x;
        int _dim;
        int _size;
        int _start;
        int _end;
        int _distribute;

       public:
        EmbeddingOP(const std::string& file_path,
                    int myrank=0,
                    int ranks=1,
                    int block_x=2,
                    int block_y=1,
                    int thread_x=1024);
        ~EmbeddingOP();
        void process(const TensorIntCUDA&, TensorCUDA& output);
        void process(const TensorIntCUDA& input, TensorCUDA& output, SynchronizeCUDA* sync);
        int get_dim();
        int get_size();
        TensorCUDA* get_embedding();
        int get_start();
};

class LayerNormlizeOP{
    private:
        int _block_x;
        int _thread_x;
        TensorCUDA* _scalar = NULL;
        void get_bias(TensorCUDA& input, TensorCUDA& mean,TensorCUDA& output);
        void get_reduce_mean(const TensorCUDA&,TensorCUDA& output);
        void get_var_scalar(TensorCUDA& input, TensorCUDA& mean);
        void normlize_scale(TensorCUDA& input, TensorCUDA& var);

    public:
        LayerNormlizeOP(const std::string& file_path, int block_x = 1, int thread_x = 1024);
        ~LayerNormlizeOP();
        void prcocess(TensorCUDA&,TensorCUDA&);
        void prcocess(TensorCUDA&,TensorCUDA&,SynchronizeCUDA*);
};



class MatMulOP{
    public:
        MatMulOP();
        void process(const TensorCUDA&,const TensorCUDA& ,TensorCUDA&,cublasHandle_t&);
        void process(const TensorCUDA& left, const TensorCUDA& right,TensorCUDA& result,cublasHandle_t& handle,SynchronizeCUDA* sync);

};