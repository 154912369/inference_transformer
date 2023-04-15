#include "tensor/op.h"
#include "cuda_op/index_select.h"
#include "cuda_op/linear.h"


EmbeddingOP::EmbeddingOP(const std::string& file_path,
                    int block_x,
                    int block_y,
                    int thread_x){
    TensorCPU tmp(file_path);
    _embedding = new TensorCUDA(tmp);
    _block_x = block_x;
    _block_y = block_y;
    _thread_x = thread_x;
    _dim = _embedding->get_shape()[1];
    _size = _embedding->get_shape()[0];
}
EmbeddingOP::~EmbeddingOP(){
    if(_embedding!=NULL){
        delete _embedding;
    }
}
int EmbeddingOP::get_dim() { return _dim; }
int EmbeddingOP::get_size() { return _size; }

void EmbeddingOP::process(const TensorIntCUDA& input, TensorCUDA& output){
    index_select(*_embedding, input, output, _block_x, _block_y, _thread_x);
}


MatAddOP::MatAddOP(int block_x, int block_y,int thread_x){
    _block_x = block_x;
    _block_y = block_y;
    _thread_x = thread_x;
}
void process(const TensorCUDA&, const TensorCUDA&, TensorCUDA& output);
void MatAddOP::process(const TensorCUDA& input1, const TensorCUDA& input2, TensorCUDA& output){
    mat_add(input1, input2, output, _block_x, _block_y, _thread_x);
}