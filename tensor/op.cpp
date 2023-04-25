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

TensorCUDA* EmbeddingOP::get_embedding(){
    return  _embedding;
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

// class ReduceSumOP{
//     int _block_x;
//     int _thread_x;
//     public:
//         ReduceSumOP(int block_x = 2, int thread_x = 1024);
//         void process(const TensorCUDA&,TensorCUDA& output);
// };

LayerNormlizeOP::LayerNormlizeOP(const std::string& file_path,int block_x, int thread_x){
    TensorCPU tmp(file_path);
    _scalar = new TensorCUDA(tmp);
    _block_x = block_x;
    _thread_x = thread_x;

}
LayerNormlizeOP::~LayerNormlizeOP(){
    if(_scalar==NULL){
        delete _scalar;
    }
}

void LayerNormlizeOP::get_reduce_mean(const TensorCUDA& input,TensorCUDA& output){
    mat_2d_reduce_sum(input,output,_block_x,_thread_x);
  
}

void LayerNormlizeOP::get_bias(TensorCUDA& input, TensorCUDA& mean, TensorCUDA& output){
    mat_reduce_vector(input, mean,output, _block_x, _thread_x);
}

void LayerNormlizeOP::get_var_scalar(TensorCUDA& input, TensorCUDA& var){
    mat_2d_reduce_var(input, var, _block_x, _thread_x);
}

void LayerNormlizeOP::normlize_scale(TensorCUDA& input, TensorCUDA& var){
    mat_layer_normlize_scale(input, var, *_scalar, _block_x, _thread_x);

}

void LayerNormlizeOP::prcocess(TensorCUDA& result, TensorCUDA& output){
    std::vector<int>  emb_out_mean_shape = {result.get_shape()[0]};
    TensorCUDA emb_out_mean(emb_out_mean_shape);
    get_reduce_mean(result, emb_out_mean);
    get_bias(result, emb_out_mean, output);

    TensorCUDA var(emb_out_mean_shape);
    get_var_scalar(output, var);
    normlize_scale(output, var);
}




MatMulOP::MatMulOP(){

}

void MatMulOP::process(const TensorCUDA& left, const TensorCUDA& right,TensorCUDA& result){
    matmul(left, right, result);
}
