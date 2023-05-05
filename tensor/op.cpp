#include "tensor/op.h"
#include "cuda_op/index_select.h"
#include "cuda_op/linear.h"


EmbeddingOP::EmbeddingOP(const std::string& file_path,
                    int myrank,
                    int ranks,
                    int block_x,
                    int block_y,
                    int thread_x){
    
    TensorCPU _tensor_cpu(file_path);
    cudaError_t cudaStatus;
    auto& shape  = _tensor_cpu.get_shape();
    int size = (shape[0]-1)/ranks + 1;
    _size = size;
    if(ranks == myrank+1){
        _size = shape[0]+size-ranks*size;
    }
    _dim = shape[1];
    
    if(myrank*size<shape[0]){
        _start = myrank *size;
        _end =  (myrank+1)*size;
        if(_end>shape[0]){
            _end = shape[0];
        }
        _size =_end-_start;
        _embedding = new TensorCUDA(std::vector<int>({_size, _dim}));
        _embedding->set_value(_tensor_cpu.get()+(myrank*size*_dim));
    }else{
        _embedding = new TensorCUDA(std::vector<int>({1, _dim}));
        _start = 0;
        _end = 0;
    }
    

    _block_x = block_x;
    _block_y = block_y;
    _thread_x = thread_x;
    _distribute =ranks>1;
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

void EmbeddingOP::process(const TensorIntCUDA& input, TensorCUDA& output, SynchronizeCUDA* sync){
    index_select(*_embedding, input, output, _block_x, _block_y, _thread_x, _start, _end);
    sync->AllReduce(output.get(), output.get(),output.get_size(), ncclFloat32,ncclSum);
}

TensorCUDA* EmbeddingOP::get_embedding(){
    return  _embedding;
}

int EmbeddingOP::get_start(){
    return  _start;
}



MatAddOP::MatAddOP(int block_x, int block_y,int thread_x){
    _block_x = block_x;
    _block_y = block_y;
    _thread_x = thread_x;
}


void MatAddOP::process(const TensorCUDA& input1, const TensorCUDA& input2, TensorCUDA& output){
    mat_add(input1, input2, output, _block_x, _block_y, _thread_x);
}


void MatAddOP::process(const TensorCUDA& input1, const TensorCUDA& input2, TensorCUDA& output,SynchronizeCUDA* sync){
    int size =input1.get_size()/sync->get_rank_size();
    mat_add(input1, input2, output, _block_x, _block_y, _thread_x,
            sync->get_rank()*size,sync->get_rank()*size+size);
    sync->AllGather(output.get()+sync->get_rank()*size, output.get(),size, ncclFloat32);
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

void LayerNormlizeOP::prcocess(TensorCUDA& result_pa, TensorCUDA& output_pa,SynchronizeCUDA* sync){
    int size = (result_pa.get_shape()[0]-1)/sync->get_rank_size()+1;
    int start = sync->get_rank()*size;
    int end=sync->get_rank()*size+size;
    if(end>result_pa.get_shape()[0]){
        end=result_pa.get_shape()[0];
    }
    int dim = result_pa.get_shape()[1];
    std::vector<int> shape = {end-start, dim};
    if(start<result_pa.get_shape()[0]){
        SubTensorCUDA result(&result_pa, shape, start*dim);
        SubTensorCUDA output(&output_pa, shape, start*dim);

        std::vector<int>  emb_out_mean_shape = {result.get_shape()[0]};
        TensorCUDA emb_out_mean(emb_out_mean_shape);
        get_reduce_mean(result, emb_out_mean);
        get_bias(result, emb_out_mean, output);

        TensorCUDA var(emb_out_mean_shape);
        get_var_scalar(output, var);
        normlize_scale(output, var);
    }
    for(int i =0;i<sync->get_rank_size()&&i*size<result_pa.get_shape()[0];i++){
        int count = size*dim;
        if(i*size+size>result_pa.get_shape()[0]){
            count = (result_pa.get_shape()[0]-i*size)*dim;
        }
        sync->BroadCast(output_pa.get()+i*size*dim,output_pa.get()+i*size*dim, count, ncclFloat32, i);
    }
   
}




MatMulOP::MatMulOP(){

}

void MatMulOP::process(const TensorCUDA& left, const TensorCUDA& right,TensorCUDA& result,cublasHandle_t& handle){
    matmul(left, right, result, handle);
}

void MatMulOP::process(const TensorCUDA& left, const TensorCUDA& right,TensorCUDA& result,cublasHandle_t& handle,SynchronizeCUDA* sync){
    TensorCUDA tmp(std::vector<int>({sync->get_rank_size(), left.get_shape()[0], right.get_shape()[1]}));
    SubTensorCUDA sub_tmp(&tmp, std::vector<int>({left.get_shape()[0], right.get_shape()[1]}), result.get_size()/sync->get_rank_size()*sync->get_rank());
    matmul(left, right, sub_tmp, handle);
    sync->AllGather(sub_tmp.get(), tmp.get(),result.get_size()/sync->get_rank_size(), ncclFloat32);
    tmp.reshape_copy(result);


}
