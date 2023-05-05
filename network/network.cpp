#include "network/network.h"
#include "cuda_op/linear.h"
#include "cuda_op/common.h"
#include<unistd.h>  


std::string repace_num(std::string input,int layer){
    std::string res;
    std::string old_str = "%d";
    std::string str = std::to_string(layer);
    size_t pos = input.find(old_str);
    if (pos != std::string::npos) {
        input.replace(pos, old_str.length(), str);
    }
    return input;
}

void Transformer::init_root(const std::string& model_path){

    // sub op
    _word_embedding = new EmbeddingOP(model_path+"word_embedding", _synchronize_cuda->get_rank(), _synchronize_cuda->get_rank_size());
    _sent_embedding = new EmbeddingOP(model_path+"sent_embedding", _synchronize_cuda->get_rank(), _synchronize_cuda->get_rank_size());
    _role_embedding = new EmbeddingOP(model_path+"role_embedding", _synchronize_cuda->get_rank(), _synchronize_cuda->get_rank_size());
    _pos_embedding = new EmbeddingOP(model_path+"pos_embedding",_synchronize_cuda->get_rank(), _synchronize_cuda->get_rank_size(), 1,2,32);
    _post_encoder_layer_norm =  new LayerNormlizeOP(model_path+"post_encoder_layer_norm_scale");
    _matmulop  = new MatMulOP();
    _rope_op = new RoPeOP();
    int myrank = _synchronize_cuda->get_rank();
    int ranks = _synchronize_cuda->get_rank_size();
    for(int i =0; i < _layer_size; i++){
            //params
        _reduce_2d_sum_op.push_back(std::unique_ptr<LayerNormlizeOP>(
                new LayerNormlizeOP(repace_num(model_path+"encoder_layer_%d_pre_att_layer_norm_scale", i))));
        _pre_ffn_layer_norm.push_back(std::unique_ptr<LayerNormlizeOP>(
                new LayerNormlizeOP(repace_num(model_path+"encoder_layer_%d_pre_ffn_layer_norm_scale", i))));

        _query_w.push_back(std::unique_ptr<TensorCUDA>(
            new TensorCUDA(repace_num(model_path+"encoder_layer_%d_multi_head_att_query_fc.w_0", i),myrank,ranks)));
        _key_w.push_back(std::unique_ptr<TensorCUDA>(
            new TensorCUDA(repace_num(model_path+"encoder_layer_%d_multi_head_att_key_fc.w_0", i),myrank,ranks)));
        _value_w.push_back(std::unique_ptr<TensorCUDA>(
            new TensorCUDA(repace_num(model_path+"encoder_layer_%d_multi_head_att_value_fc.w_0", i),myrank,ranks)));
        _output_w.push_back(std::unique_ptr<TensorCUDA>(
            new TensorCUDA(repace_num(model_path+"encoder_layer_%d_multi_head_att_output_fc.w_0", i),myrank,ranks)));
        _ffn0_weight.push_back(std::unique_ptr<TensorCUDA>(
            new TensorCUDA(repace_num(model_path+"encoder_layer_%d_ffn_fc_0.w_0", i),myrank,ranks)));
        _ffn1_weight.push_back(std::unique_ptr<TensorCUDA>(
            new TensorCUDA(repace_num(model_path+"encoder_layer_%d_ffn_fc_1.w_0", i),myrank,ranks)));
    }

    

}
void Transformer::sync_length(int length){
    if(_disribute && _synchronize_cuda->get_rank() != 0){
        length = _synchronize->get_synchronize_length();
       
    }else{
        if(_disribute){
            _synchronize->set_length(length);
        }
    }
}
TensorCUDA* Transformer::get_embedding_out(int* token_type_list,
                    int* role_ids_list,
                    int* sent_ids_list,
                    int length){
     if(_disribute && _synchronize_cuda->get_rank() != 0){
        length = _synchronize->get_length();
       
    }
    TensorIntCUDA token_ids(length);
    TensorIntCUDA role_ids( length);
    TensorIntCUDA sent_ids(length);
    
    if(!_disribute || _synchronize_cuda->get_rank() == 0){
        token_ids.set_result(token_type_list);
        role_ids.set_result(role_ids_list);
        sent_ids.set_result(sent_ids_list);
    }

    _synchronize_cuda->BroadCast(token_ids.get(), token_ids.get(), token_ids.get_size(),  ncclInt );
    _synchronize_cuda->BroadCast(role_ids.get(), role_ids.get(), token_ids.get_size(),  ncclInt );
    _synchronize_cuda->BroadCast(sent_ids.get(), sent_ids.get(), token_ids.get_size(),  ncclInt );
    

    
    
    TensorCUDA token_type_embedding(std::vector<int>({token_ids.get_size(), _word_embedding->get_dim()}));
    TensorCUDA role_type_embedding(std::vector<int>({token_ids.get_size(), _word_embedding->get_dim()}));
    TensorCUDA sent_type_embedding(std::vector<int>({token_ids.get_size(), _word_embedding->get_dim()}));
    TensorCUDA* emb_out = new TensorCUDA(std::vector<int>({token_ids.get_size(), _word_embedding->get_dim()}));

    
    if(_disribute){
         _word_embedding->process(token_ids, token_type_embedding, _synchronize_cuda);
        _role_embedding->process(role_ids, role_type_embedding, _synchronize_cuda);
        _sent_embedding->process(sent_ids,sent_type_embedding, _synchronize_cuda);
        _mat_add_op->process(token_type_embedding, role_type_embedding, *emb_out, _synchronize_cuda);
         _mat_add_op->process(*emb_out, sent_type_embedding, *emb_out, _synchronize_cuda);
    }else{
        _word_embedding->process(token_ids, token_type_embedding);
        _role_embedding->process(role_ids, role_type_embedding);
        _sent_embedding->process(sent_ids,sent_type_embedding);
        _mat_add_op->process(token_type_embedding, role_type_embedding, *emb_out);
        _mat_add_op->process(*emb_out, sent_type_embedding, *emb_out);
    }

    
    return emb_out;
}

TensorCUDA* Transformer::get_pos_embedding_out(int* token_type_list, int length){
    if(_disribute && _synchronize_cuda->get_rank() != 0){
        length = _synchronize->get_length();
       
    }
    TensorIntCUDA token_ids(length);
    
    if(!_disribute || _synchronize_cuda->get_rank() == 0){
        token_ids.set_result(token_type_list);
    }
     _synchronize_cuda->BroadCast(token_ids.get(), token_ids.get(), token_ids.get_size(),  ncclInt );
    TensorCUDA* pos_type_embedding= new TensorCUDA(std::vector<int>({length, _pos_embedding->get_dim()}));
    
    if(_disribute){
        _pos_embedding->process(token_ids, *pos_type_embedding,_synchronize_cuda);
    }else{
         _pos_embedding->process(token_ids, *pos_type_embedding);
    }
    return pos_type_embedding;
}


void Transformer::get_q_k_v(TensorCUDA& tensor, TensorCUDA& pos_type_embedding, TensorCUDA& q, TensorCUDA& k, TensorCUDA& v,int layer_index,  KeyValueCache& key_value_cache,cublasHandle_t& handle){
    std::vector<int> shape = tensor.get_shape();
    shape[1]=shape[1]/_synchronize_cuda->get_rank_size();
    TensorDynamicCUDA* key_cache = new TensorDynamicCUDA(shape);
    TensorDynamicCUDA* value_cache = new TensorDynamicCUDA(shape);
    // SubTensorCUDA attention(&attention_pa, head_shape, q.get_size()*_synchronize_cuda->get_rank());

    _matmulop->process(  tensor, *_query_w[layer_index],*key_cache ,handle);

    _rope_op->process(*key_cache,pos_type_embedding,*key_cache);
    key_cache->reshape_copy(q);


    _matmulop->process( tensor, *_key_w[layer_index], *key_cache,handle);
    _rope_op->process(*key_cache,pos_type_embedding,*key_cache);
    key_cache->reshape_copy(k);


    _matmulop->process( tensor, *_value_w[layer_index], *value_cache,handle);
    value_cache->reshape_copy(v);
    key_value_cache.add_cache(key_cache,value_cache);

}

Transformer::Transformer(const std::string& path, int layer_size,SynChronize* synchronize,int myRank, int nRanks, int localRank){
    _layer_size = layer_size;
    _synchronize = synchronize;
    init_cuda();
    _mat_add_op =  new MatAddOP();
    _synchronize_cuda = new SynchronizeCUDA(myRank, nRanks,localRank);
    if(nRanks > 1){
        _synchronize = synchronize;
        _synchronize_cuda->syn();
        _disribute = true;
    }
    init_root(path);
}
Transformer::~Transformer(){
    if(_word_embedding){
        delete _word_embedding;
    }
    if(_role_embedding){
        delete _role_embedding;
    }
    if(_sent_embedding){
        delete _sent_embedding;
    }
    if(_pos_embedding){
        delete _pos_embedding;
    }
    if(_mat_add_op){
        delete _mat_add_op;
    }

    if(_matmulop){
        delete _matmulop;
    }

    if(_rope_op){
        delete _rope_op;
    }

    if(_post_encoder_layer_norm){
        delete _post_encoder_layer_norm;
    }

    if(_synchronize){
        delete _synchronize;
    }

    if(_synchronize_cuda){
        delete _synchronize_cuda;
    }

 
}

void Transformer::get_attention_output(TensorCUDA& tensor,TensorCUDA& pos_type_embedding, int layer_index, KeyValueCache& key_value_cache, cublasHandle_t& handle){
     std::vector<int> head_shape = {32/_synchronize_cuda->get_rank_size(), tensor.get_shape()[0], tensor.get_shape()[1]/32};
    TensorCUDA q(head_shape );
    TensorCUDA k(head_shape ); 
    TensorCUDA v(head_shape ); //32*3*64
    get_q_k_v(tensor, pos_type_embedding, q, k, v,layer_index,key_value_cache,handle);
    TensorCUDA p(std::vector<int>({q.get_shape()[0],q.get_shape()[1],q.get_shape()[1]}));
    batch_matmul(q,k,p,handle); 
    expf(p);
    TensorCUDA attention_pa(std::vector<int>({32,q.get_shape()[1],q.get_shape()[2]})); //32*3*3
    SubTensorCUDA attention(&attention_pa, head_shape, q.get_size()*_synchronize_cuda->get_rank());
    batch_matmul_without_transpose(p,v,attention,handle);
   
    TensorCUDA mean(std::vector<int>({q.get_shape()[0],q.get_shape()[1]}));
    mat_3d_reduce_sum(p, mean);
    // transformer.
    batch_mat_divide_mat(attention,mean);
    
    if(_disribute){

        _synchronize_cuda->AllGather(attention.get(), attention_pa.get(), attention.get_size(), ncclFloat32);
    }
    
    TensorCUDA result(tensor.get_shape());

    attention_pa.reshape_copy(result);
    _matmulop->process(  result, *_output_w[layer_index], tensor,handle, _synchronize_cuda);

}
void Transformer::gelu_distribute(TensorCUDA& input){
    int size =input.get_size()/_synchronize_cuda->get_rank_size();
    gelu(input, _synchronize_cuda->get_rank()*size, size);
    _synchronize_cuda->AllGather(input.get()+_synchronize_cuda->get_rank()*size, input.get(),size, ncclFloat32);
}
void Transformer::encoder_layer(TensorCUDA& tensor,TensorCUDA& pos_type_embedding, int layer_index, KeyValueCache& key_value_cache, cublasHandle_t& handle){
    TensorCUDA result(tensor.get_shape());
   
    if(_disribute){ 
        _reduce_2d_sum_op[layer_index]->prcocess(tensor, result, _synchronize_cuda);
        
        get_attention_output(result,pos_type_embedding,layer_index, key_value_cache,handle);
        _mat_add_op->process(tensor, result,result, _synchronize_cuda);

        _pre_ffn_layer_norm[layer_index]->prcocess(result, tensor, _synchronize_cuda);
        
        TensorCUDA tmp1(std::vector<int>({result.get_shape()[0], result.get_shape()[1]*4}));
        _matmulop->process(tensor, *_ffn0_weight[layer_index],tmp1,handle,_synchronize_cuda);
        gelu_distribute(tmp1);

        _matmulop->process(tmp1,*_ffn1_weight[layer_index],tensor,handle,_synchronize_cuda);
        _mat_add_op->process(tensor, result,tensor,_synchronize_cuda);
    }else{

        _reduce_2d_sum_op[layer_index]->prcocess(tensor, result);
        get_attention_output(result,pos_type_embedding,layer_index, key_value_cache,handle);

        _mat_add_op->process(tensor, result,result);

        _pre_ffn_layer_norm[layer_index]->prcocess(result, tensor);


        TensorCUDA tmp1(std::vector<int>({result.get_shape()[0], result.get_shape()[1]*4}));
        _matmulop->process(tensor, *_ffn0_weight[layer_index],tmp1,handle);
         
        gelu(tmp1);
               
        _matmulop->process(tmp1,*_ffn1_weight[layer_index],tensor,handle);
        _mat_add_op->process(tensor, result,tensor);
    }
    
}

void Transformer::encode(TensorCUDA& tensor,TensorCUDA& pos_type_embedding,  KeyValueCache& key_value_cache, cublasHandle_t& handle){
    for(int i=0;i<_layer_size;i++){
        encoder_layer(tensor, pos_type_embedding, i,key_value_cache,handle);
    }
    if(_disribute){
        _post_encoder_layer_norm->prcocess(tensor, tensor);
    }else{
        _post_encoder_layer_norm->prcocess(tensor, tensor, _synchronize_cuda);
    }
    
}

void Transformer::decoder_layer(TensorCUDA& tensor,TensorCUDA& pos_type_embedding, int layer_index, KeyValueCache& key_value_cache, cublasHandle_t& handle){
    TensorCUDA result(tensor.get_shape());
    if(_disribute){ 
        _reduce_2d_sum_op[layer_index]->prcocess(tensor, result, _synchronize_cuda);
        decoder_get_attention_output(result,pos_type_embedding,layer_index, key_value_cache,handle);
        _mat_add_op->process(tensor, result,result, _synchronize_cuda);
        _pre_ffn_layer_norm[layer_index]->prcocess(result, tensor, _synchronize_cuda);
        TensorCUDA tmp1(std::vector<int>({result.get_shape()[0], result.get_shape()[1]*4}));
        _matmulop->process(tensor, *_ffn0_weight[layer_index],tmp1,handle,_synchronize_cuda);
        gelu_distribute(tmp1);
        _matmulop->process(tmp1,*_ffn1_weight[layer_index],tensor,handle,_synchronize_cuda);
        _mat_add_op->process(tensor, result,tensor,_synchronize_cuda);
    }else{
        _reduce_2d_sum_op[layer_index]->prcocess(tensor, result);
        decoder_get_attention_output(result,pos_type_embedding,layer_index, key_value_cache,handle);
        _mat_add_op->process(tensor, result,result);
        _pre_ffn_layer_norm[layer_index]->prcocess(result, tensor);
        TensorCUDA tmp1(std::vector<int>({result.get_shape()[0], result.get_shape()[1]*4}));
        _matmulop->process(tensor, *_ffn0_weight[layer_index],tmp1,handle);
        gelu(tmp1);
        _matmulop->process(tmp1,*_ffn1_weight[layer_index],tensor,handle);
        _mat_add_op->process(tensor, result,tensor);
    }
    

}

void Transformer::decode(TensorCUDA& tensor,TensorCUDA& pos_type_embedding,  KeyValueCache& key_value_cache, cublasHandle_t& handle){
    for(int i=0;i<_layer_size;i++){
        decoder_layer(tensor, pos_type_embedding, i,key_value_cache,handle);
    }

    if(_disribute){
        _post_encoder_layer_norm->prcocess(tensor, tensor);
    }else{
        _post_encoder_layer_norm->prcocess(tensor, tensor, _synchronize_cuda);
    }
}

void Transformer::decoder_get_attention_output(TensorCUDA& tensor,TensorCUDA& pos_type_embedding, int layer_index, KeyValueCache& key_value_cache, cublasHandle_t& handle){
    std::vector<int> head_shape = {32/_synchronize_cuda->get_rank_size(), tensor.get_shape()[0], tensor.get_shape()[1]/32};
    std::vector<int> kv_head_shape = {32/_synchronize_cuda->get_rank_size(), key_value_cache.get_step()+ tensor.get_shape()[0], tensor.get_shape()[1]/32};
    TensorCUDA q(head_shape );//32,346,64
    TensorCUDA k(kv_head_shape );
    TensorCUDA v(kv_head_shape ); //32,346*2,64
    decoder_get_q_k_v(tensor, pos_type_embedding, q, k, v,layer_index,key_value_cache,handle);

    TensorCUDA p(std::vector<int>({q.get_shape()[0],q.get_shape()[1],k.get_shape()[1]})); //32,346,346*2

    batch_matmul(q,k,p,handle);
    expf(p);

    TensorCUDA attention_pa(std::vector<int>({32,p.get_shape()[1],v.get_shape()[2]})); //32,346,64
    SubTensorCUDA attention(&attention_pa, head_shape, q.get_size()*_synchronize_cuda->get_rank());
    batch_matmul_without_transpose(p,v,attention, handle);
    TensorCUDA mean(std::vector<int>({p.get_shape()[0],p.get_shape()[1]}));
    mat_3d_reduce_sum(p, mean);
    // transformer.
    batch_mat_divide_mat(attention,mean);
    if(_disribute){
        _synchronize_cuda->AllGather(attention.get(), attention_pa.get(), attention.get_size(), ncclFloat32);
    }
    // if(_synchronize_cuda->get_rank()==0){
    //     TensorCUDA tmp("/data1/renweijie/baidu/dialogue/nlg-paddle-inference/transfer_output/encode_result");
    //     if(tmp.equal(attention_pa)){
    //         printf("rank %d is equal\n", _synchronize_cuda->get_rank());
    //     }else{
    //         printf("rank %d is not equal\n",_synchronize_cuda->get_rank());
    //     }
    //     attention_pa.save("encode_result");
    // }
    TensorCUDA result(tensor.get_shape());
    // q.reshape(tensor.get_shape());
    attention_pa.reshape_copy(result);
    _matmulop->process(  result, *_output_w[layer_index], tensor,handle, _synchronize_cuda);

}

void Transformer::decoder_get_q_k_v(TensorCUDA& tensor, TensorCUDA& pos_type_embedding, TensorCUDA& q, TensorCUDA& k, TensorCUDA& v,int layer_index,  KeyValueCache& key_value_cache, cublasHandle_t& handle){

    std::vector<int> shape = tensor.get_shape();
    shape[1]=shape[1]/_synchronize_cuda->get_rank_size();
    TensorCUDA tmp(shape);
    
    _matmulop->process(  tensor, *_query_w[layer_index],tmp,handle);
    _rope_op->process(tmp,pos_type_embedding,tmp);
    tmp.reshape_copy(q);

   
    _matmulop->process( tensor, *_key_w[layer_index], tmp,handle);
    _rope_op->process(tmp,pos_type_embedding,tmp);
    TensorCUDA* key_cache = key_value_cache.incr_key_cache(tmp, layer_index);
    key_cache->reshape_copy(k);



    _matmulop->process( tensor, *_value_w[layer_index], tmp,handle);
    key_cache = key_value_cache.incr_value_cache(tmp, layer_index);
    key_cache->reshape_copy(v);

}


int Transformer::predict_last_token(TensorCUDA& tensor, cublasHandle_t& handle){
    int* token_id;
    float* token_score;
    int token_id_host[_synchronize_cuda->get_rank_size()];
    float token_score_host[_synchronize_cuda->get_rank_size()];
    cudaError_t cudaStatus = cudaMalloc(&token_id, sizeof(int)*_synchronize_cuda->get_rank_size());
    if (cudaStatus != cudaSuccess) {
        printf("cudaMalloc failed: %s\n", cudaGetErrorString(cudaStatus));
        // 进行错误处理
    }
    cudaStatus = cudaMalloc(&token_score, sizeof(float)*_synchronize_cuda->get_rank_size());
    if (cudaStatus != cudaSuccess) {
        printf("cudaMalloc failed: %s\n", cudaGetErrorString(cudaStatus));
        // 进行错误处理
    }

    get_last_token(*_word_embedding->get_embedding(), tensor,token_score+ _synchronize_cuda->get_rank(),token_id+ _synchronize_cuda->get_rank(), handle,_word_embedding->get_start());
        cudaStatus = cudaMemcpy(token_score_host, token_score, sizeof(float)*_synchronize_cuda->get_rank_size(), cudaMemcpyDeviceToHost);
    if (cudaStatus != cudaSuccess) {
        printf(" get_last_token cudaMemcpy failed: %s\n", cudaGetErrorString(cudaStatus));
        // 进行错误处理
    }
    cudaStatus = cudaMemcpy(token_id_host, token_id, sizeof(int)*_synchronize_cuda->get_rank_size(), cudaMemcpyDeviceToHost);
    if (cudaStatus != cudaSuccess) {
        printf(" get_last_token cudaMemcpy failed: %s\n", cudaGetErrorString(cudaStatus));
        // 进行错误处理
    }
    if(_disribute){
        _synchronize_cuda->AllGather(token_id+ _synchronize_cuda->get_rank(),token_id,1,ncclInt);
        _synchronize_cuda->AllGather(token_score+ _synchronize_cuda->get_rank(),token_score,1,ncclFloat);
    }


    cudaStatus = cudaMemcpy(token_id_host, token_id, sizeof(int)*_synchronize_cuda->get_rank_size(), cudaMemcpyDeviceToHost);
    if (cudaStatus != cudaSuccess) {
        printf(" get_last_token cudaMemcpy failed: %s\n", cudaGetErrorString(cudaStatus));
        // 进行错误处理
    }

    cudaStatus = cudaMemcpy(token_score_host, token_score, sizeof(float)*_synchronize_cuda->get_rank_size(), cudaMemcpyDeviceToHost);
    if (cudaStatus != cudaSuccess) {
        printf(" get_last_token cudaMemcpy failed: %s\n", cudaGetErrorString(cudaStatus));
        // 进行错误处理
    }

    cudaStatus = cudaFree(token_score);
    if (cudaStatus != cudaSuccess) {
        printf("cudaFree token score failed: %s\n", cudaGetErrorString(cudaStatus));
        // 进行错误处理
    }
    cudaStatus = cudaFree(token_id);
    if (cudaStatus != cudaSuccess) {
        printf("cudaFree token id failed: %s\n", cudaGetErrorString(cudaStatus));
        // 进行错误处理
    }
    int max_id = token_id_host[0];
    float max_score =  token_score_host[0];
    for(int i=0;i<_synchronize_cuda->get_rank_size();i++){
        if(max_score<token_score_host[i]){
            max_score = token_score_host[i];
            max_id = token_id_host[i];

        }
    }
    return max_id;
}