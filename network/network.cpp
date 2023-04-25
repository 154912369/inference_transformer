#include "network/network.h"
#include "cuda_op/linear.h"
#include "cuda_op/common.h"


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
    _word_embedding = new EmbeddingOP(model_path+"word_embedding");
    _sent_embedding = new EmbeddingOP(model_path+"sent_embedding");
    _role_embedding = new EmbeddingOP(model_path+"role_embedding");
    _pos_embedding = new EmbeddingOP(model_path+"pos_embedding",1,2,32);
    _post_encoder_layer_norm =  new LayerNormlizeOP(model_path+"post_encoder_layer_norm_scale");
    _matmulop  = new MatMulOP();
    _rope_op = new RoPeOP();
    for(int i =0; i < _layer_size; i++){
            //params
        _reduce_2d_sum_op.push_back(std::unique_ptr<LayerNormlizeOP>(
                new LayerNormlizeOP(repace_num(model_path+"encoder_layer_%d_pre_att_layer_norm_scale", i))));
        _pre_ffn_layer_norm.push_back(std::unique_ptr<LayerNormlizeOP>(
                new LayerNormlizeOP(repace_num(model_path+"encoder_layer_%d_pre_ffn_layer_norm_scale", i))));

        _query_w.push_back(std::unique_ptr<TensorCUDA>(
            new TensorCUDA(repace_num(model_path+"encoder_layer_%d_multi_head_att_query_fc.w_0", i))));
        _key_w.push_back(std::unique_ptr<TensorCUDA>(
            new TensorCUDA(repace_num(model_path+"encoder_layer_%d_multi_head_att_key_fc.w_0", i))));
        _value_w.push_back(std::unique_ptr<TensorCUDA>(
            new TensorCUDA(repace_num(model_path+"encoder_layer_%d_multi_head_att_value_fc.w_0", i))));
        _output_w.push_back(std::unique_ptr<TensorCUDA>(
            new TensorCUDA(repace_num(model_path+"encoder_layer_%d_multi_head_att_output_fc.w_0", i))));
        _ffn0_weight.push_back(std::unique_ptr<TensorCUDA>(
            new TensorCUDA(repace_num(model_path+"encoder_layer_%d_ffn_fc_0.w_0", i))));
        _ffn1_weight.push_back(std::unique_ptr<TensorCUDA>(
            new TensorCUDA(repace_num(model_path+"encoder_layer_%d_ffn_fc_1.w_0", i))));
    }

    

}

TensorCUDA* Transformer::get_embedding_out(int* token_type_list,
                    int* role_ids_list,
                    int* sent_ids_list,
                    int length){


    TensorIntCUDA token_ids(token_type_list, length);
    TensorIntCUDA role_ids( role_ids_list, length);
    TensorIntCUDA sent_ids(sent_ids_list, length);

    TensorCUDA token_type_embedding({token_ids.get_size(), _word_embedding->get_dim()});
    TensorCUDA role_type_embedding({token_ids.get_size(), _word_embedding->get_dim()});
    TensorCUDA sent_type_embedding({token_ids.get_size(), _word_embedding->get_dim()});
    TensorCUDA* emb_out = new TensorCUDA({token_ids.get_size(), _word_embedding->get_dim()});

    _word_embedding->process(token_ids, token_type_embedding);
    _role_embedding->process(role_ids, role_type_embedding);
    _sent_embedding->process(sent_ids,sent_type_embedding);


    _mat_add_op->process(token_type_embedding, role_type_embedding, *emb_out);
    _mat_add_op->process(*emb_out, sent_type_embedding, *emb_out);
    return emb_out;
}

TensorCUDA* Transformer::get_pos_embedding_out(int* token_type_list, int length){
    TensorIntCUDA token_ids(token_type_list, length);
    TensorCUDA* pos_type_embedding= new TensorCUDA({length, _pos_embedding->get_dim()});
    _pos_embedding->process(token_ids, *pos_type_embedding);
    return pos_type_embedding;
}


void Transformer::get_q_k_v(TensorCUDA& tensor, TensorCUDA& pos_type_embedding, TensorCUDA& q, TensorCUDA& k, TensorCUDA& v,int layer_index,  KeyValueCache& key_value_cache){

    TensorDynamicCUDA* key_cache = new TensorDynamicCUDA(tensor.get_shape());
    TensorDynamicCUDA* value_cache = new TensorDynamicCUDA(tensor.get_shape());


    _matmulop->process(  tensor, *_query_w[layer_index],*key_cache );
    _rope_op->process(*key_cache,pos_type_embedding,*key_cache);
    key_cache->reshape_copy(q);
   

    _matmulop->process( tensor, *_key_w[layer_index], *key_cache);
    _rope_op->process(*key_cache,pos_type_embedding,*key_cache);
    key_cache->reshape_copy(k);


    _matmulop->process( tensor, *_value_w[layer_index], *value_cache);
    value_cache->reshape_copy(v);
    key_value_cache.add_cache(key_cache,value_cache);

}

Transformer::Transformer(const std::string& path, int layer_size){
    _layer_size = layer_size;
    init_cuda();
    _mat_add_op =  new MatAddOP();
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

 
}

void Transformer::get_attention_output(TensorCUDA& tensor,TensorCUDA& pos_type_embedding, int layer_index, KeyValueCache& key_value_cache){
     std::vector<int> head_shape = {32, tensor.get_shape()[0], tensor.get_shape()[1]/32};
    TensorCUDA q(head_shape );
    TensorCUDA k(head_shape ); 
    TensorCUDA v(head_shape ); //32*3*64
    get_q_k_v(tensor, pos_type_embedding, q, k, v,layer_index,key_value_cache);
    TensorCUDA p({q.get_shape()[0],q.get_shape()[1],q.get_shape()[1]});
    batch_matmul(q,k,p); 
    expf(p);
    TensorCUDA attention({q.get_shape()[0],q.get_shape()[1],q.get_shape()[2]}); //32*3*3
    batch_matmul_without_transpose(p,v,attention);
    TensorCUDA mean({q.get_shape()[0],q.get_shape()[1]});
    mat_3d_reduce_sum(p, mean);
    // transformer.
    batch_mat_divide_mat(attention,mean);
    q.reshape(tensor.get_shape());
    attention.reshape_copy(q);
    _matmulop->process(  q, *_output_w[layer_index], tensor);
}

void Transformer::encoder_layer(TensorCUDA& tensor,TensorCUDA& pos_type_embedding, int layer_index, KeyValueCache& key_value_cache){
    TensorCUDA result(tensor.get_shape());
    _reduce_2d_sum_op[layer_index]->prcocess(tensor, result);
    get_attention_output(result,pos_type_embedding,layer_index, key_value_cache);
    _mat_add_op->process(tensor, result,result);
    _pre_ffn_layer_norm[layer_index]->prcocess(result, tensor);
    TensorCUDA tmp1({result.get_shape()[0], result.get_shape()[1]*4});
    _matmulop->process(tensor, *_ffn0_weight[layer_index],tmp1);
    gelu(tmp1);
    _matmulop->process(tmp1,*_ffn1_weight[layer_index],tensor);
    _mat_add_op->process(tensor, result,tensor);
}

void Transformer::encode(TensorCUDA& tensor,TensorCUDA& pos_type_embedding,  KeyValueCache& key_value_cache){
    for(int i=0;i<_layer_size;i++){
        encoder_layer(tensor, pos_type_embedding, i,key_value_cache);
    }

    _post_encoder_layer_norm->prcocess(tensor, tensor);
}

void Transformer::decoder_layer(TensorCUDA& tensor,TensorCUDA& pos_type_embedding, int layer_index, KeyValueCache& key_value_cache){
    TensorCUDA result(tensor.get_shape());
    _reduce_2d_sum_op[layer_index]->prcocess(tensor, result);
    decoder_get_attention_output(result,pos_type_embedding,layer_index, key_value_cache);
    _mat_add_op->process(tensor, result,result);
    _pre_ffn_layer_norm[layer_index]->prcocess(result, tensor);
    TensorCUDA tmp1({result.get_shape()[0], result.get_shape()[1]*4});
    _matmulop->process(tensor, *_ffn0_weight[layer_index],tmp1);
    gelu(tmp1);
    _matmulop->process(tmp1,*_ffn1_weight[layer_index],tensor);
    _mat_add_op->process(tensor, result,tensor);
}

void Transformer::decode(TensorCUDA& tensor,TensorCUDA& pos_type_embedding,  KeyValueCache& key_value_cache){
    for(int i=0;i<_layer_size;i++){
        decoder_layer(tensor, pos_type_embedding, i,key_value_cache);
    }

    _post_encoder_layer_norm->prcocess(tensor, tensor);
}

void Transformer::decoder_get_attention_output(TensorCUDA& tensor,TensorCUDA& pos_type_embedding, int layer_index, KeyValueCache& key_value_cache){
    std::vector<int> head_shape = {32, tensor.get_shape()[0], tensor.get_shape()[1]/32};
    std::vector<int> kv_head_shape = {32, key_value_cache.get_step()+ tensor.get_shape()[0], tensor.get_shape()[1]/32};
    TensorCUDA q(head_shape );//32,346,64
    TensorCUDA k(kv_head_shape );
    TensorCUDA v(kv_head_shape ); //32,346*2,64
    decoder_get_q_k_v(tensor, pos_type_embedding, q, k, v,layer_index,key_value_cache);
    TensorCUDA p({q.get_shape()[0],q.get_shape()[1],k.get_shape()[1]}); //32,346,346*2

    batch_matmul(q,k,p);
    expf(p);
    TensorCUDA attention({q.get_shape()[0],p.get_shape()[1],v.get_shape()[2]}); //32,346,64
    batch_matmul_without_transpose(p,v,attention);
    TensorCUDA mean({p.get_shape()[0],p.get_shape()[1]});
    mat_3d_reduce_sum(p, mean);
    // transformer.
    batch_mat_divide_mat(attention,mean);
    q.reshape(tensor.get_shape());
    attention.reshape_copy(q);
    _matmulop->process(  q, *_output_w[layer_index], tensor);
}

void Transformer::decoder_get_q_k_v(TensorCUDA& tensor, TensorCUDA& pos_type_embedding, TensorCUDA& q, TensorCUDA& k, TensorCUDA& v,int layer_index,  KeyValueCache& key_value_cache){
    TensorCUDA tmp(tensor.get_shape());
    _matmulop->process(  tensor, *_query_w[layer_index],tmp);
    _rope_op->process(tmp,pos_type_embedding,tmp);
    tmp.reshape_copy(q);
   
    _matmulop->process( tensor, *_key_w[layer_index], tmp);
    _rope_op->process(tmp,pos_type_embedding,tmp);
    TensorCUDA* key_cache = key_value_cache.incr_key_cache(tmp, layer_index);
    key_cache->reshape_copy(k);



    _matmulop->process( tensor, *_value_w[layer_index], tmp);
    key_cache = key_value_cache.incr_value_cache(tmp, layer_index);
    key_cache->reshape_copy(v);

}


int Transformer::predict_last_token(TensorCUDA& tensor){
    return get_last_token(*_word_embedding->get_embedding(), tensor);
}