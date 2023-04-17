#include "network/network.h"

void Transformer::init_root(const std::string& model_path){
    _word_embedding = new EmbeddingOP(model_path+"word_embedding");
    _sent_embedding = new EmbeddingOP(model_path+"sent_embedding");
    _role_embedding = new EmbeddingOP(model_path+"role_embedding");
    _pos_embedding = new EmbeddingOP(model_path+"pos_embedding",1,2,32);
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

Transformer::Transformer(const std::string& path){
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
}