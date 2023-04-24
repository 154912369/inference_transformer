#include "network/network.h"
#include "common/cache.h"
#include "common/time.cpp"
#include "unistd.h"
#include <vector>
#include "input.cpp"
#include "cuda_op/linear.h"
#include "cuda_op/common.h"
std::string model_path = "/data1/renweijie/baidu/dialogue/nlg-paddle-inference/params/";
std::string root_path = "/data1/renweijie/baidu/dialogue/nlg-paddle-inference/transfer_output/";


int main(int argc, char* argv[]){

    Transformer transformer(model_path, 32);

    // pre layer
    auto enc_out = transformer.get_embedding_out(
        token_ids_int.data(), role_ids_int.data(), sent_ids_int.data(), sent_ids_int.size()
    );
    TensorCUDA* pos_type_embedding = transformer.get_pos_embedding_out(pos_ids_int.data(), pos_ids_int.size());
    KeyValueCache key_value_cache;
    transformer.encode(*enc_out,*pos_type_embedding, key_value_cache);
    transformer.decode(*enc_out,*pos_type_embedding, key_value_cache);
    TensorCUDA tmp(std::string("/data1/renweijie/baidu/dialogue/nlg-paddle-inference/transfer_output/enc_output"));
    if( enc_out->equal(tmp)){
        printf("embedding 10 equal\n");
    }else{
        printf("embedding is not equal\n");
    }
    enc_out->print();
    

    return 0;
}
