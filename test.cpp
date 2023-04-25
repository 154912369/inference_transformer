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
    int size = 257;
    Transformer transformer(model_path, 1);

    // pre layer
    auto enc_out = transformer.get_embedding_out(
        token_ids_int.data(), role_ids_int.data(), sent_ids_int.data(), 257
    );
    TensorCUDA* pos_type_embedding = transformer.get_pos_embedding_out(pos_ids_int.data(), pos_ids_int.size());
    KeyValueCache key_value_cache;
    transformer.encode(*enc_out,*pos_type_embedding, key_value_cache);

    TensorCUDA tmp(std::string("/data1/renweijie/baidu/dialogue/nlg-paddle-inference/transfer_output/enc_output_first_step"));
    int tokenize_id = transformer.predict_last_token(*enc_out);
    printf("predict last token : %d ", tokenize_id);
    int role_id = 1;
    int sentence_id = 0;
    while (size<270){
        auto enc_out1 = transformer.get_embedding_out(
            &tokenize_id, &role_id, &sentence_id, 1
        );
        TensorCUDA* pos_type_embedding1 = transformer.get_pos_embedding_out(&size, 1);
        transformer.decode(*enc_out1,*pos_type_embedding1, key_value_cache);
        tokenize_id = transformer.predict_last_token(*enc_out);
        printf("%d ",tokenize_id);
        size+=1;
    }
    

    // TensorCUDA tmp1(std::string("/data1/renweijie/baidu/dialogue/nlg-paddle-inference/transfer_output/enc_output_second_step"));
    // if( enc_out1->equal(tmp)){
    //     printf("embedding 11 equal\n");
    // }else{
    //     printf("embedding is not equal\n");
    // }

    enc_out->print();
    

    return 0;
}
