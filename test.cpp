#include "network/network.h"
#include "common/time.cpp"
#include "unistd.h"
#include <vector>
#include "input.cpp"
std::string model_path = "/data1/renweijie/baidu/dialogue/nlg-paddle-inference/params/";
std::string root_path = "/data1/renweijie/baidu/dialogue/nlg-paddle-inference/transfer_output/";

int main(int argc, char* argv[]){

    Transformer transformer(model_path);
    auto result = transformer.get_embedding_out(
        token_ids_int.data(), role_ids_int.data(), sent_ids_int.data(), sent_ids_int.size()
    );

    TensorCUDA result1(root_path+"emb_out");

    LayerNormlizeOP reduce_2d_sum_op(model_path+"encoder_layer_0_pre_att_layer_norm_scale");
    reduce_2d_sum_op.prcocess(result1);
    TensorCUDA tmp_bias(root_path+"pre_process_layer_result_first");
    if( tmp_bias.equal(result1)){
        printf("embedding is equal\n");
    }else{
        printf("embedding is not equal\n");
    }

    // result1.print();
    // tmp_bias.print();



    // if(result->equal(result1)){
    //     printf("embedding is equal\n");
    // }else{
    //     printf("embedding is not equal\n");
    // }

    // result = transformer.get_pos_embedding_out(
    //     pos_ids_int.data(),  sent_ids_int.size()
    // );
    // TensorCUDA result2(root_path+"pos_emb_out");
    // if(result->equal(result2)){
    //     printf("pos_embedding is equal\n");
    // }else{
    //     printf("pos_embedding is not equal\n");
    // }
  




    return 0;
}
