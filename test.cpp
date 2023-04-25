#include "network/network.h"
#include "common/cache.h"
#include "common/time.cpp"
#include "unistd.h"
#include <vector>
// #include "input.cpp"
#include "cuda_op/linear.h"
#include "cuda_op/common.h"
#include <unordered_map>
#include "common/string_utils.h"
std::string model_path = "/data1/renweijie/baidu/dialogue/nlg-paddle-inference/params/";
std::string root_path = "/data1/renweijie/baidu/dialogue/nlg-paddle-inference/transfer_output/";

std::vector<int> token_ids_int = {1,999,2};
std::vector<int> role_ids_int= {0,1,1};
std::vector<int> pos_ids_int={0,1,2};
std::vector<int> sent_ids_int={0,0,0};
int main(int argc, char* argv[]){
    std::unordered_map<int, std::string> words;
    std::unordered_map<std::string, int> word2item;
    std::string line;
    std::fstream file("/data1/renweijie/baidu/dialogue/nlg-paddle-inference/projects/TOD-API/vocab_32.addunk.txt");
    while (getline(file,line)){
        std::vector<std::string> tokens = split(line,'\t');
        if(tokens.size()!=2){
            printf("line %s is not equal to 2\n",line.c_str());
        }
        auto& word = tokens[0];
        trim(word);
        int i = std::stoi(tokens[1]);
        // if(word2item.size()!=i){
        //     printf("line %s number is not equal to 2\n",line.c_str());
        // }
        word2item[word] =i;
        words[i]=word;
    }


    int size = role_ids_int.size();
    size= 3;
    Transformer transformer(model_path, 32);
    for(int i=0;i<size;i++){
        printf("%s\n", words[token_ids_int[i]].c_str());
    }
    // pre layer
    auto enc_out = transformer.get_embedding_out(
        token_ids_int.data(), role_ids_int.data(), sent_ids_int.data(), size
    );
    TensorCUDA* pos_type_embedding = transformer.get_pos_embedding_out(pos_ids_int.data(), size);
    KeyValueCache key_value_cache;
    transformer.encode(*enc_out,*pos_type_embedding, key_value_cache);



    int tokenize_id = 1;
    // printf("predict last token : %d %s\n", tokenize_id,words[tokenize_id].c_str());
    int role_id = 0;
    int sentence_id = 1;
    while (size<10){
        auto enc_out1 = transformer.get_embedding_out(
            &tokenize_id, &role_id, &sentence_id, 1
        );
        TensorCUDA* pos_type_embedding1 = transformer.get_pos_embedding_out(&size, 1);
        transformer.decode(*enc_out1,*pos_type_embedding1, key_value_cache);
        tokenize_id = transformer.predict_last_token(*enc_out1);
        printf("%d %s\n",tokenize_id, words[tokenize_id].c_str());
        size+=1;
    }
    
    // auto enc_out1 = transformer.get_embedding_out(
    //         token_ids_int.data(), role_ids_int.data(), sent_ids_int.data(), size
    //     );
    // TensorCUDA* pos_type_embedding1 = transformer.get_pos_embedding_out(pos_ids_int.data(), size);
    // transformer.decode(*enc_out1,*pos_type_embedding1, key_value_cache);
    // TensorCUDA tmp1(std::string("/data1/renweijie/baidu/dialogue/nlg-paddle-inference/transfer_output/prefix"));
    // if( enc_out1->equal(tmp1)){
    //     printf("embedding 11 equal\n");
    // }else{
    //     printf("embedding is not equal\n");
    // }

    // enc_out->print();
    

    return 0;
}
