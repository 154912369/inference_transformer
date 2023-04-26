#include "network/network.h"
#include "common/cache.h"
#include "common/time.cpp"
#include "unistd.h"
#include <vector>
#include "cuda_op/linear.h"
#include "cuda_op/common.h"
#include <unordered_map>
#include "common/string_utils.h"
#include "sentencepiece_processor.h"







int main(int argc, char* argv[]){
    sentencepiece::SentencePieceProcessor processor;
    // std::string model_path = "/data1/renweijie/baidu/dialogue/nlg-paddle-inference/params/";
    // auto status = processor.Load("/data1/renweijie/baidu/dialogue/nlg-paddle-inference/projects/TOD-API/spm.model");
    // std::fstream file("/data1/renweijie/baidu/dialogue/nlg-paddle-inference/projects/TOD-API/vocab_32.addunk.txt");

    std::string model_path = argv[1];
    auto status = processor.Load(argv[2]);
    std::fstream file(argv[3]);
    if (!status.ok()) {
    std::cerr << status.ToString() << std::endl;
    // error
    }else{
        printf("load suee");
    }


    std::unordered_map<int, std::string> words;
    std::unordered_map<std::string, int> word2item;
    std::string line;
   
    std::string a("▁");
    printf("_ size is %d\n",a.size());
    while (getline(file,line)){
        std::vector<std::string> tokens = split(line,'\t');
        if(tokens.size()!=2){
            printf("line %s is not equal to 2\n",line.c_str());
        }
        auto& word = tokens[0];
        // trim(word);
        int i = std::stoi(tokens[1]);
        // if(word2item.size()!=i){
        //     printf("line %s number is not equal to 2\n",line.c_str());
        // }
        word2item[word] =i;
        if(word.find("▁")==0){
            word=word.substr(3);
        }
        if(word.size()>3 && word.substr(word.size() - 4)=="▁"){
            word=word.substr(0,word.size() - 3);
        }
        words[i]=word;
    }


    Transformer transformer(model_path, 32);

    std::vector<int> token_ids_int = {1};
    std::vector<int> role_ids_int= {0};
    std::vector<int> pos_ids_int={0};
    std::vector<int> sent_ids_int={0};
    int size = 1;
    std::string input_str;
    std::string result;
    KeyValueCache key_value_cache;
    bool first = true;
    cublasHandle_t handle;
    cublasCreate(&handle);
    while(true){
        std::vector<std::string> pieces;
        if(!first){
            std::cout << "bot: "<<result.c_str()<<std::endl;
            std::cout << "human: ";
            result="";
        }else{
            std::cout << "human: ";
        }
        std::getline(std::cin, input_str);
        if(input_str.size()==1&&input_str[0]=='n'){
            std::string input_str;
            std::string result;
            token_ids_int = {1};
            role_ids_int= {0};
            pos_ids_int={0};
            sent_ids_int={0};
            first = true;
            key_value_cache.reset();
            continue;
        }
        processor.Encode(input_str, &pieces);

        token_ids_int = {1};
        role_ids_int= {0};
        pos_ids_int={0};
        sent_ids_int={0};
        for (std::string &token : pieces) {
            if(word2item.find(token)!=word2item.end()){
                token_ids_int.push_back(word2item[token]);

            }else{
                token_ids_int.push_back(word2item[std::string("<unk>")]);
            }
            pos_ids_int.push_back(size);
            sent_ids_int.push_back(0);
            role_ids_int.push_back(1);
            size += 1;

        }
        token_ids_int.push_back(2);
        pos_ids_int.push_back(size);
        sent_ids_int.push_back(0);
        role_ids_int.push_back(1);
        size += 1;

        // pre layer
        auto enc_out = transformer.get_embedding_out(
            token_ids_int.data(), role_ids_int.data(), sent_ids_int.data(),  token_ids_int.size()
        );
        TensorCUDA* pos_type_embedding = transformer.get_pos_embedding_out(pos_ids_int.data(),  token_ids_int.size());
        if(first){
            transformer.encode(*enc_out,*pos_type_embedding, key_value_cache, handle);
            first=false;
        }else{
            transformer.decode(*enc_out,*pos_type_embedding, key_value_cache, handle);
        }
        delete pos_type_embedding;
        delete enc_out;
        // std::shared_ptr<mt::text_process_v2::TextProcessor> text_processor = std::make_shared<mt::text_process_v2::TextProcessor>();


        int tokenize_id = 1;
        int role_id = 0;
        int sentence_id = 1;
        int answer_size= 0 ;
        while (tokenize_id!=2&&answer_size<30){
            auto enc_out1 = transformer.get_embedding_out(
                &tokenize_id, &role_id, &sentence_id, 1
            );
            TensorCUDA* pos_type_embedding1 = transformer.get_pos_embedding_out(&size, 1);
            transformer.decode(*enc_out1,*pos_type_embedding1, key_value_cache, handle);
            tokenize_id = transformer.predict_last_token(*enc_out1, handle);
            result+=words[tokenize_id];
            size+=1;
            answer_size+=1;
            delete pos_type_embedding1;
            delete enc_out1;
        }
        // printf("result: %s", result.c_str());
    }
    cublasDestroy(handle);
    

    

    return 0;
}
