#include "run_cuda.h"
#include <string>


#include "nccl.h"
#include <vector>
#include "multi/run_cuda.h"
#include "multi/synchronize_cuda.h"
#include "network/network.h"
#include <unordered_map>
#include "common/string_utils.h"
#include "common/input.h"




#define CUDACHECK(cmd) do {                         \
  cudaError_t e = cmd;                              \
  if( e != cudaSuccess ) {                          \
    printf("Failed: Cuda error %s:%d '%s'\n",             \
        __FILE__,__LINE__,cudaGetErrorString(e));   \
    exit(EXIT_FAILURE);                             \
  }                                                 \
} while(0)


#define NCCLCHECK(cmd) do {                         \
  ncclResult_t r = cmd;                             \
  if (r!= ncclSuccess) {                            \
    printf("Failed, NCCL error %s:%d '%s'\n",             \
        __FILE__,__LINE__,ncclGetErrorString(r));   \
    exit(EXIT_FAILURE);                             \
  }                                                 \
} while(0)


void run_cuda(int myRank, int nRanks, int localRank, SynChronize* synronize,char* argv[]){
    std::string model_path = argv[1];
    Transformer transformer(model_path, 32,synronize,myRank,nRanks,localRank);
    Input input(argv[2]);
    std::fstream file(argv[3]);



    std::unordered_map<int, std::string> words;
    std::unordered_map<std::string, int> word2item;
    std::string line;
   
    std::string a("▁");
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


    // Transformer transformer(model_path, 32);

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
        if(myRank==0){
          if(!first){
              std::cout << "bot: "<<result.c_str() << std::endl;
              std::cout << "human: ";
              result="";
          }else{
              std::cout << "human: ";
          }
          first = input.get_input(pieces);
          if(pieces.size()==1&&pieces[0][0]=='n'){
              std::string input_str;
              std::string result;
              token_ids_int = {1};
              role_ids_int= {0};
              pos_ids_int={0};
              sent_ids_int={0};
              first = true;
              key_value_cache.reset();
              synronize->reset_cache();
              continue;
          }
            synronize->sync_get_input();
        }else{
            if(synronize->wait_for_input()){
                std::string input_str;
                std::string result;
                token_ids_int = {1};
                role_ids_int= {0};
                pos_ids_int={0};
                sent_ids_int={0};
                first = true;
                key_value_cache.reset();
            }
        }
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
        transformer.sync_length(token_ids_int.size());
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

        int tokenize_id = 1;
        int role_id = 0;
        int sentence_id = 1;
        int answer_size= 0 ;
        transformer.sync_length(1);
        while (tokenize_id!=2&&answer_size<30){
            auto enc_out1 = transformer.get_embedding_out(
                &tokenize_id, &role_id, &sentence_id, 1
            );
            TensorCUDA* pos_type_embedding1 = transformer.get_pos_embedding_out(&size, 1);
            // transformer.decode(*enc_out,*pos_type_embedding, key_value_cache, handle);
            TensorCUDA tmp2("/data1/renweijie/baidu/dialogue/nlg-paddle-inference/transfer_output/tmp_attention");
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
    
    

    

}
