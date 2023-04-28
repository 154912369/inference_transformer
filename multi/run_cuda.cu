#include "run_cuda.h"
#include <string>


#include "nccl.h"
#include <vector>
#include "multi/run_cuda.h"
#include "multi/synchronize_cuda.h"
#include "network/network.h"




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


void run_cuda(int myRank, int nRanks, int localRank, SynChronize* synronize){
    std::string model_path = "/data1/renweijie/baidu/dialogue/nlg-paddle-inference/params/";
    Transformer transformer(model_path, 1,synronize,myRank,nRanks,localRank);
    TensorCUDA*  enc_out;
    if(myRank==0){
      std::vector<int> token_ids_int = {1,2,3};
      std::vector<int> role_ids_int= {0,1,0};
      std::vector<int> pos_ids_int={0,1,2};
      std::vector<int> sent_ids_int={0,0,0};
      enc_out = transformer.get_embedding_out(
              token_ids_int.data(), role_ids_int.data(), sent_ids_int.data(),  token_ids_int.size()
          );
    }else{
      int* tmp;
      enc_out = transformer.get_embedding_out(tmp, tmp, tmp,  0);
    }

    

    

}
