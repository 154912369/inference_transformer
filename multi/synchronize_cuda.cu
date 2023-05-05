#include "multi/synchronize_cuda.h"
#include <iostream>
#include <fstream>
#include <unistd.h>
#include <stdint.h>
#include <sstream>
char* common_path = "./log/cuda_id.txt";
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
SynchronizeCUDA::SynchronizeCUDA(int myRank, int nRanks, int localRank){
   _myRank=myRank;
   _nRanks = nRanks;
   _localRank = localRank;
}
void SynchronizeCUDA::syn(){
    std::ofstream file_result(("./log/work."+std::to_string(_myRank)+".txt").c_str());
    if(file_result.is_open()) {
        
        file_result<<getpid();
        file_result.close();
    } 
    if(_myRank==0){
        ncclGetUniqueId(&_id);
        std::ofstream file(common_path);
        if(file.is_open()) {
           
            file << getppid();
            file<<"\n";
            file.write ((char *) _id.internal, sizeof(_id.internal) * sizeof(char));
            file.close();
        } else{
            printf("open failure as expected: %s\n\n\n\n\n\n",strerror(errno));
        
        }
        printf("main rank %d syncronize nccl id size is %d\n", _myRank, sizeof(_id.internal));
         sleep(1);
    }else{
        sleep(1);
        int ppid = getppid();
        int correct_pid = getpid();
        std::string line;
        int pid;
        int try_size = 1;
        while(pid!=ppid&&try_size<10&&correct_pid!=pid){
            std::ifstream file(common_path);
            if(file.is_open()) {
                getline(file, line);
                if(line.size()>0){
                    pid = std::stoi(line);
                    printf("get parent pid is %d ppid is %d\n",pid,ppid);
                    std::stringstream buffer;
                    buffer << file.rdbuf();
                    line = buffer.str();
                    sleep(1);
                    try_size +=1;
                }

            }
            file.close();
        }
        const char* result = line.c_str();
        for(int i=0;i<line.size();i++){
            _id.internal[i]=result[i];
        }
        printf("rank %d syncronize nccl id size is %d\n", _myRank, line.c_str());

    }
    //picking a GPU based on localRank, allocate device buffers
    printf("local rank is %d, _myRank is %d, ranks is %d\n", _localRank, _myRank, _nRanks);
    CUDACHECK(cudaSetDevice(_localRank));
      //initializing NCCL
    NCCLCHECK(ncclCommInitRank(&_comm, _nRanks, _id, _myRank));
    CUDACHECK(cudaStreamCreate(&_stream));

}

ncclUniqueId& SynchronizeCUDA::getNcclId(){
    return _id;
}

int SynchronizeCUDA::get_rank(){
    return _myRank;
}

int SynchronizeCUDA::get_rank_size(){
    return _nRanks;
}


SynchronizeCUDA::~SynchronizeCUDA(){
      //finalizing NCCL
      if(_nRanks>1){
          ncclCommDestroy(_comm);
      }

  printf("start to destroy rank %d\n",_myRank);
}

void SynchronizeCUDA::AllReduce(const void* sendbuff, void* recvbuff, size_t count,
    ncclDataType_t datatype, ncclRedOp_t op){

  //communicating using NCCL
  NCCLCHECK(ncclAllReduce(sendbuff, recvbuff, count,  datatype, op, _comm, _stream));
  //completing NCCL operation by synchronizing on the CUDA stream
  CUDACHECK(cudaStreamSynchronize(_stream));

}

void SynchronizeCUDA::BroadCast(const void* sendbuff, void* recvbuff, size_t count, 
    ncclDataType_t datatype, int device){
    if(_nRanks==1)return;

  //communicating using NCCL
  NCCLCHECK(ncclBroadcast(sendbuff, recvbuff, count,  datatype, device, _comm, _stream));
  //completing NCCL operation by synchronizing on the CUDA stream
  CUDACHECK(cudaStreamSynchronize(_stream));

}


void SynchronizeCUDA::AllGather(const void* sendbuff, void* recvbuff, size_t count, 
    ncclDataType_t datatype){
    if(_nRanks==1)return;

  //communicating using NCCL
  NCCLCHECK(ncclAllGather(sendbuff, recvbuff, count,  datatype,  _comm, _stream));
  //completing NCCL operation by synchronizing on the CUDA stream
  CUDACHECK(cudaStreamSynchronize(_stream));

}
