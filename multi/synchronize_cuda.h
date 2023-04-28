#pragma once
#include "nccl.h"
class SynchronizeCUDA{
   
    int _myRank, _nRanks = 4, _localRank = 4;
    ncclUniqueId _id;
    ncclComm_t _comm ;
    cudaStream_t _stream;
public:
     SynchronizeCUDA(int myRank, int nRanks, int localRank);
     ~SynchronizeCUDA();
    void syn();
    ncclUniqueId& getNcclId();
    int get_rank();
    void AllReduce(const void* sendbuff, void* recvbuff, size_t count,
    ncclDataType_t datatype, ncclRedOp_t op);
    void BroadCast(const void* sendbuff, void* recvbuff, size_t count, 
    ncclDataType_t datatype);
};

