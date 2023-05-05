#include <semaphore.h>
#include <unistd.h>
#include <stdio.h>
#include <stdlib.h>
#include <fcntl.h>
#include "multi/run_cuda.h"
#include <sys/ipc.h>
#include <sys/shm.h>
#include "multi/synchronize_cpu.h"
#include <cstdlib>
#include <cstring>
#include <iostream>
#include <vector>
#include <sstream>
#include <string>
SynChronize* synronize;
void signalHandler( int signum ){
    printf("Interrupt signal ( %d ) received.\n",signum );
 
    // 清理并关闭
    // 终止程序  
    synronize->stop_subprocess();
   exit(signum);  
 
}
int main(int argc, char* argv[]) {
        // 创建共享内存，大小为100字节
    int shmid = shmget(IPC_PRIVATE, 100, IPC_CREAT | 0666);
    if (shmid == -1) {
        printf("Error: Failed to create shared memory." );
        return -1;
    }

    // 将共享内存映射到当前进程的地址空间
    int *shared_mem = (int*) shmat(shmid, 0, 0);
    
    shared_mem[0]=0;
    shared_mem[1]=0;
    int myRank, nRanks=4, localRank = nRanks;
    const char* env_var = std::getenv("CUDA_VISIBLE_DEVICES");
    if (env_var != nullptr) {
        std::string env_var_str(env_var);
        std::stringstream ss(env_var_str);
        std::string token;
        std::vector<int> devices;
        while (std::getline(ss, token, ',')) {
            devices.push_back(std::stoi(token));
        }
        nRanks =devices.size();
        // localRank = devices[myRank];
    }
    

    __pid_t cpids[nRanks-1];
    int pid=1;
    for(int i=0;i<nRanks-1;i++){
        if(pid!=0){
            printf(" %d fork\n",getpid());
            pid =fork();
            cpids[i]=pid;
            if(pid==0){
                myRank=i+1;
            }
        }
    }
    if(myRank==0||myRank==3){
        if(nRanks!=1){
             myRank = 3-myRank;
        }
       
    }
    localRank = myRank;
    
    synronize = new SynChronize(shared_mem, cpids, nRanks-1, myRank==0);
    printf("my pid is %d, my rank is %d, parent pid is %d\n", getpid(),  myRank, getppid());
    run_cuda(myRank, nRanks, localRank,synronize,argv);
    return 0;
}