#include <semaphore.h>
#include <unistd.h>
#include <stdio.h>
#include <stdlib.h>
#include <fcntl.h>
#include "multi/run_cuda.h"
#include <sys/ipc.h>
#include <sys/shm.h>
#include "multi/synchronize_cpu.h"

int main() {
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
    int myRank, nRanks=4, localRank = 4;
    __pid_t cpids[nRanks-1];
    int pid=1;
    for(int i=0;i<nRanks-1;i++){
        if(pid!=0){
            pid =fork();
            cpids[i]=pid;
            if(pid==0){
                myRank=i+1;
            }
        }
    }
    SynChronize* synronize = new SynChronize(shared_mem, cpids, nRanks-1,myRank==0);
    printf("my pid is %d, my rank is %d, parent pid is %d\n", getpid(),  myRank, getppid());
    run_cuda(myRank, nRanks, myRank,synronize);
    return 0;
}