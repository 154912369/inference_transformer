#include "multi/synchronize_cpu.h"
#include<unistd.h>  
#include <cstdio>
#include <signal.h>
#include<stdlib.h>

SynChronize::SynChronize(int* shaired_memory, __pid_t* cpid, int cpid_size,bool  is_parent){
    mem = shaired_memory;
    size = 0;
    _cpid_size = cpid_size;
    _cpid=cpid;
    _is_parent = is_parent;

}
int SynChronize::get_length(){
    return *mem;
}

void SynChronize::set_length(int length){
    mem[0]=length;
    mem[1]+=1;
    size+=1;
}

int SynChronize::get_synchronize_length(){
    size+=1;
    while(size!=mem[1]){
        sleep(1);
        
    }
    return *mem;
}

void SynChronize::stop_subprocess(){
    sleep(1);
    if(_is_parent){
        for(int i=0;i<_cpid_size;i++){
            kill(_cpid[i], SIGKILL);
            printf("process %d stop child pid %d\n",getpid(), _cpid[i]);
        }
    }
    exit(0);

}