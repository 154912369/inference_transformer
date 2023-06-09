#include "multi/synchronize_cpu.h"
#include<unistd.h>  
#include <cstdio>
#include <signal.h>
#include<stdlib.h>

SynChronize::SynChronize(int* shaired_memory, __pid_t* cpid, int cpid_size,bool  is_parent){
    mem = shaired_memory;
    size = 0;
    input_size = 0;
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
}

bool SynChronize::wait_for_input(){
    input_size += 1;
    while(input_size!=mem[2]){
        sleep(0.1);
        
    }
    if(mem[3]!=reset_size){
        reset_size+=1;
        return true;
    }else{
        return false;
    }
}

void SynChronize::sync_get_input(){
    input_size += 1;
    mem[2] += 1;
}
void SynChronize::reset_cache(){
    mem[3] += 1;
    reset_size+=1;
 }